# ruff: noqa: N803, N806
"""Triton 2D Hilbert decoder (4-bit compacted LUT)."""

import os

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.torch._luts import (
    TorchCacheMode,
    lut_2d4b_q_bs_i64,
    lut_2d4b_sq_sb_i16,
    lut_2d7b_sq_sb_i16,
)

# Toggle between 16-bit state-aware LUT path and 64-bit compacted LUT path.
USE_16BIT_2D_DECODE_KERNEL = True
# When using the 16-bit path, select 7-bit tile variant instead of 4-bit.
USE_7BIT_2D_DECODE_16BIT_KERNEL = False

_printed_best_configs: set[tuple[str, int, int, bool]] = set()
_printed_all_timing_keys: set[tuple[str, int, int, bool]] = set()
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_elements", "NBITS", "SHMEM_LUT"],
)
@triton.jit
def hilbert_decode_2d_4bit_compacted_bs(
    idx_ptr: tl.tensor,
    out_x_ptr: tl.tensor,
    out_y_ptr: tl.tensor,
    n_elements: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Decode Hilbert indices to (x, y) using a 4-bit LUT."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x3
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 4
    IDX_MASK: tl.constexpr = (1 << (2 * NBITS)) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]

    X_OUT_DTYPE: tl.constexpr = out_x_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]
    Y_OUT_DTYPE: tl.constexpr = out_y_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    if SHMEM_LUT:
        lut_local = tl.load(lut_ptr + tl.arange(0, 256), eviction_policy="evict_last")

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = tl.load(idx_ptr + offsets, mask=mask, other=0)

    if not SHMEM_LUT:
        # 8 bit tile packing (x and y) can result in negative values when using int8 inputs,
        # so we convert to uint8.
        if idx.dtype == tl.int8:
            idx = idx.to(tl.uint8)

    if DROP_BITS > 0:
        idx = idx & IDX_MASK

    x = tl.zeros([BLOCK_SIZE], dtype=X_OUT_DTYPE)
    y = tl.zeros([BLOCK_SIZE], dtype=Y_OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -4):
        q = (idx >> (bit << 1)) & 0xFF

        if SHMEM_LUT:
            lut_val = tl.gather(lut_local, q, axis=0)
        else:
            lut_val = tl.load(lut_ptr + q, eviction_policy="evict_last")

        bs = lut_val >> state

        b_x = (bs & 0xF000) >> 12
        b_y = (bs & 0x0F00) >> 8
        x |= b_x << bit
        y |= b_y << bit

        state = bs & 0xFF

    tl.store(out_x_ptr + offsets, x, mask=mask)
    tl.store(out_y_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_elements", "NBITS", "SHMEM_LUT"],
)
@triton.jit
def hilbert_decode_2d_4bit_sb(
    idx_ptr: tl.tensor,
    out_x_ptr: tl.tensor,
    out_y_ptr: tl.tensor,
    n_elements: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Decode Hilbert indices to (x, y) using packed (state, q) LUT."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x3
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 4
    IDX_MASK: tl.constexpr = (1 << (2 * NBITS)) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]

    X_OUT_DTYPE: tl.constexpr = out_x_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]
    Y_OUT_DTYPE: tl.constexpr = out_y_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    if SHMEM_LUT:
        lut_local = tl.load(lut_ptr + tl.arange(0, 1024), eviction_policy="evict_last")

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = tl.load(idx_ptr + offsets, mask=mask, other=0)

    if not SHMEM_LUT:
        # 8 bit q extraction can be negative for int8 index inputs; reinterpret as uint8.
        if idx.dtype == tl.int8:
            idx = idx.to(tl.uint8)

    if DROP_BITS > 0:
        idx = idx & IDX_MASK

    x = tl.zeros([BLOCK_SIZE], dtype=X_OUT_DTYPE)
    y = tl.zeros([BLOCK_SIZE], dtype=Y_OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -4):
        q = (idx >> (bit << 1)) & 0xFF
        lut_idx = state | q

        if SHMEM_LUT:
            sb = tl.gather(lut_local, lut_idx, axis=0)
        else:
            sb = tl.load(lut_ptr + lut_idx, eviction_policy="evict_last")

        # Keep in int32 for masks/shifts; this is faster than narrow integer ops.
        sb = sb.to(tl.int32)

        b = sb & 0xFF
        b_x = (b >> 4) & 0xF
        b_y = b & 0xF
        x |= b_x.to(X_OUT_DTYPE) << bit
        y |= b_y.to(Y_OUT_DTYPE) << bit
        state = sb & 0x300

    tl.store(out_x_ptr + offsets, x, mask=mask)
    tl.store(out_y_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_elements", "NBITS", "SHMEM_LUT"],
)
@triton.jit
def hilbert_decode_2d_7bit_sq(
    idx_ptr: tl.tensor,
    out_x_ptr: tl.tensor,
    out_y_ptr: tl.tensor,
    n_elements: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Decode Hilbert indices using packed (state, q14) -> (state, b14) LUT."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x6
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 7
    IDX_MASK: tl.constexpr = (1 << (2 * NBITS)) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]

    X_OUT_DTYPE: tl.constexpr = out_x_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]
    Y_OUT_DTYPE: tl.constexpr = out_y_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    if SHMEM_LUT:
        lut_local = tl.load(lut_ptr + tl.arange(0, 65536), eviction_policy="evict_last")

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = tl.load(idx_ptr + offsets, mask=mask, other=0)

    if idx.dtype == tl.int8 or idx.dtype == tl.uint8:
        idx = idx.to(tl.uint16)

    if DROP_BITS > 0:
        idx = idx & IDX_MASK

    x = tl.zeros([BLOCK_SIZE], dtype=X_OUT_DTYPE)
    y = tl.zeros([BLOCK_SIZE], dtype=Y_OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -7):
        q = (idx >> (bit << 1)) & 0x3FFF
        lut_idx = state | q

        if SHMEM_LUT:
            sb = tl.gather(lut_local, lut_idx, axis=0)
        else:
            sb = tl.load(lut_ptr + lut_idx, eviction_policy="evict_last")

        b = sb & 0x3FFF
        b_x = (b >> 7) & 0x7F
        b_y = b & 0x7F
        x |= b_x.to(X_OUT_DTYPE) << bit
        y |= b_y.to(Y_OUT_DTYPE) << bit
        state = sb & 0xC000

    tl.store(out_x_ptr + offsets, x, mask=mask)
    tl.store(out_y_ptr + offsets, y, mask=mask)


def _maybe_print_best_config(
    *,
    device: torch.device,
    n_elements: int,
    nbits: int,
    shmem_lut: bool,
    kernel: object,
    kernel_name: str,
) -> None:
    key = (str(device), int(n_elements), int(nbits), bool(shmem_lut))
    if key in _printed_best_configs:
        return

    best_cfg = getattr(kernel, "best_config", None)
    if best_cfg is None:
        return

    _printed_best_configs.add(key)
    print(
        f"[triton.autotune] {kernel_name} "
        f"device={device} n_elements={n_elements} nbits={nbits} "
        f"shmem_lut={shmem_lut} best_config={best_cfg}"
    )


def _maybe_print_all_config_timings(
    *,
    device: torch.device,
    n_elements: int,
    nbits: int,
    shmem_lut: bool,
    bytes_per_point: int,
    kernel: object,
    kernel_name: str,
) -> None:
    if os.getenv("HILBERTSFC_TRITON_PRINT_ALL_AUTOTUNE_CONFIGS", "0") != "1":
        return

    key = (str(device), int(n_elements), int(nbits), bool(shmem_lut))
    if key in _printed_all_timing_keys:
        return

    timings = getattr(kernel, "configs_timings", None)
    if not timings:
        return

    _printed_all_timing_keys.add(key)

    print(
        f"[triton.autotune] all configs {kernel_name} "
        f"device={device} n_elements={n_elements} nbits={nbits} "
        f"shmem_lut={shmem_lut}"
    )

    def _median_ms(v: object) -> float:
        if isinstance(v, (tuple, list)) and len(v) > 0:
            return float(v[0])
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    ordered = sorted(timings.items(), key=lambda item: _median_ms(item[1]))
    best_ms = _median_ms(ordered[0][1])

    print("  median_us    delta      mpts/s      gb/s  config")

    for cfg, t in ordered:
        t_ms = _median_ms(t)
        delta_pct = 0.0 if best_ms <= 0 else ((t_ms / best_ms) - 1.0) * 100.0
        t_s = t_ms * 1e-3
        mpts_s = float("inf") if t_s <= 0 else (n_elements / t_s) / 1e6
        gb_s = float("inf") if t_s <= 0 else (n_elements * bytes_per_point) / t_s / 1e9
        print(
            f"  {t_ms * 1e3:9.2f}  {delta_pct:+7.2f}%  "
            f"{mpts_s:10.2f}  {gb_s:8.2f}  {cfg}"
        )


def hilbert_decode_2d_triton(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode indices to (x, y).

    - Does not require `index` to be uint64; any integer dtype is accepted.
    - Output buffers can be passed to control dtype and allocation.
    """

    validate_nbits_2d(nbits)

    if out_x is None:
        out_x = torch.empty_like(index, dtype=torch.int64)
    if out_y is None:
        out_y = torch.empty_like(index, dtype=torch.int64)

    n_elements = index.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if USE_16BIT_2D_DECODE_KERNEL:
        if USE_7BIT_2D_DECODE_16BIT_KERNEL:
            kernel = hilbert_decode_2d_7bit_sq
            kernel_name = "hilbert_decode_2d_7bit_sq"
            lut = lut_2d7b_sq_sb_i16(device=index.device, cache=lut_cache)
        else:
            kernel = hilbert_decode_2d_4bit_sb
            kernel_name = "hilbert_decode_2d_4bit_sq"
            lut = lut_2d4b_sq_sb_i16(device=index.device, cache=lut_cache)
    else:
        kernel = hilbert_decode_2d_4bit_compacted_bs
        kernel_name = "hilbert_decode_2d_4bit_compacted_bs"
        lut = lut_2d4b_q_bs_i64(device=index.device, cache=lut_cache)

    # Triton < 3.3.0 does not have tl.gather, fall back to LUT in global memory.
    # This is still performant due to caching.
    load_lut_into_shared_memory = True if hasattr(tl, "gather") else False
    if USE_16BIT_2D_DECODE_KERNEL and USE_7BIT_2D_DECODE_16BIT_KERNEL:
        # 65,536-entry int16 LUT is too large for practical shared-memory preload.
        load_lut_into_shared_memory = False

    kernel[grid](  # type: ignore[reportIndexIssue]
        index,
        out_x,
        out_y,
        n_elements,
        lut,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
    )

    # Debug autotune inspection uses Python attribute reflection on Triton kernel
    # objects, which breaks fullgraph tracing in torch.compile.
    if not torch.compiler.is_compiling():
        _maybe_print_best_config(
            device=index.device,
            n_elements=n_elements,
            nbits=nbits,
            shmem_lut=load_lut_into_shared_memory,
            kernel=kernel,
            kernel_name=kernel_name,
        )
        _maybe_print_all_config_timings(
            device=index.device,
            n_elements=n_elements,
            nbits=nbits,
            shmem_lut=load_lut_into_shared_memory,
            bytes_per_point=index.element_size()
            + out_x.element_size()
            + out_y.element_size(),
            kernel=kernel,
            kernel_name=kernel_name,
        )

    return out_x, out_y
