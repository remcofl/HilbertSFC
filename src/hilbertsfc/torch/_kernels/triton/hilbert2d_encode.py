# ruff: noqa: N803, N806
"""Triton 2D Hilbert encoder (4-bit compacted LUT)."""

import os

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.torch._luts import (
    TorchCacheMode,
    lut_2d4b_b_qs_i64,
    lut_2d4b_sb_sq_i16,
    lut_2d7b_sb_sq_i16,
)

# Toggle between 16-bit state-aware LUT path and 64-bit compacted LUT path.
USE_16BIT_2D_ENCODE_KERNEL = True
# When using the 16-bit path, select 7-bit tile variant instead of 4-bit.
USE_7BIT_2D_ENCODE_16BIT_KERNEL = False

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
_printed_all_timing_keys: set[tuple[str, int, int, bool]] = set()


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_elements", "NBITS", "SHMEM_LUT"],
)
@triton.jit
def hilbert_encode_2d_4bit_compacted_qs(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Encode 2D coordinates (x, y) to a Hilbert index."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x3
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 4
    COORD_MASK: tl.constexpr = (1 << NBITS) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]
    OUT_DTYPE: tl.constexpr = out_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    # Optional LUT preload.
    if SHMEM_LUT:
        lut_local = tl.load(lut_ptr + tl.arange(0, 256), eviction_policy="evict_last")
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    if not SHMEM_LUT:
        # 8 bit tile packing (x and y) can result in negative values when using int8 inputs,
        # so we convert to uint8.
        if x.dtype == tl.int8:
            x = x.to(tl.uint8)
        if y.dtype == tl.int8:
            y = y.to(tl.uint8)

    if DROP_BITS > 0:
        x = x & COORD_MASK
        y = y & COORD_MASK

    idx = tl.zeros([BLOCK_SIZE], dtype=OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -4):
        b_x = (x >> bit) & 0xF
        b_y = (y >> bit) & 0xF
        b = (b_x << 4) | b_y

        if SHMEM_LUT:
            # Handles negative b values when indexing with int8 inputs.
            lut_val = tl.gather(lut_local, b, axis=0)
        else:
            lut_val = tl.load(lut_ptr + b, eviction_policy="evict_last")

        qs = lut_val >> state
        q = (qs & 0xFF00) >> 8
        idx |= q << (bit * 2)
        state = qs & 0xFF

    tl.store(out_ptr + offsets, idx, mask=mask)


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_elements", "NBITS", "SHMEM_LUT"],
)
@triton.jit
def hilbert_encode_2d_4bit_sq(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Encode 2D coordinates (x, y) to a Hilbert index."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x3
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 4
    COORD_MASK: tl.constexpr = (1 << NBITS) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]
    OUT_DTYPE: tl.constexpr = out_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    # Optional LUT preload.
    if SHMEM_LUT:
        lut_local = tl.load(lut_ptr + tl.arange(0, 1024), eviction_policy="evict_last")
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    if not SHMEM_LUT:
        # 8 bit tile packing (x and y) can result in negative values when using int8 inputs,
        # so we convert to uint8.
        if x.dtype == tl.int8:
            x = x.to(tl.uint8)
        if y.dtype == tl.int8:
            y = y.to(tl.uint8)

    if DROP_BITS > 0:
        x = x & COORD_MASK
        y = y & COORD_MASK

    idx = tl.zeros([BLOCK_SIZE], dtype=OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -4):
        b_x = (x >> bit) & 0xF
        b_y = (y >> bit) & 0xF
        b = (b_x << 4) | b_y

        lut_idx = state | b

        if SHMEM_LUT:
            # Handles negative b values when indexing with int8 inputs.
            sq = tl.gather(lut_local, lut_idx, axis=0)
        else:
            sq = tl.load(lut_ptr + lut_idx, eviction_policy="evict_last")

        sq = sq.to(tl.int32)  # Considerable speed up

        q = sq & 0xFF
        idx |= q.to(OUT_DTYPE) << (bit << 1)
        state = sq & 0x300

    tl.store(out_ptr + offsets, idx, mask=mask)


def _choose_launch_config(n_elements: int, *, shmem_lut: bool) -> tuple[int, int]:
    """Choose a reasonable default without autotune.

    Returns
    -------
    block_size: int
    num_warps: int
    """
    if not shmem_lut:  # 128, 2 to 256, 1 at 4 million plus
        if n_elements <= 2 << 22:
            return 128, 2
        return 256, 1

    if n_elements <= 2 << 16:
        return 256, 4
    if n_elements <= 2 << 20:
        return 512, 8
    if n_elements <= 2 << 22:
        return 1024, 8
    return 2048, 8  # For 3d is seems better to exclude this last option


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_elements", "NBITS", "SHMEM_LUT"],
)
@triton.jit
def hilbert_encode_2d_7bit_sq(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Encode 2D coordinates using packed (state, b14) -> (state, q14) LUT."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x6
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 7
    COORD_MASK: tl.constexpr = (1 << NBITS) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]
    OUT_DTYPE: tl.constexpr = out_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    if SHMEM_LUT:
        lut_local = tl.load(lut_ptr + tl.arange(0, 65536), eviction_policy="evict_last")

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    if x.dtype == tl.int8 or x.dtype == tl.uint8:
        x = x.to(tl.uint16)
    if y.dtype == tl.int8 or y.dtype == tl.uint8:
        y = y.to(tl.uint16)

    if DROP_BITS > 0:
        x = x & COORD_MASK
        y = y & COORD_MASK

    idx = tl.zeros([BLOCK_SIZE], dtype=OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -7):
        b_x = (x >> bit) & 0x7F
        b_y = (y >> bit) & 0x7F
        b = (b_x << 7) | b_y

        lut_idx = state | b

        if SHMEM_LUT:
            sq = tl.gather(lut_local, lut_idx, axis=0)
        else:
            sq = tl.load(lut_ptr + lut_idx, eviction_policy="evict_last")

        q = sq & 0x3FFF
        idx |= q.to(OUT_DTYPE) << (bit << 1)
        state = sq & 0xC000

    tl.store(out_ptr + offsets, idx, mask=mask)


def hilbert_encode_2d_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
) -> torch.Tensor:
    """Encode (x, y) to Hilbert indices.

    - Does not require x/y to be uint64; any integer dtype is accepted.
    - Returns minimal uint dtype that can hold the index (based on 2*nbits).
    """
    validate_nbits_2d(nbits)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    n_elements = out.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if USE_16BIT_2D_ENCODE_KERNEL:
        if USE_7BIT_2D_ENCODE_16BIT_KERNEL:
            kernel = hilbert_encode_2d_7bit_sq
            kernel_name = "hilbert_encode_2d_7bit_sq"
            lut = lut_2d7b_sb_sq_i16(device=x.device, cache=lut_cache)
        else:
            kernel = hilbert_encode_2d_4bit_sq
            kernel_name = "hilbert_encode_2d_4bit_sq"
            lut = lut_2d4b_sb_sq_i16(device=x.device, cache=lut_cache)
    else:
        kernel = hilbert_encode_2d_4bit_compacted_qs
        kernel_name = "hilbert_encode_2d_4bit_compacted_qs"
        lut = lut_2d4b_b_qs_i64(device=x.device, cache=lut_cache)

    # Triton < 3.3.0 does not have tl.gather, fall back to LUT in global memory.
    # This is still performant due to caching.
    load_lut_into_shared_memory = True if hasattr(tl, "gather") else False
    if USE_16BIT_2D_ENCODE_KERNEL and USE_7BIT_2D_ENCODE_16BIT_KERNEL:
        # 65,536-entry int16 LUT is too large for practical shared-memory preload.
        load_lut_into_shared_memory = False

    kernel[grid](  # type: ignore[reportIndexIssue]
        x,
        y,
        out,
        n_elements,
        lut,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
    )

    if os.getenv("HILBERTSFC_TRITON_PRINT_ALL_AUTOTUNE_CONFIGS", "0") == "1":
        key = (
            str(x.device),
            int(n_elements),
            int(nbits),
            bool(load_lut_into_shared_memory),
        )
        if key not in _printed_all_timing_keys:
            timings = getattr(kernel, "configs_timings", None)
            if timings:
                _printed_all_timing_keys.add(key)
                print(
                    f"[triton.autotune] all configs {kernel_name} "
                    f"device={x.device} n_elements={n_elements} nbits={nbits} "
                    f"shmem_lut={load_lut_into_shared_memory}"
                )

                def _median_ms(v: object) -> float:
                    if isinstance(v, (tuple, list)) and len(v) > 0:
                        return float(v[0])
                    if isinstance(v, (int, float)):
                        return float(v)
                    return 0.0

                bytes_per_point = (
                    x.element_size() + y.element_size() + out.element_size()
                )
                ordered = sorted(timings.items(), key=lambda item: _median_ms(item[1]))
                best_ms = _median_ms(ordered[0][1])

                print("  median_us    delta      mpts/s      gb/s  config")

                for cfg, t in ordered:
                    t_ms = _median_ms(t)
                    delta_pct = (
                        0.0 if best_ms <= 0 else ((t_ms / best_ms) - 1.0) * 100.0
                    )
                    t_s = t_ms * 1e-3
                    mpts_s = float("inf") if t_s <= 0 else (n_elements / t_s) / 1e6
                    gb_s = (
                        float("inf")
                        if t_s <= 0
                        else (n_elements * bytes_per_point) / t_s / 1e9
                    )
                    print(
                        f"  {t_ms * 1e3:9.2f}  {delta_pct:+7.2f}%  "
                        f"{mpts_s:10.2f}  {gb_s:8.2f}  {cfg}"
                    )

    return out
