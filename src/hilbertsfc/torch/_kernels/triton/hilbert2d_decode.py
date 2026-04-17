# ruff: noqa: N803, N806
"""Triton 2D Hilbert decoder (4-bit compacted LUT)."""

import os

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.torch._luts import TorchCacheMode, lut_2d4b_q_bs_i64

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


def _maybe_print_best_config(
    *,
    device: torch.device,
    n_elements: int,
    nbits: int,
    shmem_lut: bool,
) -> None:
    key = (str(device), int(n_elements), int(nbits), bool(shmem_lut))
    if key in _printed_best_configs:
        return

    best_cfg = getattr(hilbert_decode_2d_4bit_compacted_bs, "best_config", None)
    if best_cfg is None:
        return

    _printed_best_configs.add(key)
    print(
        "[triton.autotune] hilbert_decode_2d_4bit_compacted_bs "
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
) -> None:
    if os.getenv("HILBERTSFC_TRITON_PRINT_ALL_AUTOTUNE_CONFIGS", "0") != "1":
        return

    key = (str(device), int(n_elements), int(nbits), bool(shmem_lut))
    if key in _printed_all_timing_keys:
        return

    timings = getattr(hilbert_decode_2d_4bit_compacted_bs, "configs_timings", None)
    if not timings:
        return

    _printed_all_timing_keys.add(key)

    print(
        "[triton.autotune] all configs hilbert_decode_2d_4bit_compacted_bs "
        f"device={device} n_elements={n_elements} nbits={nbits} "
        f"shmem_lut={shmem_lut}"
    )

    def _median_ms(v: object) -> float:
        if isinstance(v, (tuple, list)) and len(v) > 0:
            return float(v[0])
        return float(v)

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

    lut = lut_2d4b_q_bs_i64(device=index.device, cache=lut_cache)

    # Triton < 3.3.0 does not have tl.gather, fall back to LUT in global memory.
    # This is still performant due to caching.
    load_lut_into_shared_memory = True if hasattr(tl, "gather") else False

    hilbert_decode_2d_4bit_compacted_bs[grid](  # type: ignore[reportIndexIssue]
        index,
        out_x,
        out_y,
        n_elements,
        lut,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
    )

    _maybe_print_best_config(
        device=index.device,
        n_elements=n_elements,
        nbits=nbits,
        shmem_lut=load_lut_into_shared_memory,
    )
    _maybe_print_all_config_timings(
        device=index.device,
        n_elements=n_elements,
        nbits=nbits,
        shmem_lut=load_lut_into_shared_memory,
        bytes_per_point=index.element_size() + out_x.element_size() + out_y.element_size(),
    )

    return out_x, out_y
