# ruff: noqa: N803, N806
"""Triton 2D Hilbert encoder (4-bit compacted LUT)."""

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.torch._luts import TorchCacheMode, lut_2d4b_b_qs_i64


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


def _choose_launch_config(n_elements: int, *, shmem_lut: bool) -> tuple[int, int]:
    """Choose a reasonable default without autotune.

    Returns
    -------
    block_size: int
    num_warps: int
    """
    if not shmem_lut:
        return 256, 4

    if n_elements <= 131072:
        return 256, 4
    return 512, 4


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

    lut = lut_2d4b_b_qs_i64(device=x.device, cache=lut_cache)

    # Triton < 3.3.0 does not have tl.gather, fall back to LUT in global memory.
    # This is still performant due to caching.
    load_lut_into_shared_memory = True if hasattr(tl, "gather") else False
    block_size, num_warps = _choose_launch_config(
        n_elements,
        shmem_lut=load_lut_into_shared_memory,
    )

    hilbert_encode_2d_4bit_compacted_qs[grid](  # type: ignore[reportIndexIssue]
        x,
        y,
        out,
        n_elements,
        lut,
        BLOCK_SIZE=block_size,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
        num_warps=num_warps,  # type: ignore[reportCallIssue]
    )

    return out
