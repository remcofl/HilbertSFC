# ruff: noqa: N803, N806
"""Triton 2D Hilbert decoder (4-bit compacted LUT)."""

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.torch._luts import TorchCacheMode, lut_2d4b_q_bs_i64


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
    block_size, num_warps = _choose_launch_config(
        n_elements,
        shmem_lut=load_lut_into_shared_memory,
    )

    hilbert_decode_2d_4bit_compacted_bs[grid](  # type: ignore[reportIndexIssue]
        index,
        out_x,
        out_y,
        n_elements,
        lut,
        BLOCK_SIZE=block_size,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
        num_warps=num_warps,  # type: ignore[reportCallIssue]
    )

    return out_x, out_y
