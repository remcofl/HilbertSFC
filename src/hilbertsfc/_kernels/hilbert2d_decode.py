"""2D decode kernel builders."""

from __future__ import annotations

import numba as nb
import numpy as np

from .._cache import kernel_cache
from .._luts import lut_2d4b_q_bs_u64
from .._nbits import validate_nbits_2d
from .._typing import UIntArray


@nb.njit(inline="always")
def _hilbert_decode_2d_4bit_compacted_bs(idx, nbits, lut):
    x = y = 0
    state = 0  # Start state is either 0 or 1 based on nbits parity
    start_bit = (nbits - 1) & ~0x3

    # Mask higher than nbits, use fact that q = 00, s=00 -> b=00
    drop_bits = start_bit - nbits + 4
    if drop_bits > 0:
        idx &= np.uint64((1 << (nbits << 1)) - 1)  # Merges with 0xFF mask

    # Process quadrants from MSB to LSB (..., 16, 8, 0)
    for bit in range(start_bit, -1, -4):
        q = (idx >> (bit << 1)) & 0xFF  # Extract the Hilbert quadrant (8 bits)
        bs = lut[q] >> state  # Also eliminates zext when idx uint64

        b_x = (bs & 0xF000) >> 12  # Free lshr
        b_y = (bs & 0x0F00) >> 8
        x |= b_x << bit  # Append 4 bits to x
        y |= b_y << bit  # Append 4 bits to y

        state = bs & 0xFF  # Advance state

    return x, y


@kernel_cache
def build_hilbert_decode_2d_impl(nbits: int):
    """Return a specialized scalar decoder: index -> (x, y)."""

    validate_nbits_2d(nbits)

    lut = lut_2d4b_q_bs_u64()

    @nb.njit(inline="always", cache=True)
    def decode_2d(index: int) -> tuple[int, int]:
        return _hilbert_decode_2d_4bit_compacted_bs(index, nbits, lut)

    return decode_2d


@kernel_cache
def build_hilbert_decode_2d_batch_impl(nbits: int, *, parallel: bool = False):
    """Return a specialized batch decoder: (indices, xs, ys) -> (xs, ys)."""

    validate_nbits_2d(nbits)

    decode_scalar = build_hilbert_decode_2d_impl(nbits)

    @nb.njit(parallel=parallel, cache=True)
    def decode_2d_batch(indices: UIntArray, xs: UIntArray, ys: UIntArray) -> None:
        n = indices.shape[0]
        for i in nb.prange(n):  # ty:ignore[not-iterable]
            xs[i], ys[i] = decode_scalar(indices[i])

    return decode_2d_batch
