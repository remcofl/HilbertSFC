"""2D encode kernel builders."""

import numba as nb
import numpy as np

from hilbertsfc._typing import IntScalar, TileNBits2D, UIntArray

from .._cache import kernel_cache
from .._luts import lut_2d4b_b_qs_u64, lut_2d7b_b_qs_u64
from .._nbits import validate_nbits_2d


@nb.njit(inline="always")
def _hilbert_encode_2d_4bit_compacted_qs(x, y, nbits, lut):
    idx = 0
    state = 0  # Start state is either 0 or 1 based on nbits parity
    start_bit = (nbits - 1) & ~0x3  # Round down to nearest multiple of 4

    drop_bits = start_bit - nbits + 4
    if drop_bits > 0:  # Conditional compilation
        mask = np.uint64((1 << nbits) - 1)
        x &= mask  # Free (merges with first 0xF mask in loop)
        y &= mask

    # Process bits from MSB to LSB (..., 8, 4, 0)
    for bit in range(start_bit, -1, -4):
        b = ((x >> bit) & 0xF) << 4 | ((y >> bit) & 0xF)
        qs = lut[b] >> state  # state is << 4: (0, 16, 32, 48)
        q = (qs & 0xFF00) >> 8  # >> 8 to get quadrant (free lshr)

        idx |= q << (bit << 1)  # Append 8 quadrant bits to idx
        state = qs & 0xFF  # advance state

    return np.uint64(idx)


@nb.njit(inline="always")
def _hilbert_encode_2d_7bit_compacted_qs(x, y, nbits, lut):
    idx = 0
    start_bit = (nbits - 1) // 7 * 7  # Round down to nearest multiple of 7
    state = (start_bit + 7) & 0x1  # Start state is either 0 or 1 based on nbits parity

    drop_bits = start_bit - nbits + 7
    if drop_bits > 0:  # Conditional compilation
        mask = np.uint64((1 << nbits) - 1)
        x &= mask  # Free (merges with first 0xF mask in loop)
        y &= mask

    # Process bits from MSB to LSB (..., 14, 7, 0)
    for bit in range(start_bit, -1, -7):
        b = ((x >> bit) & 0x7F) << 7 | ((y >> bit) & 0x7F)
        qs = lut[b] >> (state << 4)  # select lane based on state
        q = (qs & 0xFFFC) >> 2  # >> 8 to get 14 quadrant bits(free lshr)

        idx |= q << (bit << 1)  # Append 14 quadrant bits to idx
        state = qs & 0x3  # advance state

    return np.uint64(idx)


@kernel_cache
def build_hilbert_encode_2d_impl(nbits: int, *, tile_nbits: TileNBits2D = 7):
    """Return a specialized scalar encoder: (x, y) -> index."""

    validate_nbits_2d(nbits)

    if tile_nbits == 7:
        lut = lut_2d7b_b_qs_u64()
        kernel = _hilbert_encode_2d_7bit_compacted_qs
    elif tile_nbits == 4:
        lut = lut_2d4b_b_qs_u64()
        kernel = _hilbert_encode_2d_4bit_compacted_qs
    else:
        # Should be unreachable due to type + normalization.
        raise ValueError("tile_nbits must be 4 or 7")

    @nb.njit(inline="always", cache=True)
    def encode_2d(x: IntScalar, y: IntScalar) -> int:
        return kernel(x, y, nbits, lut)  # type: ignore[reportReturnType]

    return encode_2d


@kernel_cache
def build_hilbert_encode_2d_batch_impl(
    nbits: int, *, parallel: bool = False, tile_nbits: TileNBits2D = 7
):
    """Return a specialized batch encoder: (xs, ys, out) -> out."""

    validate_nbits_2d(nbits)

    encode_scalar = build_hilbert_encode_2d_impl(nbits, tile_nbits=tile_nbits)

    @nb.njit(parallel=parallel, cache=True)
    def encode_2d_batch(xs: UIntArray, ys: UIntArray, out: UIntArray) -> None:
        n = xs.shape[0]
        for i in nb.prange(n):  # type: ignore[not-iterable]
            out[i] = encode_scalar(xs[i], ys[i])

    return encode_2d_batch
