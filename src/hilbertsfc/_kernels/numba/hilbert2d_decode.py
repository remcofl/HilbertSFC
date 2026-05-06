"""2D decode Numba kernels+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._luts import lut_2d4b_q_bs_u64, lut_2d7b_q_bs_u64
from ..._nbits import validate_nbits_2d
from ...types import IntScalar, TileNBits2D, UIntArray


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


@nb.njit(inline="always")
def _hilbert_decode_2d_7bit_compacted_bs(idx, nbits, lut):
    x = y = 0
    start_bit = (nbits - 1) // 7 * 7
    state = (start_bit + 7) & 0x1  # Start state is either 0 or 1 based on nbits parity

    drop_bits = start_bit - nbits + 7
    if drop_bits > 0:
        idx &= np.uint64((1 << (nbits << 1)) - 1)

    # Process quadrants from MSB to LSB (..., 14, 7, 0)
    for bit in range(start_bit, -1, -7):
        q = (idx >> (bit << 1)) & 0x3FFF
        bs = lut[q] >> (state << 4)

        b_x = (bs & 0xFE00) >> 9
        b_y = (bs & 0x01FC) >> 2
        x |= b_x << bit  # Append 7 bits to x
        y |= b_y << bit  # Append 7 bits to y

        state = bs & 0x3

    return x, y


def _auto_tile_nbits_2d(nbits: int) -> TileNBits2D:
    """Determine the best tile size for 2D *decode* kernels given ``nbits``.

    Policy ``ceil(nbits/4) == ceil(nbits/7)`` -> 4, else 7. This means:
    - ``4`` for very small domains (``nbits <= 4``) and for ``nbits == 8``.
    - ``7`` otherwise.
    """

    if nbits <= 4 or nbits == 8:
        return 4
    return 7


@kernel_cache
def build_hilbert_decode_2d_impl(nbits: int, *, tile_nbits: TileNBits2D | None = None):
    """Return a specialized scalar decoder: index -> (x, y)."""

    validate_nbits_2d(nbits)

    if tile_nbits is None:
        tile_nbits = _auto_tile_nbits_2d(nbits)

    if tile_nbits == 7:
        lut = lut_2d7b_q_bs_u64()

        @nb.njit(inline="always", cache=True)
        def decode_2d_7bit(index: IntScalar) -> tuple[int, int]:
            return _hilbert_decode_2d_7bit_compacted_bs(index, nbits, lut)

        return decode_2d_7bit

    elif tile_nbits == 4:
        lut = lut_2d4b_q_bs_u64()

        @nb.njit(inline="always", cache=True)
        def decode_2d_4bit(index: IntScalar) -> tuple[int, int]:
            return _hilbert_decode_2d_4bit_compacted_bs(index, nbits, lut)

        return decode_2d_4bit

    else:
        raise ValueError("tile_nbits must be 4 or 7 (or None for auto)")


@kernel_cache
def build_hilbert_decode_2d_batch_impl(
    nbits: int, *, parallel: bool = False, tile_nbits: TileNBits2D | None = None
):
    """Return a specialized batch decoder: (indices, xs, ys) -> (xs, ys)."""

    validate_nbits_2d(nbits)

    if tile_nbits is None:
        tile_nbits = _auto_tile_nbits_2d(nbits)

    if tile_nbits == 7:
        lut = lut_2d7b_q_bs_u64()
        if parallel:

            @nb.njit(parallel=True, cache=True)
            def decode_2d_batch_7bit_parallel(
                indices: UIntArray, xs: UIntArray, ys: UIntArray
            ) -> None:
                n = indices.size
                for i in nb.prange(n):  # type: ignore[not-iterable]
                    xs.flat[i], ys.flat[i] = _hilbert_decode_2d_7bit_compacted_bs(
                        indices.flat[i], nbits, lut
                    )

            return decode_2d_batch_7bit_parallel

        @nb.njit(parallel=False, cache=True)
        def decode_2d_batch_7bit_serial(
            indices: UIntArray, xs: UIntArray, ys: UIntArray
        ) -> None:
            n = indices.size
            for i in range(n):
                xs.flat[i], ys.flat[i] = _hilbert_decode_2d_7bit_compacted_bs(
                    indices.flat[i], nbits, lut
                )

        return decode_2d_batch_7bit_serial

    if tile_nbits == 4:
        lut = lut_2d4b_q_bs_u64()
        if parallel:

            @nb.njit(parallel=True, cache=True)
            def decode_2d_batch_4bit_parallel(
                indices: UIntArray, xs: UIntArray, ys: UIntArray
            ) -> None:
                n = indices.size
                for i in nb.prange(n):  # type: ignore[not-iterable]
                    xs.flat[i], ys.flat[i] = _hilbert_decode_2d_4bit_compacted_bs(
                        indices.flat[i], nbits, lut
                    )

            return decode_2d_batch_4bit_parallel

        @nb.njit(parallel=False, cache=True)
        def decode_2d_batch_4bit_serial(
            indices: UIntArray, xs: UIntArray, ys: UIntArray
        ) -> None:
            n = indices.size
            for i in range(n):
                xs.flat[i], ys.flat[i] = _hilbert_decode_2d_4bit_compacted_bs(
                    indices.flat[i], nbits, lut
                )

        return decode_2d_batch_4bit_serial

    raise ValueError("tile_nbits must be 4 or 7 (or None for auto)")
