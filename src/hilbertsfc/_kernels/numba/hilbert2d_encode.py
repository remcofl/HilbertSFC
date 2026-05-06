"""2D encode Numba kernels+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._luts import lut_2d4b_b_qs_u64, lut_2d7b_b_qs_u64
from ..._nbits import validate_nbits_2d
from ...types import IntScalar, TileNBits2D, UIntArray


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


def _auto_tile_nbits_2d(nbits: int) -> TileNBits2D:
    """Determine the best tile size for 2D *encode* kernels given ``nbits``.

    Policy ``ceil(nbits/4) == ceil(nbits/7)`` -> 4, else 7. This means:
    - ``4`` for very small domains (``nbits <= 4``) and for ``nbits == 8``.
    - ``7`` otherwise.
    """

    if nbits <= 4 or nbits == 8:
        return 4
    return 7


@kernel_cache
def build_hilbert_encode_2d_impl(nbits: int, *, tile_nbits: TileNBits2D | None = None):
    """Return a specialized scalar encoder: (x, y) -> index."""

    validate_nbits_2d(nbits)

    if tile_nbits is None:
        tile_nbits = _auto_tile_nbits_2d(nbits)

    if tile_nbits == 7:
        lut = lut_2d7b_b_qs_u64()

        @nb.njit(inline="always", cache=True)
        def encode_2d_7bit(x: IntScalar, y: IntScalar) -> int:
            return _hilbert_encode_2d_7bit_compacted_qs(  # type: ignore[reportReturnType]
                x, y, nbits, lut
            )

        return encode_2d_7bit

    elif tile_nbits == 4:
        lut = lut_2d4b_b_qs_u64()

        @nb.njit(inline="always", cache=True)
        def encode_2d_4bit(x: IntScalar, y: IntScalar) -> int:
            return _hilbert_encode_2d_4bit_compacted_qs(  # type: ignore[reportReturnType]
                x, y, nbits, lut
            )

        return encode_2d_4bit

    else:
        raise ValueError("tile_nbits must be 4 or 7 (or None for auto)")


@kernel_cache
def build_hilbert_encode_2d_batch_impl(
    nbits: int, *, parallel: bool = False, tile_nbits: TileNBits2D | None = None
):
    """Return a specialized batch encoder: (xs, ys, out) -> out."""

    validate_nbits_2d(nbits)

    if tile_nbits is None:
        tile_nbits = _auto_tile_nbits_2d(nbits)

    if tile_nbits == 7:
        lut = lut_2d7b_b_qs_u64()
        if parallel:

            @nb.njit(parallel=True, cache=True)
            def encode_2d_batch_7bit_parallel(
                xs: UIntArray, ys: UIntArray, out: UIntArray
            ) -> None:
                n = xs.size
                for i in nb.prange(n):  # type: ignore[not-iterable]
                    out.flat[i] = _hilbert_encode_2d_7bit_compacted_qs(
                        xs.flat[i], ys.flat[i], nbits, lut
                    )

            return encode_2d_batch_7bit_parallel

        @nb.njit(parallel=False, cache=True)
        def encode_2d_batch_7bit_serial(
            xs: UIntArray, ys: UIntArray, out: UIntArray
        ) -> None:
            n = xs.size
            for i in range(n):
                out.flat[i] = _hilbert_encode_2d_7bit_compacted_qs(
                    xs.flat[i], ys.flat[i], nbits, lut
                )

        return encode_2d_batch_7bit_serial

    if tile_nbits == 4:
        lut = lut_2d4b_b_qs_u64()
        if parallel:

            @nb.njit(parallel=True, cache=True)
            def encode_2d_batch_4bit_parallel(
                xs: UIntArray, ys: UIntArray, out: UIntArray
            ) -> None:
                n = xs.size
                for i in nb.prange(n):  # type: ignore[not-iterable]
                    out.flat[i] = _hilbert_encode_2d_4bit_compacted_qs(
                        xs.flat[i], ys.flat[i], nbits, lut
                    )

            return encode_2d_batch_4bit_parallel

        @nb.njit(parallel=False, cache=True)
        def encode_2d_batch_4bit_serial(
            xs: UIntArray, ys: UIntArray, out: UIntArray
        ) -> None:
            n = xs.size
            for i in range(n):
                out.flat[i] = _hilbert_encode_2d_4bit_compacted_qs(
                    xs.flat[i], ys.flat[i], nbits, lut
                )

        return encode_2d_batch_4bit_serial

    raise ValueError("tile_nbits must be 4 or 7 (or None for auto)")
