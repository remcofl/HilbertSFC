"""3D encode Numba kernel+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._luts import lut_3d2b_sb_so, lut_3d3b_sb_so
from ..._nbits import validate_nbits_3d
from ...types import IntScalar, LutUIntDTypeLike, TileNBits3D, UIntArray


@nb.njit(inline="always")
def _hilbert_encode_3d_2bit_so(x, y, z, nbits, lut):
    idx = 0
    state = 0  # Start state is either 0 or 5 based on parity
    start_bit = (nbits - 1) & ~0x1  # Round down to even number

    drop_bits = start_bit - nbits + 2
    if drop_bits > 0:  # Conditional compilation
        mask = np.uint64((1 << nbits) - 1)  # Free (merges with first 0x3 mask in loop)
        x &= mask
        y &= mask
        z &= mask

    # Process bits from MSB to LSB
    for bit in range(start_bit, -1, -2):
        b_x = (x >> bit) & 0x3
        b_y = (y >> bit) & 0x3
        b_z = (z >> bit) & 0x3
        b = (b_x << 4) | (b_y << 2) | b_z  # Combine to 6-bit bitband

        so = lut[state | b]  # state is << 6
        o = so & 0x3F  # Extract octant (6 bits)

        idx |= o << (3 * bit)  # Append 6 octant bits to idx
        state = so & 0x7C0  # Update state

    return np.uint64(idx)


@nb.njit(inline="always")
def _hilbert_encode_3d_3bit_so(x, y, z, nbits, lut):
    idx = 0
    start_bit = (nbits - 1) // 3 * 3
    state = (5 << 9) if (start_bit + 3) & 0x1 else 0

    drop_bits = start_bit - nbits + 3
    if drop_bits > 0:
        mask = np.uint64((1 << nbits) - 1)
        x &= mask
        y &= mask
        z &= mask

    for bit in range(start_bit, -1, -3):
        b_x = (x >> bit) & 0x7
        b_y = (y >> bit) & 0x7
        b_z = (z >> bit) & 0x7
        b = (b_x << 6) | (b_y << 3) | b_z

        so = lut[state | b]
        idx |= (so & 0x1FF) << (3 * bit)
        state = so & 0x3E00

    return np.uint64(idx)


@kernel_cache
def build_hilbert_encode_3d_impl(
    nbits: int, *, tile_nbits: TileNBits3D = 3, lut_dtype: LutUIntDTypeLike = np.uint16
):
    """Return a specialized scalar encoder: (x, y, z) -> index."""

    validate_nbits_3d(nbits)
    if tile_nbits == 3:
        lut = lut_3d3b_sb_so(lut_dtype)

        @nb.njit(inline="always", cache=True)
        def encode_3d_3bit(x: IntScalar, y: IntScalar, z: IntScalar) -> int:
            return _hilbert_encode_3d_3bit_so(x, y, z, nbits, lut)  # type: ignore[reportReturnType]

        return encode_3d_3bit

    if tile_nbits == 2:
        lut = lut_3d2b_sb_so(lut_dtype)

        @nb.njit(inline="always", cache=True)
        def encode_3d(x: IntScalar, y: IntScalar, z: IntScalar) -> int:
            return _hilbert_encode_3d_2bit_so(x, y, z, nbits, lut)  # type: ignore[reportReturnType]

        return encode_3d

    raise ValueError("tile_nbits must be 2 or 3")


@kernel_cache
def build_hilbert_encode_3d_batch_impl(
    nbits: int,
    *,
    parallel: bool = False,
    tile_nbits: TileNBits3D = 3,
    lut_dtype: LutUIntDTypeLike = np.uint16,
):
    """Return a specialized batch encoder: (xs, ys, zs, out) -> out."""

    validate_nbits_3d(nbits)

    if tile_nbits == 3:
        lut = lut_3d3b_sb_so(lut_dtype)
        if parallel:

            @nb.njit(parallel=True, cache=True)
            def encode_3d_batch_3bit_parallel(
                xs: UIntArray, ys: UIntArray, zs: UIntArray, out: UIntArray
            ) -> None:
                for i in nb.prange(xs.size):  # type: ignore[not-iterable]
                    out.flat[i] = _hilbert_encode_3d_3bit_so(
                        xs.flat[i], ys.flat[i], zs.flat[i], nbits, lut
                    )

            return encode_3d_batch_3bit_parallel

        @nb.njit(parallel=False, cache=True)
        def encode_3d_batch_3bit_serial(
            xs: UIntArray, ys: UIntArray, zs: UIntArray, out: UIntArray
        ) -> None:
            for i in range(xs.size):
                out.flat[i] = _hilbert_encode_3d_3bit_so(
                    xs.flat[i], ys.flat[i], zs.flat[i], nbits, lut
                )

        return encode_3d_batch_3bit_serial

    if tile_nbits == 2:
        lut = lut_3d2b_sb_so(lut_dtype)

        if parallel:

            @nb.njit(parallel=True, cache=True)
            def encode_3d_batch_parallel(
                xs: UIntArray, ys: UIntArray, zs: UIntArray, out: UIntArray
            ) -> None:
                n = xs.size
                for i in nb.prange(n):  # type: ignore[not-iterable]
                    out.flat[i] = _hilbert_encode_3d_2bit_so(
                        xs.flat[i], ys.flat[i], zs.flat[i], nbits, lut
                    )

            return encode_3d_batch_parallel

        @nb.njit(parallel=False, cache=True)
        def encode_3d_batch_serial(
            xs: UIntArray, ys: UIntArray, zs: UIntArray, out: UIntArray
        ) -> None:
            n = xs.size
            for i in range(n):
                out.flat[i] = _hilbert_encode_3d_2bit_so(
                    xs.flat[i], ys.flat[i], zs.flat[i], nbits, lut
                )

        return encode_3d_batch_serial

    raise ValueError("tile_nbits must be 2 or 3")
