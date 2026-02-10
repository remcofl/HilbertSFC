"""3D encode kernel builders."""

from __future__ import annotations

import numba as nb
import numpy as np

from .._cache import kernel_cache
from .._luts import lut_3d2b_sb_so
from .._nbits import validate_nbits_3d
from .._typing import LutUIntDTypeLike, UIntArray


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

    return idx


@kernel_cache
def build_hilbert_encode_3d_impl(
    nbits: int, *, lut_dtype: LutUIntDTypeLike = np.uint16
):
    """Return a specialized scalar encoder: (x, y, z) -> index."""

    validate_nbits_3d(nbits)

    lut = lut_3d2b_sb_so(lut_dtype)

    @nb.njit(inline="always", cache=True)
    def encode_3d(x: int, y: int, z: int) -> int:
        return _hilbert_encode_3d_2bit_so(x, y, z, nbits, lut)

    return encode_3d


@kernel_cache
def build_hilbert_encode_3d_batch_impl(
    nbits: int, *, lut_dtype: LutUIntDTypeLike = np.uint16, parallel: bool = False
):
    """Return a specialized batch encoder: (xs, ys, zs, out) -> out."""

    validate_nbits_3d(nbits)

    encode_scalar = build_hilbert_encode_3d_impl(nbits, lut_dtype=lut_dtype)

    @nb.njit(parallel=parallel, cache=True)
    def encode_3d_batch(
        xs: UIntArray, ys: UIntArray, zs: UIntArray, out: UIntArray
    ) -> None:
        n = xs.shape[0]
        for i in nb.prange(n):  # ty:ignore[not-iterable]
            out[i] = encode_scalar(xs[i], ys[i], zs[i])

    return encode_3d_batch
