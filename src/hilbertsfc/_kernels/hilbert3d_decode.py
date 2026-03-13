"""3D decode kernel builders."""

import numba as nb
import numpy as np

from .._cache import kernel_cache
from .._luts import lut_3d2b_so_sb
from .._nbits import validate_nbits_3d
from .._typing import IntScalar, LutUIntDTypeLike, UIntArray


@nb.njit(inline="always")
def _hilbert_decode_3d_2bit_sb(idx, nbits, lut):
    x = y = z = 0
    state = 0  # Start state is either 0 or 5 based on parity
    start_bit = (nbits - 1) & ~0x1  # Every extra 2 nbits adds 6 bits to index

    drop_bits = start_bit - nbits + 2
    if drop_bits > 0:  # Conditional compilation
        idx &= np.uint64((1 << (3 * nbits)) - 1)  # Merges with 0x3F mask

    for bit in range(start_bit, -1, -2):
        o = (idx >> (3 * bit)) & 0x3F  # Extract 6 octant bits

        sb = lut[state | o]  # state is << 6
        b_x = (sb & 0x30) >> 4  # These shifts are all free lshr
        b_y = (sb & 0x0C) >> 2
        b_z = sb & 0x03

        x |= b_x << bit
        y |= b_y << bit
        z |= b_z << bit

        state = sb & 0x7C0  # Update state

    return x, y, z


@kernel_cache
def build_hilbert_decode_3d_impl(
    nbits: int, *, lut_dtype: LutUIntDTypeLike = np.uint16
):
    """Return a specialized scalar decoder: index -> (x, y, z)."""

    validate_nbits_3d(nbits)

    lut = lut_3d2b_so_sb(lut_dtype)

    @nb.njit(inline="always", cache=True)
    def decode_3d(index: IntScalar) -> tuple[int, int, int]:
        return _hilbert_decode_3d_2bit_sb(index, nbits, lut)

    return decode_3d


@kernel_cache
def build_hilbert_decode_3d_batch_impl(
    nbits: int, *, lut_dtype: LutUIntDTypeLike = np.uint16, parallel: bool = False
):
    """Return a specialized batch decoder: (indices, xs, ys, zs) -> (xs, ys, zs)."""

    validate_nbits_3d(nbits)

    decode_scalar = build_hilbert_decode_3d_impl(nbits, lut_dtype=lut_dtype)

    @nb.njit(parallel=parallel, cache=True)
    def decode_3d_batch(
        indices: UIntArray, xs: UIntArray, ys: UIntArray, zs: UIntArray
    ) -> None:
        n = indices.shape[0]
        for i in nb.prange(n):  # type: ignore[not-iterable]
            xs[i], ys[i], zs[i] = decode_scalar(indices[i])

    return decode_3d_batch
