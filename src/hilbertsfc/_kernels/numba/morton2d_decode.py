"""2D Morton decode Numba kernels+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._nbits import validate_nbits_2d
from ...types import IntScalar, UIntArray

u64 = np.uint64
u32 = np.uint32


@nb.njit(inline="always")
def _compact1by1_u32(x, nbits):
    x &= u32(0x55555555)
    if nbits > 1:
        x = (x | (x >> 1)) & u32(0x33333333)
    if nbits > 2:
        x = (x | (x >> 2)) & u32(0x0F0F0F0F)
    if nbits > 4:
        x = (x | (x >> 4)) & u32(0x00FF00FF)
    if nbits > 8:
        x = (x | (x >> 8)) & u32(0x0000FFFF)
    return x


@nb.njit(inline="always")
def _compact1by1_u64(x, nbits):
    x &= u64(0x5555555555555555)
    if nbits > 1:
        x = (x | (x >> 1)) & u64(0x3333333333333333)
    if nbits > 2:
        x = (x | (x >> 2)) & u64(0x0F0F0F0F0F0F0F0F)
    if nbits > 4:
        x = (x | (x >> 4)) & u64(0x00FF00FF00FF00FF)
    if nbits > 8:
        x = (x | (x >> 8)) & u64(0x0000FFFF0000FFFF)
    if nbits > 16:
        x = (x | (x >> 16)) & u64(0x00000000FFFFFFFF)
    return x


@nb.njit(inline="always")
def _morton_decode_2d(index, nbits):
    if nbits <= 16:
        if nbits < 16:
            index = u32(index) & u32((1 << (nbits << 1)) - 1)
        else:
            index = u32(index)

        x = _compact1by1_u32(index, nbits)
        y = _compact1by1_u32(index >> 1, nbits)
        return x, y

    if nbits < 32:
        index &= u64((1 << (nbits << 1)) - 1)

    x = _compact1by1_u64(index, nbits)
    y = _compact1by1_u64(index >> 1, nbits)
    return x, y


@kernel_cache
def build_morton_decode_2d_impl(nbits: int):
    """Return a specialized scalar decoder: index -> (x, y)."""

    validate_nbits_2d(nbits)

    @nb.njit(inline="always", cache=False)
    def decode_2d(index: IntScalar) -> tuple[int, int]:
        return _morton_decode_2d(index, nbits)  # type: ignore[reportReturnType]

    return decode_2d


@kernel_cache
def build_morton_decode_2d_batch_impl(nbits: int, *, parallel: bool = False):
    """Return a specialized batch decoder: (indices, xs, ys) -> (xs, ys)."""

    validate_nbits_2d(nbits)

    decode_scalar = build_morton_decode_2d_impl(nbits)

    @nb.njit(parallel=parallel, cache=False)
    def decode_2d_batch(indices: UIntArray, xs: UIntArray, ys: UIntArray) -> None:
        n = indices.size
        for i in nb.prange(n):  # type: ignore[not-iterable]
            xs.flat[i], ys.flat[i] = decode_scalar(indices.flat[i])

    return decode_2d_batch
