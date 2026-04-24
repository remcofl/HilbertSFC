"""2D Morton encode Numba kernels+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._nbits import validate_nbits_2d
from ...types import IntScalar, UIntArray

u64 = np.uint64


@nb.njit(inline="always")
def _part1by1(x, nbits):
    if nbits > 16:
        x = (x | (x << 16)) & u64(0x0000FFFF0000FFFF)
    if nbits > 8:
        x = (x | (x << 8)) & u64(0x00FF00FF00FF00FF)
    if nbits > 4:
        x = (x | (x << 4)) & u64(0x0F0F0F0F0F0F0F0F)
    if nbits > 2:
        x = (x | (x << 2)) & u64(0x3333333333333333)
    if nbits > 1:
        x = (x | (x << 1)) & u64(0x5555555555555555)
    return x


@nb.njit(inline="always")
def _morton_encode_2d(x, y, nbits):
    if nbits < 32:
        mask = u64((1 << nbits) - 1)
        x &= mask
        y &= mask

    return u64(_part1by1(x, nbits) | (_part1by1(y, nbits) << 1))


@kernel_cache
def build_morton_encode_2d_impl(nbits: int):
    """Return a specialized scalar encoder: (x, y) -> index."""

    validate_nbits_2d(nbits)

    @nb.njit(inline="always", cache=False)
    def encode_2d(x: IntScalar, y: IntScalar) -> int:
        return _morton_encode_2d(x, y, nbits)  # type: ignore[reportReturnType]

    return encode_2d


@kernel_cache
def build_morton_encode_2d_batch_impl(nbits: int, *, parallel: bool = False):
    """Return a specialized batch encoder: (xs, ys, out) -> out."""

    validate_nbits_2d(nbits)

    encode_scalar = build_morton_encode_2d_impl(nbits)

    @nb.njit(parallel=parallel, cache=False)
    def encode_2d_batch(xs: UIntArray, ys: UIntArray, out: UIntArray) -> None:
        n = xs.size
        for i in nb.prange(n):  # type: ignore[not-iterable]
            out.flat[i] = encode_scalar(xs.flat[i], ys.flat[i])

    return encode_2d_batch
