"""3D Morton encode Numba kernels+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._nbits import validate_nbits_3d
from ...types import IntScalar, UIntArray

u64 = np.uint64


@nb.njit(inline="always")
def _part1by2(x, nbits):
    if nbits > 10:
        x = (x | (x << 32)) & u64(0x001F00000000FFFF)
    if nbits > 5:
        x = (x | (x << 16)) & u64(0x001F0000FF0000FF)
    if nbits > 3:
        x = (x | (x << 8)) & u64(0x100F00F00F00F00F)
    if nbits > 2:
        x = (x | (x << 4)) & u64(0x10C30C30C30C30C3)
    if nbits > 1:
        x = (x | (x << 2)) & u64(0x1249249249249249)
    return x


@nb.njit(inline="always")
def _morton_encode_3d(x, y, z, nbits):
    if nbits < 21:
        mask = u64((1 << nbits) - 1)
        x &= mask
        y &= mask
        z &= mask

    return u64(
        _part1by2(x, nbits) | (_part1by2(y, nbits) << 1) | (_part1by2(z, nbits) << 2)
    )


@kernel_cache
def build_morton_encode_3d_impl(nbits: int):
    """Return a specialized scalar encoder: (x, y, z) -> index."""

    validate_nbits_3d(nbits)

    @nb.njit(inline="always", cache=False)
    def encode_3d(x: IntScalar, y: IntScalar, z: IntScalar) -> int:
        return _morton_encode_3d(x, y, z, nbits)  # type: ignore[reportReturnType]

    return encode_3d


@kernel_cache
def build_morton_encode_3d_batch_impl(nbits: int, *, parallel: bool = False):
    """Return a specialized batch encoder: (xs, ys, zs, out) -> out."""

    validate_nbits_3d(nbits)

    encode_scalar = build_morton_encode_3d_impl(nbits)

    @nb.njit(parallel=parallel, cache=False)
    def encode_3d_batch(
        xs: UIntArray, ys: UIntArray, zs: UIntArray, out: UIntArray
    ) -> None:
        n = xs.size
        for i in nb.prange(n):  # type: ignore[not-iterable]
            out.flat[i] = encode_scalar(xs.flat[i], ys.flat[i], zs.flat[i])

    return encode_3d_batch
