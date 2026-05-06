"""3D Morton decode Numba kernels+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._nbits import validate_nbits_3d
from ...types import IntScalar, UIntArray

u64 = np.uint64
u32 = np.uint32


@nb.njit(inline="always")
def _compact1by2_u32(x, nbits):
    x &= u32(0x09249249)
    if nbits > 1:
        x = (x ^ (x >> 2)) & u32(0x030C30C3)
    if nbits > 2:
        x = (x ^ (x >> 4)) & u32(0x0300F00F)
    if nbits > 3:
        x = (x ^ (x >> 8)) & u32(0x030000FF)
    if nbits > 5:
        x = (x ^ (x >> 16)) & u32(0x000003FF)
    return x


@nb.njit(inline="always")
def _compact1by2_u64(x, nbits):
    x &= u64(0x1249249249249249)
    if nbits > 1:
        x = (x ^ (x >> 2)) & u64(0x10C30C30C30C30C3)
    if nbits > 2:
        x = (x ^ (x >> 4)) & u64(0x100F00F00F00F00F)
    if nbits > 3:
        x = (x ^ (x >> 8)) & u64(0x001F0000FF0000FF)
    if nbits > 5:
        x = (x ^ (x >> 16)) & u64(0x001F00000000FFFF)
    if nbits > 10:
        x = (x ^ (x >> 32)) & u64(0x00000000001FFFFF)
    return x


@nb.njit(inline="always")
def _morton_decode_3d(index, nbits):
    if nbits <= 10:
        if nbits < 10:
            index = u32(index) & u32((1 << (3 * nbits)) - 1)
        else:
            index = u32(index)

        x = _compact1by2_u32(index, nbits)
        y = _compact1by2_u32(index >> 1, nbits)
        z = _compact1by2_u32(index >> 2, nbits)
        return x, y, z

    if nbits < 21:
        index &= u64((1 << (3 * nbits)) - 1)

    x = _compact1by2_u64(index, nbits)
    y = _compact1by2_u64(index >> 1, nbits)
    z = _compact1by2_u64(index >> 2, nbits)
    return x, y, z


@kernel_cache
def build_morton_decode_3d_impl(nbits: int):
    """Return a specialized scalar decoder: index -> (x, y, z)."""

    validate_nbits_3d(nbits)

    @nb.njit(inline="always", cache=True)
    def decode_3d(index: IntScalar) -> tuple[int, int, int]:
        return _morton_decode_3d(index, nbits)  # type: ignore[reportReturnType]

    return decode_3d


@kernel_cache
def build_morton_decode_3d_batch_impl(nbits: int, *, parallel: bool = False):
    """Return a specialized batch decoder: (indices, xs, ys, zs) -> (xs, ys, zs)."""

    validate_nbits_3d(nbits)

    if parallel:

        @nb.njit(parallel=True, cache=True)
        def decode_3d_batch_parallel(
            indices: UIntArray, xs: UIntArray, ys: UIntArray, zs: UIntArray
        ) -> None:
            n = indices.size
            for i in nb.prange(n):  # type: ignore[not-iterable]
                xs.flat[i], ys.flat[i], zs.flat[i] = _morton_decode_3d(
                    indices.flat[i], nbits
                )

        return decode_3d_batch_parallel

    @nb.njit(parallel=False, cache=True)
    def decode_3d_batch_serial(
        indices: UIntArray, xs: UIntArray, ys: UIntArray, zs: UIntArray
    ) -> None:
        n = indices.size
        for i in range(n):
            xs.flat[i], ys.flat[i], zs.flat[i] = _morton_decode_3d(
                indices.flat[i], nbits
            )

    return decode_3d_batch_serial
