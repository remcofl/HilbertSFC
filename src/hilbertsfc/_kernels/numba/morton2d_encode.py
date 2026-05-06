"""2D Morton encode Numba kernels+builders."""

import numba as nb
import numpy as np

from ..._cache import kernel_cache
from ..._nbits import validate_nbits_2d
from ...types import IntScalar, UIntArray

u64 = np.uint64
u32 = np.uint32


@nb.njit(inline="always")
def _part1by1_u32(x, nbits):
    if nbits > 8:
        x = (x | (x << 8)) & u32(0x00FF00FF)
    if nbits > 4:
        x = (x | (x << 4)) & u32(0x0F0F0F0F)
    if nbits > 2:
        x = (x | (x << 2)) & u32(0x33333333)
    if nbits > 1:
        x = (x | (x << 1)) & u32(0x55555555)
    return x


@nb.njit(inline="always")
def _part1by1_u64(x, nbits):
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
    if nbits <= 16:
        # For a fixed SIMD vector width, u32 operations have twice as many lanes as u64 operations.
        # However, this will only have effect when both the inputs and the output buffers are 32 bit or lower.
        mask = u32((1 << nbits) - 1)
        x = x & mask
        y = y & mask
        return _part1by1_u32(x, nbits) | (_part1by1_u32(y, nbits) << 1)

    mask = u64((1 << nbits) - 1)
    x &= mask
    y &= mask

    return _part1by1_u64(x, nbits) | (_part1by1_u64(y, nbits) << 1)


@kernel_cache
def build_morton_encode_2d_impl(nbits: int):
    """Return a specialized scalar encoder: (x, y) -> index."""

    validate_nbits_2d(nbits)

    @nb.njit(inline="always", cache=True)
    def encode_2d(x: IntScalar, y: IntScalar) -> int:
        return _morton_encode_2d(x, y, nbits)  # type: ignore[reportReturnType]

    return encode_2d


@kernel_cache
def build_morton_encode_2d_batch_impl(nbits: int, *, parallel: bool = False):
    """Return a specialized batch encoder: (xs, ys, out) -> out."""

    validate_nbits_2d(nbits)

    if parallel:

        @nb.njit(parallel=True, cache=True)
        def encode_2d_batch_parallel(
            xs: UIntArray, ys: UIntArray, out: UIntArray
        ) -> None:
            n = xs.size
            for i in nb.prange(n):  # type: ignore[not-iterable]
                out.flat[i] = _morton_encode_2d(xs.flat[i], ys.flat[i], nbits)

        return encode_2d_batch_parallel

    @nb.njit(parallel=False, cache=True)
    def encode_2d_batch_serial(xs: UIntArray, ys: UIntArray, out: UIntArray) -> None:
        n = xs.size
        for i in range(n):
            out.flat[i] = _morton_encode_2d(xs.flat[i], ys.flat[i], nbits)

    return encode_2d_batch_serial
