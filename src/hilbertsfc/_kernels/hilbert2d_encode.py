"""2D encode kernel builders."""

from __future__ import annotations

import numba as nb
import numpy as np

from hilbertsfc._typing import IntScalar, UIntArray

from .._cache import kernel_cache
from .._luts import lut_2d4b_b_qs_u64
from .._nbits import validate_nbits_2d


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


@kernel_cache
def build_hilbert_encode_2d_impl(nbits: int):
    """Return a specialized scalar encoder: (x, y) -> index."""

    validate_nbits_2d(nbits)

    lut = lut_2d4b_b_qs_u64()

    @nb.njit(inline="always", cache=True)
    def encode_2d(x: IntScalar, y: IntScalar) -> int:
        return _hilbert_encode_2d_4bit_compacted_qs(x, y, nbits, lut)  # type: ignore[reportReturnType]

    return encode_2d


@kernel_cache
def build_hilbert_encode_2d_batch_impl(nbits: int, *, parallel: bool = False):
    """Return a specialized batch encoder: (xs, ys, out) -> out."""

    validate_nbits_2d(nbits)

    encode_scalar = build_hilbert_encode_2d_impl(nbits)

    @nb.njit(parallel=parallel, cache=True)
    def encode_2d_batch(xs: UIntArray, ys: UIntArray, out: UIntArray) -> None:
        n = xs.shape[0]
        for i in nb.prange(n):  # type: ignore[not-iterable]
            out[i] = encode_scalar(xs[i], ys[i])

    return encode_2d_batch
