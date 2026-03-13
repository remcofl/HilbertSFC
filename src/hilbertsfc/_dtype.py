"""Low-overhead dtype helpers.

These functions are used in the public encode/decode APIs (array mode) to keep
validation constant-time (i.e., based only on dtype metadata, not by scanning arrays).
"""

import numpy as np


def dtype_effective_bits(dtype: np.dtype) -> int:
    """Bit-width available for non-negative integer values.

    For signed integers, the sign bit is excluded.
    """

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"expected integer dtype, got {dtype!r}")

    bits = dtype.itemsize * 8
    if np.issubdtype(dtype, np.signedinteger):
        return bits - 1
    return bits


def max_nbits_for_index_dtype(dtype: np.dtype, *, dims: int) -> int:
    """Max coordinate nbits such that the Hilbert index fits in `dtype`.

    - 2D: index uses `2*nbits` bits
    - 3D: index uses `3*nbits` bits
    """

    if dims <= 0:
        raise ValueError("dims must be positive")
    return dtype_effective_bits(dtype) // dims


def unsigned_view(arr: np.ndarray) -> np.ndarray:
    """Zero-copy view of a signed integer array as unsigned.

    This does NOT validate values are non-negative; callers should treat negative
    inputs as undefined behavior.
    """

    if np.issubdtype(arr.dtype, np.signedinteger):
        unsigned = np.dtype(f"u{arr.dtype.itemsize}")
        return arr.view(unsigned)
    return arr


def choose_lut_dtype_for_index_dtype(
    dtype: np.dtype,
) -> type[np.uint16] | type[np.uint32] | type[np.uint64]:
    """Choose one of the supported LUT element dtypes based on index width."""

    bits = dtype_effective_bits(dtype)
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    return np.uint64


def choose_uint_index_dtype(
    *, nbits: int, dims: int
) -> type[np.uint8] | type[np.uint16] | type[np.uint32] | type[np.uint64]:
    """Choose a minimal unsigned dtype that can hold a Hilbert index.

    The index uses `dims * nbits` bits.
    """

    if nbits <= 0:
        raise ValueError("nbits must be positive")
    if dims <= 0:
        raise ValueError("dims must be positive")

    bits = dims * nbits
    if bits > 64:
        raise ValueError(
            f"nbits={nbits} with dims={dims} requires {bits} index bits; "
            "no suitable uint dtype available"
        )
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    return np.uint64


def choose_uint_coord_dtype(
    *, nbits: int
) -> type[np.uint8] | type[np.uint16] | type[np.uint32]:
    """Choose a minimal unsigned dtype that can hold a coordinate."""

    if nbits <= 0:
        raise ValueError("nbits must be positive")
    if nbits <= 8:
        return np.uint8
    if nbits <= 16:
        return np.uint16
    if nbits > 32:
        raise ValueError(
            f"nbits={nbits} requires >32 coordinate bits; no suitable uint dtype available"
        )
    return np.uint32
