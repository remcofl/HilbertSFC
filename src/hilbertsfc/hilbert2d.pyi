from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from ._typing import (
    Int8Array,
    Int16Array,
    Int32Array,
    Int64Array,
    IntArray,
    IntScalar,
    UInt8Array,
    UInt16Array,
    UInt32Array,
    UInt64Array,
    UIntArray,
)

# --- Scalar versions:
@overload
def hilbert_encode_2d(
    x: IntScalar,
    y: IntScalar,
    *,
    nbits: int | None = None,
    parallel: Literal[False] = False,
) -> int: ...
@overload
def hilbert_decode_2d(
    index: IntScalar,
    *,
    nbits: int | None = None,
    parallel: Literal[False] = False,
) -> tuple[int, int]: ...

# --- Array version with out:
@overload
def hilbert_encode_2d[OutDType: np.integer](
    xs: IntArray,
    ys: IntArray,
    *,
    nbits: int | None = None,
    out: NDArray[OutDType],
    parallel: bool = False,
) -> NDArray[OutDType]: ...

# 2*nbits <= 8
type _NBits2DIndexU8 = Literal[1, 2, 3, 4]

# 2*nbits <= 16
type _NBits2DIndexU16 = Literal[5, 6, 7, 8]

# 2*nbits <= 32
type _NBits2DIndexU32 = Literal[9, 10, 11, 12, 13, 14, 15, 16]

# 2*nbits <= 64
type _NBits2DIndexU64 = Literal[
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
]

# --- Array version with nbits but no out:
@overload
def hilbert_encode_2d(
    xs: IntArray,
    ys: IntArray,
    *,
    nbits: _NBits2DIndexU8,
    out: None = None,
    parallel: bool = False,
) -> UInt8Array: ...
@overload
def hilbert_encode_2d(
    xs: IntArray,
    ys: IntArray,
    *,
    nbits: _NBits2DIndexU16,
    out: None = None,
    parallel: bool = False,
) -> UInt16Array: ...
@overload
def hilbert_encode_2d(
    xs: IntArray,
    ys: IntArray,
    *,
    nbits: _NBits2DIndexU32,
    out: None = None,
    parallel: bool = False,
) -> UInt32Array: ...
@overload
def hilbert_encode_2d(
    xs: IntArray,
    ys: IntArray,
    *,
    nbits: _NBits2DIndexU64,
    out: None = None,
    parallel: bool = False,
) -> UInt64Array: ...

# --- Array version with no nbits and no out:
@overload
def hilbert_encode_2d(
    xs: UInt8Array | Int8Array,
    ys: UInt8Array | Int8Array,
    *,
    nbits: None = None,
    out: None = None,
    parallel: bool = False,
) -> UInt16Array: ...
@overload
def hilbert_encode_2d(
    xs: UInt16Array | Int16Array,
    ys: UInt16Array | Int16Array,
    *,
    nbits: None = None,
    out: None = None,
    parallel: bool = False,
) -> UInt32Array: ...
@overload
def hilbert_encode_2d(
    xs: UInt32Array | Int32Array | UInt64Array | Int64Array,
    ys: UInt32Array | Int32Array | UInt64Array | Int64Array,
    *,
    nbits: None = None,
    out: None = None,
    parallel: bool = False,
) -> UInt64Array: ...

# (Other combinations of xs and ys dtypes, also non-literal nbits):
@overload
def hilbert_encode_2d(
    xs: IntArray,
    ys: IntArray,
    *,
    nbits: int | None = None,
    out: None = None,
    parallel: bool = False,
) -> UIntArray: ...

# --- Array version with out:
@overload
def hilbert_decode_2d[XCoordDType: np.integer, YCoordDType: np.integer](
    indices: IntArray,
    *,
    nbits: int | None = None,
    out_xs: NDArray[XCoordDType],
    out_ys: NDArray[YCoordDType],
    parallel: bool = False,
) -> tuple[NDArray[XCoordDType], NDArray[YCoordDType]]: ...

type _NBitsCoordU8 = Literal[1, 2, 3, 4, 5, 6, 7, 8]

type _NBitsCoordU16 = Literal[9, 10, 11, 12, 13, 14, 15, 16]

type _NBitsCoordU32 = Literal[
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
]

# --- Array version with nbits but no out:
@overload
def hilbert_decode_2d(
    indices: IntArray,
    *,
    nbits: _NBitsCoordU8,
    out_xs: None = None,
    out_ys: None = None,
    parallel: bool = False,
) -> tuple[UInt8Array, UInt8Array]: ...
@overload
def hilbert_decode_2d(
    indices: IntArray,
    *,
    nbits: _NBitsCoordU16,
    out_xs: None = None,
    out_ys: None = None,
    parallel: bool = False,
) -> tuple[UInt16Array, UInt16Array]: ...
@overload
def hilbert_decode_2d(
    indices: IntArray,
    *,
    nbits: _NBitsCoordU32,
    out_xs: None = None,
    out_ys: None = None,
    parallel: bool = False,
) -> tuple[UInt32Array, UInt32Array]: ...

# --- Array version with no nbits and no out:
@overload
def hilbert_decode_2d(
    indices: UInt8Array | Int8Array | UInt16Array | Int16Array,
    *,
    nbits: None = None,
    out_xs: None = None,
    out_ys: None = None,
    parallel: bool = False,
) -> tuple[UInt8Array, UInt8Array]: ...
@overload
def hilbert_decode_2d(
    indices: UInt32Array | Int32Array,
    *,
    nbits: None = None,
    out_xs: None = None,
    out_ys: None = None,
    parallel: bool = False,
) -> tuple[UInt16Array, UInt16Array]: ...
@overload
def hilbert_decode_2d(
    indices: UInt64Array | Int64Array,
    *,
    nbits: None = None,
    out_xs: None = None,
    out_ys: None = None,
    parallel: bool = False,
) -> tuple[UInt32Array, UInt32Array]: ...

# (Other combinations of indices dtype, also non-literal nbits):
@overload
def hilbert_decode_2d(
    indices: IntArray,
    *,
    nbits: int | None = None,
    out_xs: None = None,
    out_ys: None = None,
    parallel: bool = False,
) -> tuple[UIntArray, UIntArray]: ...

# --- Kernel accessors:
def get_hilbert_encode_2d_kernel(
    nbits: int,
) -> Callable[[IntScalar, IntScalar], int]: ...
def get_hilbert_decode_2d_kernel(
    nbits: int,
) -> Callable[[IntScalar], tuple[int, int]]: ...
