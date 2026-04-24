"""Shared 2D public API plumbing for scalar+array wrappers."""

import warnings
from typing import cast

import numpy as np

from ._dtype import (
    choose_uint_coord_dtype,
    choose_uint_index_dtype,
    dtype_effective_bits,
    max_nbits_for_index_dtype,
    unsigned_view,
)
from ._flatten import flatten_nocopy as _flatten_nocopy
from ._input_checks import is_int_scalar_or_0d_array
from ._nbits import MAX_NBITS_2D, validate_nbits_2d
from ._public_api_types import (
    BuildDecode2DBatch,
    BuildDecode2DScalar,
    BuildEncode2DBatch,
    BuildEncode2DScalar,
)
from .types import IntArray, IntScalar


def encode_2d_api(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    *,
    nbits: int | None,
    out: IntArray | None,
    parallel: bool,
    build_scalar: BuildEncode2DScalar,
    build_batch: BuildEncode2DBatch,
) -> int | IntArray:
    """Shared implementation for public 2D encoders."""

    x_is_scalar = is_int_scalar_or_0d_array(x)
    y_is_scalar = is_int_scalar_or_0d_array(y)
    if x_is_scalar != y_is_scalar:
        raise TypeError("x and y must both be scalars or both be arrays")

    if x_is_scalar:
        if out is not None:
            raise TypeError("out is only valid for array mode")
        if parallel:
            warnings.warn(
                "parallel=True has no effect in scalar mode",
                UserWarning,
                stacklevel=2,
            )
        return _encode_2d_scalar(
            cast(IntScalar, x),
            cast(IntScalar, y),
            nbits,
            build_scalar,
        )

    return _encode_2d_batch(
        cast(IntArray, x),
        cast(IntArray, y),
        nbits,
        out=out,
        parallel=parallel,
        build_batch=build_batch,
    )


def decode_2d_api(
    index: IntScalar | IntArray,
    *,
    nbits: int | None,
    out_x: IntArray | None,
    out_y: IntArray | None,
    parallel: bool,
    build_scalar: BuildDecode2DScalar,
    build_batch: BuildDecode2DBatch,
) -> tuple[int, int] | tuple[IntArray, IntArray]:
    """Shared implementation for public 2D decoders."""

    index_is_scalar = is_int_scalar_or_0d_array(index)
    if index_is_scalar:
        if out_x is not None or out_y is not None:
            raise TypeError("out_x/out_y are only valid for array mode")
        if parallel:
            warnings.warn(
                "parallel=True has no effect in scalar mode",
                UserWarning,
                stacklevel=2,
            )
        return _decode_2d_scalar(cast(IntScalar, index), nbits, build_scalar)

    return _decode_2d_batch(
        cast(IntArray, index),
        nbits,
        out_x=out_x,
        out_y=out_y,
        parallel=parallel,
        build_batch=build_batch,
    )


def _encode_2d_scalar(
    x: IntScalar,
    y: IntScalar,
    nbits: int | None,
    build_scalar: BuildEncode2DScalar,
) -> int:
    if nbits is None:
        nbits = MAX_NBITS_2D

    x_i, y_i = int(x), int(y)
    max_v = np.iinfo(np.uint32).max
    if x_i < 0 or y_i < 0 or x_i > max_v or y_i > max_v:
        raise ValueError(
            "Scalar inputs must be non-negative and fit in uint32; "
            f"got x={x_i}, y={y_i}"
        )

    impl = build_scalar(nbits)
    return impl(np.uint32(x_i), np.uint32(y_i))


def _decode_2d_scalar(
    index: IntScalar,
    nbits: int | None,
    build_scalar: BuildDecode2DScalar,
) -> tuple[int, int]:
    if nbits is None:
        nbits = MAX_NBITS_2D

    index_i = int(index)
    max_v = np.iinfo(np.uint64).max
    if index_i < 0 or index_i > max_v:
        raise ValueError(
            f"Scalar index must be non-negative and fit in uint64; got index={index_i}"
        )

    impl = build_scalar(nbits)
    return impl(np.uint64(index_i))


def _encode_2d_batch(
    x: IntArray,
    y: IntArray,
    nbits: int | None,
    *,
    out: IntArray | None,
    parallel: bool,
    build_batch: BuildEncode2DBatch,
) -> IntArray:
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape=}, {y.shape=}"
        )

    max_coord_nbits = max(dtype_effective_bits(x.dtype), dtype_effective_bits(y.dtype))
    if nbits is None:
        nbits = max_coord_nbits
        if nbits > MAX_NBITS_2D:
            warnings.warn(
                f"The maximum effective bits of the coordinate dtype is {nbits}, "
                f"which exceeds the algorithm maximum of {MAX_NBITS_2D}. "
                f"Using nbits={MAX_NBITS_2D} instead. This means that excess bits in the input coordinates "
                f"will be ignored. To silence this warning, explicitly set nbits<={MAX_NBITS_2D}.",
                UserWarning,
                stacklevel=2,
            )
            nbits = MAX_NBITS_2D
    else:
        validate_nbits_2d(nbits)
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in coordinate dtypes; "
                f"got {x.dtype=} with {dtype_effective_bits(x.dtype)} effective bits, "
                f"{y.dtype=} with {dtype_effective_bits(y.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}."
            )

    if out is None:
        out_dtype = choose_uint_index_dtype(nbits=nbits, dims=2)
        out_u = out = np.empty(x.shape, dtype=out_dtype, order="C")
    else:
        if out.shape != x.shape:
            raise ValueError(
                f"out must have the same shape as x/y; got {out.shape=}, {x.shape=}"
            )
        max_index_nbits = max_nbits_for_index_dtype(out.dtype, dims=2)
        if nbits > max_index_nbits:
            viable_dtype = np.dtype(choose_uint_index_dtype(nbits=nbits, dims=2))
            raise ValueError(
                f"{nbits=} does not fit in out dtype; got {out.dtype=} "
                f"which supports up to nbits={max_index_nbits}; "
                f"consider using {viable_dtype} or a wider dtype, or reduce nbits to fit the out dtype."
            )
        out_u = unsigned_view(out)

    x_u = unsigned_view(x)
    y_u = unsigned_view(y)

    x_1d = _flatten_nocopy(x_u, "x", order="C", strict=False)
    y_1d = _flatten_nocopy(y_u, "y", order="C", strict=False)
    out_1d = _flatten_nocopy(out_u, "out", order="C", strict=False)

    impl = build_batch(nbits, parallel=parallel)
    impl(x_1d, y_1d, out_1d)

    return out


def _decode_2d_batch(
    index: IntArray,
    nbits: int | None,
    *,
    out_x: IntArray | None,
    out_y: IntArray | None,
    parallel: bool,
    build_batch: BuildDecode2DBatch,
) -> tuple[IntArray, IntArray]:
    max_index_nbits = max_nbits_for_index_dtype(index.dtype, dims=2)
    if nbits is None:
        nbits = max_index_nbits
    else:
        validate_nbits_2d(nbits)
        if nbits > max_index_nbits:
            raise ValueError(
                f"{nbits=} exceeds the effective bits of the index dtype; "
                f"got {index.dtype=} which supports up to {max_index_nbits} bits for 2D coordinates. "
                f"max nbits is {max_index_nbits}."
            )

    if (out_x is None) != (out_y is None):
        raise ValueError("out_x and out_y must be provided together")

    if out_x is None or out_y is None:
        coord_dtype = choose_uint_coord_dtype(nbits=nbits)
        out_x_u = out_x = np.empty(index.shape, dtype=coord_dtype, order="C")
        out_y_u = out_y = np.empty(index.shape, dtype=coord_dtype, order="C")
    else:
        if out_x.shape != index.shape or out_y.shape != index.shape:
            raise ValueError(
                "out_x and out_y must have the same shape as index; "
                f"got {index.shape=}, {out_x.shape=}, {out_y.shape=}"
            )
        max_coord_nbits = min(
            dtype_effective_bits(out_x.dtype), dtype_effective_bits(out_y.dtype)
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in out_x/out_y dtypes; "
                f"got {out_x.dtype=} with {dtype_effective_bits(out_x.dtype)} effective bits, "
                f"{out_y.dtype=} with {dtype_effective_bits(out_y.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}"
            )

        out_x_u = unsigned_view(out_x)
        out_y_u = unsigned_view(out_y)

    index_u = unsigned_view(index)

    index_1d = _flatten_nocopy(index_u, "index", order="C", strict=False)
    out_x_1d = _flatten_nocopy(out_x_u, "out_x", order="C", strict=False)
    out_y_1d = _flatten_nocopy(out_y_u, "out_y", order="C", strict=False)

    impl = build_batch(nbits, parallel=parallel)
    impl(index_1d, out_x_1d, out_y_1d)
    return out_x, out_y
