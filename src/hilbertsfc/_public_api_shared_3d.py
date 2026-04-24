"""Shared 3D public API plumbing for scalar+array wrappers."""

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
from ._nbits import MAX_NBITS_3D, validate_nbits_3d
from ._public_api_adapters import Decode3DAdapter, Encode3DAdapter
from .types import IntArray, IntScalar


def encode_3d_api(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    z: IntScalar | IntArray,
    *,
    nbits: int | None,
    out: IntArray | None,
    parallel: bool,
    adapter: Encode3DAdapter,
) -> int | IntArray:
    """Shared implementation for public 3D encoders."""

    x_is_scalar = is_int_scalar_or_0d_array(x)
    y_is_scalar = is_int_scalar_or_0d_array(y)
    z_is_scalar = is_int_scalar_or_0d_array(z)
    if not (x_is_scalar == y_is_scalar == z_is_scalar):
        raise TypeError("x, y, z must all be scalars or all be arrays")

    if x_is_scalar:
        if out is not None:
            raise TypeError("out is only valid for array mode")
        if parallel:
            warnings.warn(
                "parallel=True has no effect in scalar mode",
                UserWarning,
                stacklevel=2,
            )
        return _encode_3d_scalar(
            cast(IntScalar, x),
            cast(IntScalar, y),
            cast(IntScalar, z),
            nbits,
            adapter,
        )

    return _encode_3d_batch(
        cast(IntArray, x),
        cast(IntArray, y),
        cast(IntArray, z),
        nbits,
        out=out,
        parallel=parallel,
        adapter=adapter,
    )


def decode_3d_api(
    index: IntScalar | IntArray,
    *,
    nbits: int | None,
    out_x: IntArray | None,
    out_y: IntArray | None,
    out_z: IntArray | None,
    parallel: bool,
    adapter: Decode3DAdapter,
) -> tuple[int, int, int] | tuple[IntArray, IntArray, IntArray]:
    """Shared implementation for public 3D decoders."""

    index_is_scalar = is_int_scalar_or_0d_array(index)
    if index_is_scalar:
        if out_x is not None or out_y is not None or out_z is not None:
            raise TypeError("out_x/out_y/out_z are only valid for array mode")
        if parallel:
            warnings.warn(
                "parallel=True has no effect in scalar mode",
                UserWarning,
                stacklevel=2,
            )
        return _decode_3d_scalar(cast(IntScalar, index), nbits, adapter)

    return _decode_3d_batch(
        cast(IntArray, index),
        nbits,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        parallel=parallel,
        adapter=adapter,
    )


def _encode_3d_scalar(
    x: IntScalar,
    y: IntScalar,
    z: IntScalar,
    nbits: int | None,
    adapter: Encode3DAdapter,
) -> int:
    if nbits is None:
        nbits = MAX_NBITS_3D

    x_i, y_i, z_i = int(x), int(y), int(z)
    max_v = np.iinfo(np.uint32).max
    if x_i < 0 or y_i < 0 or z_i < 0 or x_i > max_v or y_i > max_v or z_i > max_v:
        raise ValueError(
            "Scalar inputs must be non-negative and fit in uint32; "
            f"got x={x_i}, y={y_i}, z={z_i}"
        )

    impl = adapter.build_scalar(nbits)
    return impl(np.uint32(x_i), np.uint32(y_i), np.uint32(z_i))


def _decode_3d_scalar(
    index: IntScalar,
    nbits: int | None,
    adapter: Decode3DAdapter,
) -> tuple[int, int, int]:
    if nbits is None:
        nbits = MAX_NBITS_3D

    index_i = int(index)
    max_v = np.iinfo(np.uint64).max
    if index_i < 0 or index_i > max_v:
        raise ValueError(
            f"Scalar index must be non-negative and fit in uint64; got index={index_i}"
        )

    impl = adapter.build_scalar(nbits)
    return impl(np.uint64(index_i))


def _encode_3d_batch(
    x: IntArray,
    y: IntArray,
    z: IntArray,
    nbits: int | None,
    *,
    out: IntArray | None,
    parallel: bool,
    adapter: Encode3DAdapter,
) -> IntArray:
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"x, y, z must have the same shape; got {x.shape=}, {y.shape=}, {z.shape=}"
        )

    max_coord_nbits = max(
        dtype_effective_bits(x.dtype),
        dtype_effective_bits(y.dtype),
        dtype_effective_bits(z.dtype),
    )
    if nbits is None:
        nbits = max_coord_nbits
        if nbits > MAX_NBITS_3D:
            warnings.warn(
                f"The maximum effective bits of the coordinate dtype is {nbits}, "
                f"which exceeds the algorithm maximum of {MAX_NBITS_3D}. "
                f"Using nbits={MAX_NBITS_3D} instead. This means that excess bits in the input coordinates "
                f"will be ignored. To silence this warning, explicitly set nbits<={MAX_NBITS_3D}.",
                UserWarning,
                stacklevel=2,
            )
            nbits = MAX_NBITS_3D
    else:
        validate_nbits_3d(nbits)
        if nbits > max_coord_nbits:
            raise ValueError(
                f"nbits={nbits} does not fit in coordinate dtypes; "
                f"got {x.dtype=} with {dtype_effective_bits(x.dtype)} effective bits, "
                f"{y.dtype=} with {dtype_effective_bits(y.dtype)} effective bits; "
                f"{z.dtype=} with {dtype_effective_bits(z.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}."
            )

    if out is None:
        index_dtype = choose_uint_index_dtype(nbits=nbits, dims=3)
        out_u = out = np.empty(x.shape, dtype=index_dtype, order="C")
    else:
        if out.shape != x.shape:
            raise ValueError(
                f"out must have the same shape as x/y/z; got {out.shape=}, {x.shape=}"
            )
        max_index_nbits = max_nbits_for_index_dtype(out.dtype, dims=3)
        if nbits > max_index_nbits:
            viable_dtype = np.dtype(choose_uint_index_dtype(nbits=nbits, dims=3))
            raise ValueError(
                f"nbits={nbits} does not fit in out dtype; got {out.dtype=} "
                f"which supports up to nbits={max_index_nbits}; "
                f"consider using {viable_dtype} out dtype, or reduce nbits to fit the out dtype."
            )
        out_u = unsigned_view(out)

    x_u = unsigned_view(x)
    y_u = unsigned_view(y)
    z_u = unsigned_view(z)

    x_1d = _flatten_nocopy(x_u, "x", order="C", strict=False)
    y_1d = _flatten_nocopy(y_u, "y", order="C", strict=False)
    z_1d = _flatten_nocopy(z_u, "z", order="C", strict=False)
    out_1d = _flatten_nocopy(out_u, "out", order="C", strict=False)

    impl = adapter.build_batch(nbits, parallel, out.dtype)
    impl(x_1d, y_1d, z_1d, out_1d)
    return out


def _decode_3d_batch(
    index: IntArray,
    nbits: int | None,
    *,
    out_x: IntArray | None,
    out_y: IntArray | None,
    out_z: IntArray | None,
    parallel: bool,
    adapter: Decode3DAdapter,
) -> tuple[IntArray, IntArray, IntArray]:
    max_index_nbits = max_nbits_for_index_dtype(index.dtype, dims=3)
    if nbits is None:
        nbits = max_index_nbits
    else:
        validate_nbits_3d(nbits)
        if nbits > max_index_nbits:
            raise ValueError(
                f"{nbits=} exceeds the effective bits of the index dtype; "
                f"got {index.dtype=} which supports up to {max_index_nbits} bits for 3D coordinates. "
                f"max nbits is {max_index_nbits}."
            )

    provided = (out_x is not None) + (out_y is not None) + (out_z is not None)
    if provided not in (0, 3):
        raise ValueError("out_x, out_y, out_z must be provided together")

    if out_x is None or out_y is None or out_z is None:
        coord_dtype = choose_uint_coord_dtype(nbits=nbits)
        out_x_u = out_x = np.empty(index.shape, dtype=coord_dtype, order="C")
        out_y_u = out_y = np.empty(index.shape, dtype=coord_dtype, order="C")
        out_z_u = out_z = np.empty(index.shape, dtype=coord_dtype, order="C")
    else:
        if (
            out_x.shape != index.shape
            or out_y.shape != index.shape
            or out_z.shape != index.shape
        ):
            raise ValueError(
                "out_x, out_y, out_z must have the same shape as index; "
                f"got {index.shape=}, {out_x.shape=}, {out_y.shape=}, {out_z.shape=}"
            )
        max_coord_nbits = min(
            dtype_effective_bits(out_x.dtype),
            dtype_effective_bits(out_y.dtype),
            dtype_effective_bits(out_z.dtype),
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in out_x/out_y dtypes; "
                f"got {out_x.dtype=} with {dtype_effective_bits(out_x.dtype)} effective bits, "
                f"{out_y.dtype=} with {dtype_effective_bits(out_y.dtype)} effective bits; "
                f"{out_z.dtype=} with {dtype_effective_bits(out_z.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}"
            )

        out_x_u = unsigned_view(out_x)
        out_y_u = unsigned_view(out_y)
        out_z_u = unsigned_view(out_z)

    index_u = unsigned_view(index)

    index_1d = _flatten_nocopy(index_u, "index", order="C", strict=False)
    out_x_1d = _flatten_nocopy(out_x_u, "out_x", order="C", strict=False)
    out_y_1d = _flatten_nocopy(out_y_u, "out_y", order="C", strict=False)
    out_z_1d = _flatten_nocopy(out_z_u, "out_z", order="C", strict=False)

    impl = adapter.build_batch(nbits, parallel, index.dtype)
    impl(index_1d, out_x_1d, out_y_1d, out_z_1d)
    return out_x, out_y, out_z
