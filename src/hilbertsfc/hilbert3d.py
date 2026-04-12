"""Public 3D Hilbert API.

This module provides the main user-facing API for 3D Hilbert encoding and decoding:

- ``hilbert_encode_3d``
- ``hilbert_decode_3d``

These accept either Python/NumPy scalar integers (scalar mode) or NumPy integer
arrays (array mode).

For advanced use (embedding Hilbert encode/decode into your own Numba-compiled
code), this module also exposes kernel accessors:

- ``get_hilbert_encode_3d_kernel``
- ``get_hilbert_decode_3d_kernel``

"""

import warnings
from collections.abc import Callable
from typing import cast

import numpy as np

from ._dispatch import (
    get_decode_3d_batch_builder,
    get_decode_3d_scalar_builder,
    get_encode_3d_batch_builder,
    get_encode_3d_scalar_builder,
)
from ._dtype import (
    choose_lut_dtype_for_index_dtype,
    choose_uint_coord_dtype,
    choose_uint_index_dtype,
    dtype_effective_bits,
    max_nbits_for_index_dtype,
    unsigned_view,
)
from ._flatten import flatten_nocopy as _flatten_nocopy
from ._input_checks import is_int_scalar_or_0d_array
from ._nbits import MAX_NBITS_3D, validate_nbits_3d
from .types import IntArray, IntScalar, LutUIntDTypeLike


def hilbert_encode_3d(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    z: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out: IntArray | None = None,
    parallel: bool = False,
) -> int | IntArray:
    """Encode 3D integer coordinates to Hilbert indices.

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``x``, ``y``, and ``z`` are scalar integers, returns a
    Python ``int``.
    - **Array mode**: if ``x``, ``y``, and ``z`` are NumPy integer arrays, returns
    an array of unsigned indices with the same shape and supports ``out=``.

    Parameters
    ----------
    x, y, z
        Coordinates to encode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer arrays of identical shape.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis. For inputs outside that domain, only the low
        ``nbits`` bits of each coordinate are used.

        Must satisfy ``1 <= nbits <= 21``. If provided, it must also fit within
        the usable bits of the coordinate dtype.

        If ``None``:

        - Array mode: inferred from the coordinate dtype using its usable bit
        width, capped at 21. For example, ``uint16`` -> 16, ``int16`` -> 15
        (sign bit excluded), and ``uint64``/``int64`` -> 21.
        - Scalar mode: defaults to 21.

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input coordinate range.
    out
        Optional output array for array mode.

        Must have the same shape as ``x``, ``y``, and ``z`` and an unsigned integer
        dtype wide enough to hold ``3 * nbits`` bits.

        Not allowed in scalar mode.
    parallel
        Controls whether the array-mode Numba kernel may execute in parallel.

        In scalar mode, this argument is accepted for API consistency but has no
        effect. If ``True``, a ``UserWarning`` is emitted.

        The number of threads can be controlled with ``NUMBA_NUM_THREADS`` or
        ``numba.set_num_threads()``.

    Returns
    -------
    int | IntArray
        Hilbert index or indices.

        - Scalar mode: the Hilbert index as a Python ``int``.
        - Array mode: an array of unsigned indices.

        If ``out`` is not provided in array mode, the output dtype is the smallest
        unsigned integer type that can hold ``3 * nbits`` bits.

    Raises
    ------
    TypeError
        If ``x``, ``y``, and ``z`` are not all scalars or not all arrays, if a
        non-integer input is provided, or if ``out`` is used in scalar mode.
    ValueError
        If ``nbits`` is invalid, if array inputs have mismatched shapes, or if
        ``out`` has the wrong shape or an insufficient dtype.
    """
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
        return _hilbert_encode_3d_scalar(
            cast(IntScalar, x),
            cast(IntScalar, y),
            cast(IntScalar, z),
            nbits,
        )

    return _hilbert_encode_3d_batch(
        cast(IntArray, x),
        cast(IntArray, y),
        cast(IntArray, z),
        nbits,
        out=out,
        parallel=parallel,
    )


def hilbert_decode_3d(
    index: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out_x: IntArray | None = None,
    out_y: IntArray | None = None,
    out_z: IntArray | None = None,
    parallel: bool = False,
) -> tuple[int, int, int] | tuple[IntArray, IntArray, IntArray]:
    """Decode Hilbert indices to 3D integer coordinates.

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``index`` is a scalar integer, returns ``(x, y, z)`` as
    Python ``int`` values.
    - **Array mode**: if ``index`` is a NumPy integer array, returns coordinate
    arrays with the same shape and supports ``out_x=``, ``out_y=``, and
    ``out_z=``.

    Parameters
    ----------
    index
        Hilbert index or indices to decode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer array.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis and a Hilbert index range of
        ``[0, 2**(3 * nbits))``. For indices outside that range, only the low
        ``3 * nbits`` bits are used.

        Must satisfy ``1 <= nbits <= 21``. If provided, it must also fit within
        the usable bits of the index dtype.

        If ``None``:

        - Array mode: inferred from the index dtype as one third of its usable bit
        width, capped at 21. For example, ``uint16`` -> 5, and ``int64`` -> 21.
        - Scalar mode: defaults to 21.

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input index range.
    out_x, out_y, out_z
        Optional coordinate output arrays for array mode. Either provide all
        three or none.

        Each must have the same shape as ``index`` and an unsigned integer dtype
        wide enough to hold ``nbits`` bits.

        Not allowed in scalar mode.
    parallel
        Controls whether the array-mode Numba kernel may execute in parallel.

        In scalar mode, this argument is accepted for API consistency but has no
        effect. If ``True``, a ``UserWarning`` is emitted.

        The number of threads can be controlled with ``NUMBA_NUM_THREADS`` or
        ``numba.set_num_threads()``.

    Returns
    -------
    tuple[int, int, int] | tuple[IntArray, IntArray, IntArray]
        Decoded coordinates.

        - Scalar mode: ``(x, y, z)`` as Python ``int`` values.
        - Array mode: ``(x, y, z)`` as unsigned integer arrays.

        If ``out_x``, ``out_y``, and ``out_z`` are not provided in array mode,
        each result dtype is the smallest unsigned integer type that can hold
        ``nbits`` bits.

    Raises
    ------
    TypeError
        If a non-integer input is provided or if output buffers are used in scalar
        mode.
    ValueError
        If ``nbits`` is invalid, if it does not fit in the index or output dtypes,
        or if output buffers are inconsistent or have incorrect shapes.
    """

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
        return _hilbert_decode_3d_scalar(cast(IntScalar, index), nbits)

    return _hilbert_decode_3d_batch(
        cast(IntArray, index),
        nbits,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        parallel=parallel,
    )


def _hilbert_encode_3d_scalar(
    x: IntScalar,
    y: IntScalar,
    z: IntScalar,
    nbits: int | None,
) -> int:
    """Internal scalar 3D Hilbert encode."""

    if nbits is None:
        # For Python scalars, default to maximum
        nbits = MAX_NBITS_3D

    x_i, y_i, z_i = int(x), int(y), int(z)
    max_v = np.iinfo(np.uint32).max
    if x_i < 0 or y_i < 0 or z_i < 0 or x_i > max_v or y_i > max_v or z_i > max_v:
        raise ValueError(
            "Scalar inputs must be non-negative and fit in uint32; "
            f"got x={x_i}, y={y_i}, z={z_i}"
        )

    builder = get_encode_3d_scalar_builder()
    impl = builder(nbits)
    return impl(np.uint32(x_i), np.uint32(y_i), np.uint32(z_i))


def _hilbert_decode_3d_scalar(
    index: IntScalar,
    nbits: int | None,
) -> tuple[int, int, int]:
    """Internal scalar 3D Hilbert decode."""

    if nbits is None:
        # For Python scalars, default to maximum
        nbits = MAX_NBITS_3D

    index_i = int(index)
    max_v = np.iinfo(np.uint64).max
    if index_i < 0 or index_i > max_v:
        raise ValueError(
            f"Scalar index must be non-negative and fit in uint64; got index={index_i}"
        )

    builder = get_decode_3d_scalar_builder()
    impl = builder(nbits)
    return impl(np.uint64(index_i))


def _hilbert_encode_3d_batch(
    x: IntArray,
    y: IntArray,
    z: IntArray,
    nbits: int | None,
    *,
    out: IntArray | None = None,
    parallel: bool = False,
) -> IntArray:
    """Internal batch 3D Hilbert encode."""

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

    builder = get_encode_3d_batch_builder()
    impl = builder(
        nbits,
        lut_dtype=choose_lut_dtype_for_index_dtype(out.dtype),
        parallel=parallel,
    )
    impl(x_1d, y_1d, z_1d, out_1d)
    return out


def _hilbert_decode_3d_batch(
    index: IntArray,
    nbits: int | None,
    *,
    out_x: IntArray | None = None,
    out_y: IntArray | None = None,
    out_z: IntArray | None = None,
    parallel: bool = False,
) -> tuple[IntArray, IntArray, IntArray]:
    """Internal batch 3D Hilbert decode."""

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

    builder = get_decode_3d_batch_builder()
    impl = builder(
        nbits,
        lut_dtype=choose_lut_dtype_for_index_dtype(index.dtype),
        parallel=parallel,
    )
    impl(index_1d, out_x_1d, out_y_1d, out_z_1d)
    return out_x, out_y, out_z


def get_hilbert_encode_3d_kernel(
    nbits: int, *, lut_dtype: LutUIntDTypeLike = np.uint16
) -> Callable[[IntScalar, IntScalar, IntScalar], IntScalar]:
    """Return a Numba-compiled *scalar* 3D Hilbert encoder.

    This is the low-level kernel used by [`hilbert_encode_3d`][hilbertsfc.hilbert3d.hilbert_encode_3d] in scalar mode
    and is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    lut_dtype
        Element dtype used for the internal lookup tables.

        Default is ``uint16`` (smallest; 3 KiB LUT footprint). When fusing into
        a cache-intensive kernel, keeping the LUT small can help.

        For best throughput, it is usually better to match ``lut_dtype``
        to your index dtype (``uint16``/``uint32``/``uint64``), which increases LUT
        size (6 KiB for ``uint32``, 12 KiB for ``uint64``) but reduces widening.
        In isolation, even the larger LUTs typically fit comfortably within the
        per-core L1 data cache of modern CPUs.

    Returns
    -------
    callable
        A Numba-compiled function with signature ``(x: int, y: int, z: int) -> int``.
    """

    builder = get_encode_3d_scalar_builder()
    return builder(nbits, lut_dtype=lut_dtype)


def get_hilbert_decode_3d_kernel(
    nbits: int, *, lut_dtype: LutUIntDTypeLike = np.uint16
) -> Callable[[IntScalar], tuple[int, int, int]]:
    """Return a Numba-compiled *scalar* 3D Hilbert decoder.

    This is the low-level kernel used by [`hilbert_decode_3d`][hilbertsfc.hilbert3d.hilbert_decode_3d] in scalar mode
    and is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    lut_dtype
        Element dtype used for the internal lookup tables.

        See [`get_hilbert_encode_3d_kernel`][hilbertsfc.hilbert3d.get_hilbert_encode_3d_kernel] for the performance tradeoff.

    Returns
    -------
    callable
        A Numba-compiled function with signature ``(index: int) -> (x: int, y: int, z: int)``.
    """
    builder = get_decode_3d_scalar_builder()
    return builder(nbits, lut_dtype=lut_dtype)
