"""Public 2D Hilbert API.

This module provides the main user-facing API for 2D Hilbert encoding and decoding:

- ``hilbert_encode_2d``
- ``hilbert_decode_2d``

These accept either Python/NumPy scalar integers (scalar mode) or NumPy integer
arrays (array mode).

For advanced use (embedding Hilbert encode/decode into your own Numba-compiled
code), this module also exposes kernel accessors:

- ``get_hilbert_encode_2d_kernel``
- ``get_hilbert_decode_2d_kernel``

"""

import warnings
from collections.abc import Callable
from typing import cast

import numpy as np

from ._dispatch import (
    get_decode_2d_batch_builder,
    get_decode_2d_scalar_builder,
    get_encode_2d_batch_builder,
    get_encode_2d_scalar_builder,
)
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
from .types import IntArray, IntScalar, TileNBits2D


def hilbert_encode_2d(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out: IntArray | None = None,
    parallel: bool = False,
) -> int | IntArray:
    """Encode 2D integer coordinates to Hilbert indices.

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``x`` and ``y`` are scalar integers, returns a Python
    ``int``.
    - **Array mode**: if ``x`` and ``y`` are NumPy integer arrays, returns an array
    of unsigned indices with the same shape and supports ``out=``.

    Parameters
    ----------
    x, y
        Coordinates to encode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer arrays of identical shape.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis. For inputs outside that domain, only the low
        ``nbits`` bits of each coordinate are used.

        Must satisfy ``1 <= nbits <= 32``. If provided, it must also fit within
        the usable bits of the coordinate dtype.

        If ``None``:

        - Array mode: inferred from the coordinate dtype using its usable bit
        width, capped at 32. For example, ``uint16`` -> 16, ``int16`` -> 15
        (sign bit excluded), and ``uint64``/``int64`` -> 32.
        - Scalar mode: defaults to 32.

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input coordinate range.
    out
        Optional output array for array mode.

        Must have the same shape as ``x`` and ``y`` and an unsigned integer dtype
        wide enough to hold ``2 * nbits`` bits.

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
        unsigned integer type that can hold ``2 * nbits`` bits.

    Raises
    ------
    TypeError
        If ``x`` and ``y`` are not both scalars or not both arrays, if a
        non-integer input is provided, or if ``out`` is used in scalar mode.
    ValueError
        If ``nbits`` is invalid, if array inputs have mismatched shapes, or if
        ``out`` has the wrong shape or an insufficient dtype.
    """

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
        return _hilbert_encode_2d_scalar(cast(IntScalar, x), cast(IntScalar, y), nbits)

    return _hilbert_encode_2d_batch(
        cast(IntArray, x),
        cast(IntArray, y),
        nbits,
        out=out,
        parallel=parallel,
    )


def hilbert_decode_2d(
    index: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out_x: IntArray | None = None,
    out_y: IntArray | None = None,
    parallel: bool = False,
) -> tuple[int, int] | tuple[IntArray, IntArray]:
    """Decode Hilbert indices to 2D integer coordinates.

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``index`` is a scalar integer, returns ``(x, y)`` as
    Python ``int`` values.
    - **Array mode**: if ``index`` is a NumPy integer array, returns coordinate
    arrays with the same shape and supports ``out_x=`` / ``out_y=``.

    Parameters
    ----------
    index
        Hilbert index or indices to decode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer array.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis and a Hilbert index range of
        ``[0, 2**(2 * nbits))``. For indices outside that range, only the low
        ``2 * nbits`` bits are used.

        Must satisfy ``1 <= nbits <= 32``. If provided, it must also fit within
        the usable bits of the index dtype.

        If ``None``:

        - Array mode: inferred from the index dtype as half its usable bit width,
        capped at 32. For example, ``uint16`` -> 8, ``uint64`` -> 32, and
        ``int64`` -> 31 (sign bit excluded).
        - Scalar mode: defaults to 32.

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input index range.
    out_x, out_y
        Optional output coordinate arrays for array mode. Either provide both
        or neither.

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
    tuple[int, int] | tuple[IntArray, IntArray]
        Decoded coordinates.

        - Scalar mode: ``(x, y)`` as Python ``int`` values.
        - Array mode: ``(x, y)`` as unsigned integer arrays.

        If ``out_x`` and ``out_y`` are not provided in array mode, each result
        dtype is the smallest unsigned integer type that can hold ``nbits`` bits.

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
        if out_x is not None or out_y is not None:
            raise TypeError("out_x/out_y are only valid for array mode")
        if parallel:
            warnings.warn(
                "parallel=True has no effect in scalar mode",
                UserWarning,
                stacklevel=2,
            )
        return _hilbert_decode_2d_scalar(cast(IntScalar, index), nbits)

    return _hilbert_decode_2d_batch(
        cast(IntArray, index),
        nbits,
        out_x=out_x,
        out_y=out_y,
        parallel=parallel,
    )


def _hilbert_encode_2d_scalar(x: IntScalar, y: IntScalar, nbits: int | None) -> int:
    """Internal scalar 2D Hilbert encode."""

    if nbits is None:
        # For Python scalars, default to maximum
        nbits = MAX_NBITS_2D

    x_i, y_i = int(x), int(y)
    max_v = np.iinfo(np.uint32).max
    if x_i < 0 or y_i < 0 or x_i > max_v or y_i > max_v:
        raise ValueError(
            "Scalar inputs must be non-negative and fit in uint32; "
            f"got x={x_i}, y={y_i}"
        )

    builder = get_encode_2d_scalar_builder()
    impl = builder(nbits)
    return impl(np.uint32(x_i), np.uint32(y_i))


def _hilbert_decode_2d_scalar(index: IntScalar, nbits: int | None) -> tuple[int, int]:
    """Internal scalar 2D Hilbert decode."""

    if nbits is None:
        # For Python scalars, default to maximum
        nbits = MAX_NBITS_2D

    index_i = int(index)
    max_v = np.iinfo(np.uint64).max
    if index_i < 0 or index_i > max_v:
        raise ValueError(
            f"Scalar index must be non-negative and fit in uint64; got index={index_i}"
        )

    builder = get_decode_2d_scalar_builder()
    impl = builder(nbits)
    return impl(np.uint64(index_i))


def _hilbert_encode_2d_batch(
    x: IntArray,
    y: IntArray,
    nbits: int | None,
    *,
    out: IntArray | None = None,
    parallel: bool = False,
) -> IntArray:
    """Internal batch 2D Hilbert encode."""
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

    builder = get_encode_2d_batch_builder()
    impl = builder(nbits, parallel=parallel)
    impl(x_1d, y_1d, out_1d)

    return out


def _hilbert_decode_2d_batch(
    index: IntArray,
    nbits: int | None,
    *,
    out_x: IntArray | None = None,
    out_y: IntArray | None = None,
    parallel: bool = False,
) -> tuple[IntArray, IntArray]:
    """Internal batch 2D Hilbert decode."""

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

    index_y = unsigned_view(index)

    index_1d = _flatten_nocopy(index_y, "index", order="C", strict=False)
    out_x_1d = _flatten_nocopy(out_x_u, "out_x", order="C", strict=False)
    out_y_1d = _flatten_nocopy(out_y_u, "out_y", order="C", strict=False)

    builder = get_decode_2d_batch_builder()
    impl = builder(nbits, parallel=parallel)
    impl(index_1d, out_x_1d, out_y_1d)
    return out_x, out_y


def get_hilbert_encode_2d_kernel(
    nbits: int, *, tile_nbits: TileNBits2D | None = None
) -> Callable[[IntScalar, IntScalar], int]:
    """Return a Numba-compiled *scalar* 2D Hilbert encoder.

    This is the low-level kernel used by [`hilbert_encode_2d`][hilbertsfc.hilbert2d.hilbert_encode_2d] in scalar mode.
    It is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    tile_nbits
        Select the tile size (in bits) / kernel variant.

        - ``None`` (default): auto-select (``4`` for ``nbits <= 4`` or ``nbits == 8``, else ``7``).

        - ``7``: uses 7-bit compacted LUTs (128 KiB).
        - ``4``: uses 4-bit compacted LUTs (2 KiB).

        The 7-bit variant uses a larger LUT and is generally faster.
        The 4-bit variant uses a smaller LUT, which may be preferable in
        cache-intensive kernels.

    Returns
    -------
    callable
        A Numba-compiled function with signature ``(x: int, y: int) -> int``.
    """
    builder = get_encode_2d_scalar_builder()
    return builder(nbits, tile_nbits=tile_nbits)


def get_hilbert_decode_2d_kernel(
    nbits: int, *, tile_nbits: TileNBits2D | None = None
) -> Callable[[IntScalar], tuple[int, int]]:
    """Return a Numba-compiled *scalar* 2D Hilbert decoder.

    This is the low-level kernel used by [`hilbert_decode_2d`][hilbertsfc.hilbert2d.hilbert_decode_2d] in scalar mode.
    It is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    tile_nbits
        Select the tile size (in bits) / kernel variant.

        - ``None`` (default): auto-select (``4`` for ``nbits <= 4`` or ``nbits == 8``, else ``7``).

        - ``7``: uses 7-bit compacted LUTs (128 KiB).
        - ``4``: uses 4-bit compacted LUTs (2 KiB).

        The 7-bit variant uses a larger LUT and is generally faster.
        The 4-bit variant uses a smaller LUT, which may be preferable in
        cache-intensive kernels.

    Returns
    -------
    callable
        A Numba-compiled function with signature ``(index: int) -> (x: int, y: int)``.
    """
    builder = get_decode_2d_scalar_builder()
    return builder(nbits, tile_nbits=tile_nbits)
