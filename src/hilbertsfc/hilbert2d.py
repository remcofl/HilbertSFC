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
from ._typing import IntArray, IntScalar, TileNBits2D


def hilbert_encode_2d(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out: IntArray | None = None,
    parallel: bool = False,
) -> int | IntArray:
    """Encode 2D coordinates to a Hilbert index.

    This is a unified entrypoint:

    - **Scalar mode**: if ``x`` and ``y`` are scalar integers, returns a Python ``int``.
    - **Array mode**: if ``x`` and ``y`` are NumPy integer arrays, returns a NumPy
      array of indices (and supports ``out=``).

    Parameters
    ----------
    x, y
        Coordinates to encode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer arrays of identical shape.

        Boolean inputs are rejected.
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
        For best performance and tighter output dtypes, set this to the tightest
        bound that fits your coordinate range.

        Must satisfy ``1 <= nbits <= 32``. When specified, it must also
        fit in the effective bits of the largest coordinate dtype. Bits outside the
        domain are ignored.

        If ``None`` (default), inferred from the input dtype:

        - For arrays: uses the effective bits of the coordinate dtype, capped at 32.
            For example, ``uint16`` → 16 bits, ``int16`` → 15 bits (sign bit excluded),
            ``uint64`` or ``int64`` → 32 bits (algorithm maximum).
        - For Python scalars: defaults to 32.

        For array mode, if the inferred value exceeds the algorithm maximum
        (32 bits), a ``UserWarning`` is emitted and ``nbits`` is capped at 32.
    out
        Optional output array for array mode. Must have the same shape as ``x``/``y``
        and an integer dtype wide enough to hold ``2*nbits`` bits (unsigned).
        Not allowed in scalar mode.
    parallel
        Array mode: if ``True``, the underlying Numba kernel may use parallel
        execution.

        Scalar mode: accepted for API consistency, but ignored. If ``True``, a
        ``UserWarning`` is emitted.

        The number of threads can be controlled with the environment variable
        `NUMBA_NUM_THREADS` or during runtime with `numba.set_num_threads()`.

    Returns
    -------
    int or numpy.ndarray
        - Scalar mode: the Hilbert index as Python ``int``.
        - Array mode: an array of unsigned indices.

        When ``out`` is not provided, the output dtype is chosen automatically:

        - ``uint8``  if ``nbits <= 4``
        - ``uint16`` if ``nbits <= 8``
        - ``uint32`` if ``nbits <= 16``
        - ``uint64`` otherwise up to ``nbits <= 32``

    Raises
    ------
    TypeError
        If ``x`` and ``y`` are not both scalars or not both arrays, if boolean
        inputs are provided, or if ``out`` is used in scalar mode.
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
    out_xs: IntArray | None = None,
    out_ys: IntArray | None = None,
    parallel: bool = False,
) -> tuple[int, int] | tuple[IntArray, IntArray]:
    """Decode a Hilbert index to 2D coordinates.

    This is a unified entrypoint:

    - **Scalar mode**: if ``index`` is a scalar integer, returns ``(x, y)`` as
      Python ``int``.
    - **Array mode**: if ``index`` is a NumPy integer array, returns ``(xs, ys)``
      as NumPy arrays and supports ``out_xs=``/``out_ys=``.

    Parameters
    ----------
    index
        Hilbert index or indices to decode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer array.

        Boolean inputs are rejected.
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
        For best performance and tighter output dtypes, set this to the tightest
        bound that fits your coordinate range.

        Must satisfy ``1 <= nbits <= 32``. When specified, it must not
        exceed the effective bits supported by the index dtype. Bits outside the
        domain are ignored.

        If ``None`` (default), inferred from the index dtype:

        - For arrays: uses ``index_bits / 2``, where index_bits is the effective
            bits of the index dtype. For example, ``uint16`` → 8 bits, ``uint64`` → 32 bits,
            ``int64`` → 31 bits (sign bit excluded).
        - For Python scalars: defaults to 32.
    out_xs, out_ys
        Optional output buffers for array mode. Either provide both or neither.
        Must have the same shape as ``index`` and an integer dtype wide enough to
        hold values in ``[0, 2**nbits)`` (unsigned).
        Not allowed in scalar mode.
    parallel
        Array mode: if ``True``, the underlying Numba kernel may use parallel
        execution.

        Scalar mode: accepted for API consistency, but ignored. If ``True``, a
        ``UserWarning`` is emitted.

        The number of threads can be controlled with the environment variable
        `NUMBA_NUM_THREADS` or during runtime with `numba.set_num_threads()`.

    Returns
    -------
    (int, int) or (numpy.ndarray, numpy.ndarray)
        The decoded coordinates.

        When output buffers are not provided in array mode, the coordinate dtype is
        chosen automatically:

        - ``uint8``  if ``nbits <= 8``
        - ``uint16`` if ``nbits <= 16``
        - ``uint32`` if ``nbits <= 32``

    Raises
    ------
    TypeError
        If boolean inputs are provided or if output buffers are used in scalar mode.
    ValueError
        If ``nbits`` is invalid or does not fit in the provided index/coord dtypes,
        or if output buffers are inconsistent (missing one of the pair) or have
        incorrect shapes.
    """

    index_is_scalar = is_int_scalar_or_0d_array(index)
    if index_is_scalar:
        if out_xs is not None or out_ys is not None:
            raise TypeError("out_xs/out_ys are only valid for array mode")
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
        out_xs=out_xs,
        out_ys=out_ys,
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
    xs: IntArray,
    ys: IntArray,
    nbits: int | None,
    *,
    out: IntArray | None = None,
    parallel: bool = False,
) -> IntArray:
    """Internal batch 2D Hilbert encode."""
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")

    max_coord_nbits = max(
        dtype_effective_bits(xs.dtype), dtype_effective_bits(ys.dtype)
    )
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
                f"nbits={nbits} does not fit in coordinate dtypes; "
                f"got xs.dtype={xs.dtype} with {dtype_effective_bits(xs.dtype)} effective bits, "
                f"ys.dtype={ys.dtype} with {dtype_effective_bits(ys.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}."
            )

    if out is None:
        index_dtype = choose_uint_index_dtype(nbits=nbits, dims=2)
        out_u = out = np.empty(xs.shape, dtype=index_dtype, order="C")
    else:
        if out.shape != xs.shape:
            raise ValueError(
                f"out must have the same shape as xs/ys; got out.shape={out.shape}, xs.shape={xs.shape}"
            )
        max_index_nbits = max_nbits_for_index_dtype(out.dtype, dims=2)
        if nbits > max_index_nbits:
            viable_dtype = np.dtype(choose_uint_index_dtype(nbits=nbits, dims=2))
            raise ValueError(
                f"nbits={nbits} does not fit in out dtype; got out.dtype={out.dtype} "
                f"which supports up to nbits={max_index_nbits}; "
                f"consider using {viable_dtype} or a wider dtype, or reduce nbits to fit the out dtype."
            )
        out_u = unsigned_view(out)

    xs_u = unsigned_view(xs)
    ys_u = unsigned_view(ys)

    xs_1d = _flatten_nocopy(xs_u, "xs", order="C", strict=False)
    ys_1d = _flatten_nocopy(ys_u, "ys", order="C", strict=False)
    out_1d = _flatten_nocopy(out_u, "out", order="C", strict=False)

    builder = get_encode_2d_batch_builder()
    impl = builder(nbits, parallel=parallel)
    impl(xs_1d, ys_1d, out_1d)

    return out


def _hilbert_decode_2d_batch(
    indices: IntArray,
    nbits: int | None,
    *,
    out_xs: IntArray | None = None,
    out_ys: IntArray | None = None,
    parallel: bool = False,
) -> tuple[IntArray, IntArray]:
    """Internal batch 2D Hilbert decode."""

    max_index_nbits = max_nbits_for_index_dtype(indices.dtype, dims=2)
    if nbits is None:
        nbits = max_index_nbits
    else:
        validate_nbits_2d(nbits)
        if nbits > max_index_nbits:
            raise ValueError(
                f"nbits={nbits} exceeds the effective bits of the index dtype; "
                f"got indices.dtype={indices.dtype} which supports up to {max_index_nbits} bits for 2D coordinates. "
                f"max nbits is {max_index_nbits}."
            )

    if (out_xs is None) != (out_ys is None):
        raise ValueError("out_xs and out_ys must be provided together")

    if out_xs is None or out_ys is None:
        coord_dtype = choose_uint_coord_dtype(nbits=nbits)
        out_xs_u = out_xs = np.empty(indices.shape, dtype=coord_dtype, order="C")
        out_ys_u = out_ys = np.empty(indices.shape, dtype=coord_dtype, order="C")
    else:
        if out_xs.shape != indices.shape or out_ys.shape != indices.shape:
            raise ValueError(
                "out_xs and out_ys must have the same shape as indices; "
                f"got indices.shape={indices.shape}, out_xs.shape={out_xs.shape}, out_ys.shape={out_ys.shape}"
            )
        max_coord_nbits = min(
            dtype_effective_bits(out_xs.dtype), dtype_effective_bits(out_ys.dtype)
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"nbits={nbits} does not fit in out_xs/out_ys dtypes; "
                f"got out_xs.dtype={out_xs.dtype} with {dtype_effective_bits(out_xs.dtype)} effective bits, "
                f"out_ys.dtype={out_ys.dtype} with {dtype_effective_bits(out_ys.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}"
            )

        out_xs_u = unsigned_view(out_xs)
        out_ys_u = unsigned_view(out_ys)

    indices_u = unsigned_view(indices)

    indices_1d = _flatten_nocopy(indices_u, "indices", order="C", strict=False)
    out_xs_1d = _flatten_nocopy(out_xs_u, "out_xs", order="C", strict=False)
    out_ys_1d = _flatten_nocopy(out_ys_u, "out_ys", order="C", strict=False)

    builder = get_decode_2d_batch_builder()
    impl = builder(nbits, parallel=parallel)
    impl(indices_1d, out_xs_1d, out_ys_1d)
    assert out_xs is not None and out_ys is not None
    return out_xs, out_ys


def get_hilbert_encode_2d_kernel(
    nbits: int, *, tile_nbits: TileNBits2D | None = None
) -> Callable[[IntScalar, IntScalar], int]:
    """Return a Numba-compiled *scalar* 2D Hilbert encoder.

    This is the low-level kernel used by :func:`hilbert_encode_2d` in scalar mode.
    It is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    tile_nbits
        Select the tile size (in bits) / kernel variant.

        - ``None`` (default): auto-select (``4`` for ``nbits <= 4`` or ``nbits == 8``, else ``7``).

        - ``7``: uses 7-bit compacted LUTs (~65 KiB).
        - ``4``: uses 4-bit compacted LUTs (~1 KiB).

        The 7-bit variant uses a larger LUT and is generally faster.
        The 4-bit variant uses a smaller LUT, which may be preferable in
        cache-sensitive environments.

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

    This is the low-level kernel used by :func:`hilbert_decode_2d` in scalar mode.
    It is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    tile_nbits
        Select the tile size (in bits) / kernel variant.

        - ``None`` (default): auto-select (``4`` for ``nbits <= 4`` or ``nbits == 8``, else ``7``).

        - ``7``: uses 7-bit compacted LUTs (~65 KiB).
        - ``4``: uses 4-bit compacted LUTs (~1 KiB).

        The 7-bit variant uses a larger LUT and is generally faster.
        The 4-bit variant uses a smaller LUT, which may be preferable in
        cache-sensitive environments.

    Returns
    -------
    callable
        A Numba-compiled function with signature ``(index: int) -> (x: int, y: int)``.
    """
    builder = get_decode_2d_scalar_builder()
    return builder(nbits, tile_nbits=tile_nbits)
