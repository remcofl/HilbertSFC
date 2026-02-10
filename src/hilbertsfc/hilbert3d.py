"""Public 3D Hilbert API.

This module provides the main user-facing API for 2D Hilbert encoding and decoding:

- ``hilbert_encode_3d``
- ``hilbert_decode_3d``

These accept either Python/NumPy scalar integers (scalar mode) or NumPy integer
arrays (array mode).

For advanced use (embedding Hilbert encode/decode into your own Numba-compiled
code), this module also exposes kernel accessors:

- ``get_hilbert_encode_3d_kernel``
- ``get_hilbert_decode_3d_kernel``

"""

from __future__ import annotations

import warnings
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
from ._input_checks import is_int_scalar_or_0d_array as _is_int_scalar_or_0d_array
from ._typing import IntArray, IntScalar, LutUIntDTypeLike


def hilbert_encode_3d(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    z: IntScalar | IntArray,
    nbits: int,
    *,
    out: IntArray | None = None,
    parallel: bool = False,
) -> int | IntArray:
    """Encode 3D coordinates to a Hilbert index.

    This is a unified entrypoint:

    - **Scalar mode**: if ``x``, ``y``, ``z`` are scalar integers, returns a Python
      ``int``.
    - **Array mode**: if ``x``, ``y``, ``z`` are NumPy integer arrays, returns a
      NumPy array of indices (and supports ``out=``).

    Parameters
    ----------
    x, y, z
        Coordinates to encode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer arrays of identical shape.

        Boolean inputs are rejected.
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
        Must satisfy ``1 <= nbits <= 21``. Bits outside the domain are ignored.
    out
        Optional output array for array mode. Must have the same shape as
        ``x``/``y``/``z`` and an integer dtype wide enough to hold
        ``3*nbits`` bits (unsigned). Not allowed in scalar mode.
    parallel
        Array mode: if ``True``, the underlying Numba kernel may use parallel
        execution.

        Scalar mode: accepted for API consistency, but ignored. If ``True``, a
        ``UserWarning`` is emitted.

    Returns
    -------
    int or numpy.ndarray
        - Scalar mode: the Hilbert index as Python ``int``.
        - Array mode: an array of unsigned indices.

        When ``out`` is not provided, the output dtype is chosen automatically:

        - ``uint8``  if ``3*nbits <= 8``  (i.e. ``nbits <= 2``)
        - ``uint16`` if ``3*nbits <= 16`` (i.e. ``nbits <= 5``)
        - ``uint32`` if ``3*nbits <= 32`` (i.e. ``nbits <= 10``)
        - ``uint64`` otherwise (up to ``nbits <= 21``)

    Raises
    ------
    TypeError
        If inputs are mixed between scalar and array modes, if boolean inputs are
        provided, or if ``out`` is used in scalar mode.
    ValueError
        If ``nbits`` is invalid, if array inputs have mismatched shapes, or if
        ``out`` has the wrong shape or an insufficient dtype.
    """
    x_is_scalar = _is_int_scalar_or_0d_array(x)
    y_is_scalar = _is_int_scalar_or_0d_array(y)
    z_is_scalar = _is_int_scalar_or_0d_array(z)
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
        builder = get_encode_3d_scalar_builder()
        impl = builder(nbits)
        return int(impl(int(x), int(y), int(z)))

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
    nbits: int,
    *,
    out_xs: IntArray | None = None,
    out_ys: IntArray | None = None,
    out_zs: IntArray | None = None,
    parallel: bool = False,
) -> tuple[int, int, int] | tuple[IntArray, IntArray, IntArray]:
    """Decode a Hilbert index to 3D coordinates.

    This is a unified entrypoint:

    - **Scalar mode**: if ``index`` is a scalar integer, returns ``(x, y, z)`` as
      Python ``int``.
    - **Array mode**: if ``index`` is a NumPy integer array, returns ``(xs, ys, zs)``
      as NumPy arrays and supports ``out_xs=``/``out_ys=``/``out_zs=``.

    Parameters
    ----------
    index
        Hilbert index or indices to decode.

        - Scalar mode: Python ``int`` or NumPy integer scalar.
        - Array mode: NumPy integer array.

        Boolean inputs are rejected.
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
        Must satisfy ``1 <= nbits <= 21``. Bits outside the domain are ignored.
    out_xs, out_ys, out_zs
        Optional output buffers for array mode. Provide all three or none.
        Must have the same shape as ``index`` and an integer dtype wide enough to
        hold values in ``[0, 2**nbits)`` (unsigned). Not allowed in scalar mode.
    parallel
        Array mode: if ``True``, the underlying Numba kernel may use parallel
        execution.

        Scalar mode: accepted for API consistency, but ignored. If ``True``, a
        ``UserWarning`` is emitted.

    Returns
    -------
    (int, int, int) or (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The decoded coordinates.

        When output buffers are not provided in array mode, the coordinate dtype is
        chosen automatically:

        - ``uint8``  if ``nbits <= 8``
        - ``uint16`` if ``nbits <= 16``
        - ``uint32`` if ``nbits <= 21``

    Raises
    ------
    TypeError
        If boolean inputs are provided or if output buffers are used in scalar mode.
    ValueError
        If ``nbits`` is invalid or does not fit in the provided index/coord dtypes,
        or if output buffers are inconsistent (not all three provided) or have
        incorrect shapes.
    """
    index_is_scalar = _is_int_scalar_or_0d_array(index)
    if index_is_scalar:
        if out_xs is not None or out_ys is not None or out_zs is not None:
            raise TypeError("out_xs/out_ys/out_zs are only valid for batch decode")
        if parallel:
            warnings.warn(
                "parallel=True has no effect in scalar mode",
                UserWarning,
                stacklevel=2,
            )
        builder = get_decode_3d_scalar_builder()
        impl = builder(nbits)
        x, y, z = impl(int(index))
        return int(x), int(y), int(z)

    return _hilbert_decode_3d_batch(
        cast(IntArray, index),
        nbits,
        out_xs=out_xs,
        out_ys=out_ys,
        out_zs=out_zs,
        parallel=parallel,
    )


def _hilbert_encode_3d_batch(
    xs: IntArray,
    ys: IntArray,
    zs: IntArray,
    nbits: int,
    *,
    out: IntArray | None = None,
    parallel: bool = False,
) -> IntArray:
    if xs.shape != ys.shape or xs.shape != zs.shape:
        raise ValueError("xs, ys, zs must have the same shape")

    max_coord_nbits = min(
        dtype_effective_bits(xs.dtype),
        dtype_effective_bits(ys.dtype),
        dtype_effective_bits(zs.dtype),
    )
    if nbits > max_coord_nbits:
        raise ValueError(
            f"nbits={nbits} does not fit in coordinate dtype(s); max is {max_coord_nbits}"
        )

    xs_u = unsigned_view(xs)
    ys_u = unsigned_view(ys)
    zs_u = unsigned_view(zs)

    if out is None:
        out = np.empty(xs.shape, dtype=choose_uint_index_dtype(nbits=nbits, dims=3))
    else:
        if out.shape != xs.shape:
            raise ValueError(
                f"out must have the same shape as xs/ys/zs; got out.shape={out.shape}, xs.shape={xs.shape}"
            )
        max_index_nbits = max_nbits_for_index_dtype(out.dtype, dims=3)
        if nbits > max_index_nbits:
            raise ValueError(
                f"nbits={nbits} does not fit in out dtype {out.dtype}; max is {max_index_nbits}"
            )

    out_u = unsigned_view(out)

    xs_1d = _flatten_nocopy(xs_u, "xs")
    ys_1d = _flatten_nocopy(ys_u, "ys")
    zs_1d = _flatten_nocopy(zs_u, "zs")
    out_1d = _flatten_nocopy(out_u, "out")

    builder = get_encode_3d_batch_builder()
    impl = builder(
        nbits,
        lut_dtype=choose_lut_dtype_for_index_dtype(out.dtype),
        parallel=parallel,
    )
    impl(xs_1d, ys_1d, zs_1d, out_1d)
    return out


def _hilbert_decode_3d_batch(
    indices: IntArray,
    nbits: int,
    *,
    out_xs: IntArray | None = None,
    out_ys: IntArray | None = None,
    out_zs: IntArray | None = None,
    parallel: bool = False,
) -> tuple[IntArray, IntArray, IntArray]:
    max_index_nbits = max_nbits_for_index_dtype(indices.dtype, dims=3)
    if nbits > max_index_nbits:
        raise ValueError(
            f"nbits={nbits} does not fit in indices dtype {indices.dtype}; max is {max_index_nbits}"
        )

    indices_u = unsigned_view(indices)

    provided = (out_xs is not None) + (out_ys is not None) + (out_zs is not None)
    if provided not in (0, 3):
        raise ValueError("out_xs, out_ys, out_zs must be provided together")

    if out_xs is None or out_ys is None or out_zs is None:
        coord_dtype = choose_uint_coord_dtype(nbits=nbits)
        out_xs = np.empty(indices.shape, dtype=coord_dtype)
        out_ys = np.empty(indices.shape, dtype=coord_dtype)
        out_zs = np.empty(indices.shape, dtype=coord_dtype)
    else:
        if (
            out_xs.shape != indices.shape
            or out_ys.shape != indices.shape
            or out_zs.shape != indices.shape
        ):
            raise ValueError(
                "out_xs, out_ys, out_zs must have the same shape as indices; "
                f"got indices.shape={indices.shape}, out_xs.shape={out_xs.shape}, out_ys.shape={out_ys.shape}, out_zs.shape={out_zs.shape}"
            )
        max_coord_nbits = min(
            dtype_effective_bits(out_xs.dtype),
            dtype_effective_bits(out_ys.dtype),
            dtype_effective_bits(out_zs.dtype),
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"nbits={nbits} does not fit in coordinate dtype(s); max is {max_coord_nbits}"
            )

    out_xs_u = unsigned_view(out_xs)
    out_ys_u = unsigned_view(out_ys)
    out_zs_u = unsigned_view(out_zs)

    indices_1d = _flatten_nocopy(indices_u, "indices")
    out_xs_1d = _flatten_nocopy(out_xs_u, "out_xs")
    out_ys_1d = _flatten_nocopy(out_ys_u, "out_ys")
    out_zs_1d = _flatten_nocopy(out_zs_u, "out_zs")

    builder = get_decode_3d_batch_builder()
    impl = builder(
        nbits,
        lut_dtype=choose_lut_dtype_for_index_dtype(indices.dtype),
        parallel=parallel,
    )
    impl(indices_1d, out_xs_1d, out_ys_1d, out_zs_1d)
    return out_xs, out_ys, out_zs


def get_hilbert_encode_3d_kernel(
    nbits: int, *, lut_dtype: LutUIntDTypeLike = np.uint16
):
    """Return a Numba-compiled *scalar* 3D Hilbert encoder.

    This is the low-level kernel used by :func:`hilbert_encode_3d` in scalar mode
    and is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    lut_dtype
        Element dtype used for the internal lookup tables.

        Default is ``uint16`` (smallest; ~3 KiB LUT footprint). When fusing into
        a cache-intensive kernel, keeping the LUT small can help.

        For best throughput, it is usually better to match ``lut_dtype``
        to your index dtype (``uint16``/``uint32``/``uint64``), which increases LUT
        size (~6 KiB for ``uint32``, ~12 KiB for ``uint64``) but reduces widening.
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
):
    """Return a Numba-compiled *scalar* 3D Hilbert decoder.

    This is the low-level kernel used by :func:`hilbert_decode_3d` in scalar mode
    and is intended for fusing into your own ``@numba.njit`` loops.

    Parameters
    ----------
    nbits
        Number of coordinate bits (grid domain is ``[0, 2**nbits)`` per axis).
    lut_dtype
        Element dtype used for the internal lookup tables.

        See :func:`get_hilbert_encode_3d_kernel` for the performance tradeoff.

    Returns
    -------
    callable
        A Numba-compiled function with signature ``(index: int) -> (x: int, y: int, z: int)``.
    """
    builder = get_decode_3d_scalar_builder()
    return builder(nbits, lut_dtype=lut_dtype)
