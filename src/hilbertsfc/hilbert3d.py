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

from collections.abc import Callable

import numpy as np

from ._dispatch import (
    get_decode_3d_batch_builder,
    get_decode_3d_scalar_builder,
    get_encode_3d_batch_builder,
    get_encode_3d_scalar_builder,
)
from ._dtype import (
    choose_lut_dtype_for_index_dtype,
)
from ._public_api_shared_3d import decode_3d_api, encode_3d_api
from .types import IntArray, IntScalar, LutUIntDTypeLike


def _build_hilbert_encode_3d_batch(nbits, *, parallel=False, index_dtype):
    return get_encode_3d_batch_builder()(
        nbits,
        parallel=parallel,
        lut_dtype=choose_lut_dtype_for_index_dtype(index_dtype),
    )


def _build_hilbert_decode_3d_batch(nbits, *, parallel=False, index_dtype):
    return get_decode_3d_batch_builder()(
        nbits,
        parallel=parallel,
        lut_dtype=choose_lut_dtype_for_index_dtype(index_dtype),
    )


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

    return encode_3d_api(
        x,
        y,
        z,
        nbits=nbits,
        out=out,
        parallel=parallel,
        build_scalar=get_encode_3d_scalar_builder(),
        build_batch=_build_hilbert_encode_3d_batch,
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

    return decode_3d_api(
        index,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        parallel=parallel,
        build_scalar=get_decode_3d_scalar_builder(),
        build_batch=_build_hilbert_decode_3d_batch,
    )


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
