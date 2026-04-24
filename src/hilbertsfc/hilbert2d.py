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

from collections.abc import Callable

from ._dispatch import (
    get_decode_2d_batch_builder,
    get_decode_2d_scalar_builder,
    get_encode_2d_batch_builder,
    get_encode_2d_scalar_builder,
)
from ._public_api_shared_2d import decode_2d_api, encode_2d_api
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

    return encode_2d_api(
        x,
        y,
        nbits=nbits,
        out=out,
        parallel=parallel,
        build_scalar=get_encode_2d_scalar_builder(),
        build_batch=get_encode_2d_batch_builder(),
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

    return decode_2d_api(
        index,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        parallel=parallel,
        build_scalar=get_decode_2d_scalar_builder(),
        build_batch=get_decode_2d_batch_builder(),
    )


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
