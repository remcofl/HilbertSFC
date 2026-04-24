"""Public 2D Morton API.

This module provides the main user-facing API for 2D Morton encoding and decoding:

- ``morton_encode_2d``
- ``morton_decode_2d``

These accept either Python/NumPy scalar integers (scalar mode) or NumPy integer
arrays (array mode).

For advanced use (embedding Morton encode/decode into your own Numba-compiled
code), this module also exposes kernel accessors:

- ``get_morton_encode_2d_kernel``
- ``get_morton_decode_2d_kernel``

"""

from collections.abc import Callable

from ._dispatch import (
    get_morton_decode_2d_batch_builder,
    get_morton_decode_2d_scalar_builder,
    get_morton_encode_2d_batch_builder,
    get_morton_encode_2d_scalar_builder,
)
from ._public_api_shared import (
    Decode2DAdapter,
    Encode2DAdapter,
    decode_2d_api,
    encode_2d_api,
)
from .types import IntArray, IntScalar

_MORTON_ENCODE_2D_ADAPTER = Encode2DAdapter(
    build_scalar=lambda nbits: get_morton_encode_2d_scalar_builder()(nbits),
    build_batch=lambda nbits, parallel: get_morton_encode_2d_batch_builder()(
        nbits,
        parallel=parallel,
    ),
)

_MORTON_DECODE_2D_ADAPTER = Decode2DAdapter(
    build_scalar=lambda nbits: get_morton_decode_2d_scalar_builder()(nbits),
    build_batch=lambda nbits, parallel: get_morton_decode_2d_batch_builder()(
        nbits,
        parallel=parallel,
    ),
)


def morton_encode_2d(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out: IntArray | None = None,
    parallel: bool = False,
) -> int | IntArray:
    """Encode 2D integer coordinates to Morton indices.

    API semantics (parameters, returns, errors) match
    [`hilbert_encode_2d`][hilbertsfc.hilbert2d.hilbert_encode_2d].

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``x`` and ``y`` are scalar integers, returns a Python
    ``int``.
    - **Array mode**: if ``x`` and ``y`` are NumPy integer arrays, returns an array
    of unsigned indices with the same shape and supports ``out=``.
    """

    return encode_2d_api(
        x,
        y,
        nbits=nbits,
        out=out,
        parallel=parallel,
        adapter=_MORTON_ENCODE_2D_ADAPTER,
    )


def morton_decode_2d(
    index: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out_x: IntArray | None = None,
    out_y: IntArray | None = None,
    parallel: bool = False,
) -> tuple[int, int] | tuple[IntArray, IntArray]:
    """Decode Morton indices to 2D integer coordinates.

    API semantics (parameters, returns, errors) match
    [`hilbert_decode_2d`][hilbertsfc.hilbert2d.hilbert_decode_2d].

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``index`` is a scalar integer, returns ``(x, y)`` as
    Python ``int`` values.
    - **Array mode**: if ``index`` is a NumPy integer array, returns coordinate
    arrays with the same shape and supports ``out_x=`` / ``out_y=``.
    """

    return decode_2d_api(
        index,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        parallel=parallel,
        adapter=_MORTON_DECODE_2D_ADAPTER,
    )


def get_morton_encode_2d_kernel(nbits: int) -> Callable[[IntScalar, IntScalar], int]:
    """Return a Numba-compiled *scalar* 2D Morton encoder."""

    builder = get_morton_encode_2d_scalar_builder()
    return builder(nbits)


def get_morton_decode_2d_kernel(nbits: int) -> Callable[[IntScalar], tuple[int, int]]:
    """Return a Numba-compiled *scalar* 2D Morton decoder."""

    builder = get_morton_decode_2d_scalar_builder()
    return builder(nbits)
