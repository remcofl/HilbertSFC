"""Public 3D Morton API.

This module provides the main user-facing API for 3D Morton encoding and decoding:

- ``morton_encode_3d``
- ``morton_decode_3d``

These accept either Python/NumPy scalar integers (scalar mode) or NumPy integer
arrays (array mode).

For advanced use (embedding Morton encode/decode into your own Numba-compiled
code), this module also exposes kernel accessors:

- ``get_morton_encode_3d_kernel``
- ``get_morton_decode_3d_kernel``

"""

from collections.abc import Callable

from ._dispatch import (
    get_morton_decode_3d_batch_builder,
    get_morton_decode_3d_scalar_builder,
    get_morton_encode_3d_batch_builder,
    get_morton_encode_3d_scalar_builder,
)
from ._public_api_shared_3d import decode_3d_api, encode_3d_api
from .types import IntArray, IntScalar


def morton_encode_3d(
    x: IntScalar | IntArray,
    y: IntScalar | IntArray,
    z: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out: IntArray | None = None,
    parallel: bool = False,
) -> int | IntArray:
    """Encode 3D integer coordinates to Morton indices.

    API semantics (parameters, returns, errors) match
    [`hilbert_encode_3d`][hilbertsfc.hilbert3d.hilbert_encode_3d].

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``x``, ``y``, and ``z`` are scalar integers, returns a
    Python ``int``.
    - **Array mode**: if ``x``, ``y``, and ``z`` are NumPy integer arrays, returns
    an array of unsigned indices with the same shape and supports ``out=``.
    """

    build_scalar = get_morton_encode_3d_scalar_builder()
    build_batch = get_morton_encode_3d_batch_builder()

    def build_batch_wrapped(nbits, *, parallel=False, index_dtype):
        return build_batch(
            nbits,
            parallel=parallel,
        )

    return encode_3d_api(
        x,
        y,
        z,
        nbits=nbits,
        out=out,
        parallel=parallel,
        build_scalar=build_scalar,
        build_batch=build_batch_wrapped,
    )


def morton_decode_3d(
    index: IntScalar | IntArray,
    *,
    nbits: int | None = None,
    out_x: IntArray | None = None,
    out_y: IntArray | None = None,
    out_z: IntArray | None = None,
    parallel: bool = False,
) -> tuple[int, int, int] | tuple[IntArray, IntArray, IntArray]:
    """Decode Morton indices to 3D integer coordinates.

    API semantics (parameters, returns, errors) match
    [`hilbert_decode_3d`][hilbertsfc.hilbert3d.hilbert_decode_3d].

    This function supports both scalar and array inputs:

    - **Scalar mode**: if ``index`` is a scalar integer, returns ``(x, y, z)`` as
    Python ``int`` values.
    - **Array mode**: if ``index`` is a NumPy integer array, returns coordinate
    arrays with the same shape and supports ``out_x=``, ``out_y=``, and
    ``out_z=``.
    """

    build_scalar = get_morton_decode_3d_scalar_builder()
    build_batch = get_morton_decode_3d_batch_builder()

    def build_batch_wrapped(nbits, *, parallel=False, index_dtype):
        return build_batch(
            nbits,
            parallel=parallel,
        )

    return decode_3d_api(
        index,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        parallel=parallel,
        build_scalar=build_scalar,
        build_batch=build_batch_wrapped,
    )


def get_morton_encode_3d_kernel(
    nbits: int,
) -> Callable[[IntScalar, IntScalar, IntScalar], IntScalar]:
    """Return a Numba-compiled *scalar* 3D Morton encoder."""

    builder = get_morton_encode_3d_scalar_builder()
    return builder(nbits)


def get_morton_decode_3d_kernel(
    nbits: int,
) -> Callable[[IntScalar], tuple[int, int, int]]:
    """Return a Numba-compiled *scalar* 3D Morton decoder."""

    builder = get_morton_decode_3d_scalar_builder()
    return builder(nbits)
