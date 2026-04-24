"""Implementation selection (“best kernel”) for hilbertsfc.

This module is where you decide which low-level kernel implementation is
currently preferred for a given nbits/dtype/shape.

For now this is intentionally simple and returns the default stubs.
"""


def get_encode_2d_scalar_builder():
    from ._kernels.numba.hilbert2d_encode import build_hilbert_encode_2d_impl

    return build_hilbert_encode_2d_impl


def get_decode_2d_scalar_builder():
    from ._kernels.numba.hilbert2d_decode import build_hilbert_decode_2d_impl

    return build_hilbert_decode_2d_impl


def get_encode_2d_batch_builder():
    from ._kernels.numba.hilbert2d_encode import build_hilbert_encode_2d_batch_impl

    return build_hilbert_encode_2d_batch_impl


def get_decode_2d_batch_builder():
    from ._kernels.numba.hilbert2d_decode import build_hilbert_decode_2d_batch_impl

    return build_hilbert_decode_2d_batch_impl


def get_encode_3d_scalar_builder():
    from ._kernels.numba.hilbert3d_encode import build_hilbert_encode_3d_impl

    return build_hilbert_encode_3d_impl


def get_decode_3d_scalar_builder():
    from ._kernels.numba.hilbert3d_decode import build_hilbert_decode_3d_impl

    return build_hilbert_decode_3d_impl


def get_encode_3d_batch_builder():
    from ._kernels.numba.hilbert3d_encode import build_hilbert_encode_3d_batch_impl

    return build_hilbert_encode_3d_batch_impl


def get_decode_3d_batch_builder():
    from ._kernels.numba.hilbert3d_decode import build_hilbert_decode_3d_batch_impl

    return build_hilbert_decode_3d_batch_impl


def get_morton_encode_2d_scalar_builder():
    from ._kernels.numba.morton2d_encode import build_morton_encode_2d_impl

    return build_morton_encode_2d_impl


def get_morton_decode_2d_scalar_builder():
    from ._kernels.numba.morton2d_decode import build_morton_decode_2d_impl

    return build_morton_decode_2d_impl


def get_morton_encode_2d_batch_builder():
    from ._kernels.numba.morton2d_encode import build_morton_encode_2d_batch_impl

    return build_morton_encode_2d_batch_impl


def get_morton_decode_2d_batch_builder():
    from ._kernels.numba.morton2d_decode import build_morton_decode_2d_batch_impl

    return build_morton_decode_2d_batch_impl


def get_morton_encode_3d_scalar_builder():
    from ._kernels.numba.morton3d_encode import build_morton_encode_3d_impl

    return build_morton_encode_3d_impl


def get_morton_decode_3d_scalar_builder():
    from ._kernels.numba.morton3d_decode import build_morton_decode_3d_impl

    return build_morton_decode_3d_impl


def get_morton_encode_3d_batch_builder():
    from ._kernels.numba.morton3d_encode import build_morton_encode_3d_batch_impl

    return build_morton_encode_3d_batch_impl


def get_morton_decode_3d_batch_builder():
    from ._kernels.numba.morton3d_decode import build_morton_decode_3d_batch_impl

    return build_morton_decode_3d_batch_impl
