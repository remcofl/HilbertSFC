import numpy as np

from hilbertsfc import clear_all_caches, clear_kernel_caches
from hilbertsfc._dispatch import (
    get_decode_2d_batch_builder,
    get_decode_2d_scalar_builder,
    get_decode_3d_batch_builder,
    get_decode_3d_scalar_builder,
    get_encode_2d_batch_builder,
    get_encode_2d_scalar_builder,
    get_encode_3d_batch_builder,
    get_encode_3d_scalar_builder,
    get_morton_decode_2d_batch_builder,
    get_morton_decode_2d_scalar_builder,
    get_morton_decode_3d_batch_builder,
    get_morton_decode_3d_scalar_builder,
    get_morton_encode_2d_batch_builder,
    get_morton_encode_2d_scalar_builder,
    get_morton_encode_3d_batch_builder,
    get_morton_encode_3d_scalar_builder,
)
from hilbertsfc._kernels.numba.hilbert2d_decode import (
    build_hilbert_decode_2d_batch_impl,
    build_hilbert_decode_2d_impl,
)
from hilbertsfc._kernels.numba.hilbert2d_encode import (
    build_hilbert_encode_2d_batch_impl,
    build_hilbert_encode_2d_impl,
)
from hilbertsfc._kernels.numba.hilbert3d_decode import (
    build_hilbert_decode_3d_batch_impl,
    build_hilbert_decode_3d_impl,
)
from hilbertsfc._kernels.numba.hilbert3d_encode import (
    build_hilbert_encode_3d_batch_impl,
    build_hilbert_encode_3d_impl,
)
from hilbertsfc._kernels.numba.morton2d_decode import (
    build_morton_decode_2d_batch_impl,
    build_morton_decode_2d_impl,
)
from hilbertsfc._kernels.numba.morton2d_encode import (
    build_morton_encode_2d_batch_impl,
    build_morton_encode_2d_impl,
)
from hilbertsfc._kernels.numba.morton3d_decode import (
    build_morton_decode_3d_batch_impl,
    build_morton_decode_3d_impl,
)
from hilbertsfc._kernels.numba.morton3d_encode import (
    build_morton_encode_3d_batch_impl,
    build_morton_encode_3d_impl,
)
from hilbertsfc._luts import lut_2d4b_b_qs_u64


def test_dispatch_points_to_expected_builders() -> None:
    assert get_encode_2d_scalar_builder() is build_hilbert_encode_2d_impl
    assert get_decode_2d_scalar_builder() is build_hilbert_decode_2d_impl
    assert get_encode_2d_batch_builder() is build_hilbert_encode_2d_batch_impl
    assert get_decode_2d_batch_builder() is build_hilbert_decode_2d_batch_impl

    assert get_encode_3d_scalar_builder() is build_hilbert_encode_3d_impl
    assert get_decode_3d_scalar_builder() is build_hilbert_decode_3d_impl
    assert get_encode_3d_batch_builder() is build_hilbert_encode_3d_batch_impl
    assert get_decode_3d_batch_builder() is build_hilbert_decode_3d_batch_impl

    assert get_morton_encode_2d_scalar_builder() is build_morton_encode_2d_impl
    assert get_morton_decode_2d_scalar_builder() is build_morton_decode_2d_impl
    assert get_morton_encode_2d_batch_builder() is build_morton_encode_2d_batch_impl
    assert get_morton_decode_2d_batch_builder() is build_morton_decode_2d_batch_impl

    assert get_morton_encode_3d_scalar_builder() is build_morton_encode_3d_impl
    assert get_morton_decode_3d_scalar_builder() is build_morton_decode_3d_impl
    assert get_morton_encode_3d_batch_builder() is build_morton_encode_3d_batch_impl
    assert get_morton_decode_3d_batch_builder() is build_morton_decode_3d_batch_impl


def test_kernel_builder_cache_clear_refreshes_function_object() -> None:
    f1 = build_hilbert_encode_2d_impl(3)
    f2 = build_hilbert_encode_2d_impl(3)
    assert f1 is f2

    clear_kernel_caches()

    f3 = build_hilbert_encode_2d_impl(3)
    assert f3 is not f1


def test_3d_builder_accepts_tile_nbits_and_lut_dtype() -> None:
    enc_2b_u16 = build_hilbert_encode_3d_impl(2, tile_nbits=2, lut_dtype=np.uint16)
    enc_2b_u32 = build_hilbert_encode_3d_impl(2, tile_nbits=2, lut_dtype=np.uint32)
    enc_3b_u16 = build_hilbert_encode_3d_impl(2, tile_nbits=3, lut_dtype=np.uint16)

    # Each configuration has its own cached specialized kernel.
    assert enc_2b_u16 is not enc_2b_u32
    assert enc_2b_u16 is not enc_3b_u16


def test_3d_builder_rejects_invalid_tile_nbits() -> None:
    with np.testing.assert_raises_regex(ValueError, "tile_nbits must be 2 or 3"):
        build_hilbert_encode_3d_impl(3, tile_nbits=4)  # type: ignore[arg-type]
    with np.testing.assert_raises_regex(ValueError, "tile_nbits must be 2 or 3"):
        build_hilbert_decode_3d_impl(3, tile_nbits=4)  # type: ignore[arg-type]


def test_clear_all_caches_clears_luts_and_kernels() -> None:
    lut_a = lut_2d4b_b_qs_u64()
    kern_a = build_hilbert_encode_2d_impl(3)

    clear_all_caches()

    lut_b = lut_2d4b_b_qs_u64()
    kern_b = build_hilbert_encode_2d_impl(3)

    assert lut_a is not lut_b
    assert kern_a is not kern_b
