import warnings

import numpy as np

from hilbertsfc.hilbert2d import hilbert_decode_2d, hilbert_encode_2d
from hilbertsfc.hilbert3d import hilbert_decode_3d, hilbert_encode_3d


def test_batch_accepts_2d_shapes_and_roundtrips_2d(rng: np.random.Generator) -> None:
    nbits = 6
    shape = (32, 17)
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=shape, dtype=np.int16)
    ys = rng.integers(0, hi, size=shape, dtype=np.int16)

    idx = hilbert_encode_2d(xs, ys, nbits=nbits)
    assert idx.shape == shape

    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, xs.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys.astype(y2.dtype, copy=False))


def test_batch_accepts_2d_shapes_and_roundtrips_3d(rng: np.random.Generator) -> None:
    nbits = 5
    shape = (16, 9)
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=shape, dtype=np.int16)
    ys = rng.integers(0, hi, size=shape, dtype=np.int16)
    zs = rng.integers(0, hi, size=shape, dtype=np.int16)

    idx = hilbert_encode_3d(xs, ys, zs, nbits=nbits)
    assert idx.shape == shape

    x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, xs.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys.astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, zs.astype(z2.dtype, copy=False))


def test_batch_handles_empty_arrays() -> None:
    nbits = 4
    xs = np.empty((0,), dtype=np.uint8)
    ys = np.empty((0,), dtype=np.uint8)

    idx = hilbert_encode_2d(xs, ys, nbits=nbits)
    assert idx.shape == (0,)

    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    assert x2.shape == (0,)
    assert y2.shape == (0,)


def test_strided_views_work_2d(rng: np.random.Generator) -> None:
    nbits = 4
    hi = 1 << nbits
    xs = rng.integers(0, hi, size=100, dtype=np.uint8)[::2]
    ys = rng.integers(0, hi, size=100, dtype=np.uint8)[::2]

    assert xs.flags.c_contiguous is False

    idx = hilbert_encode_2d(xs, ys, nbits=nbits)
    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, xs.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys.astype(y2.dtype, copy=False))


def test_strided_views_work_on_decode_3d(rng: np.random.Generator) -> None:
    nbits = 3
    hi = 1 << nbits
    xs = rng.integers(0, hi, size=100, dtype=np.uint8)
    ys = rng.integers(0, hi, size=100, dtype=np.uint8)
    zs = rng.integers(0, hi, size=100, dtype=np.uint8)
    idx = hilbert_encode_3d(xs, ys, zs, nbits=nbits)

    idx_view = idx[::2]
    assert idx_view.flags.c_contiguous is False

    x2, y2, z2 = hilbert_decode_3d(idx_view, nbits=nbits)
    np.testing.assert_array_equal(x2, xs[::2].astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys[::2].astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, zs[::2].astype(z2.dtype, copy=False))


def test_transposed_f_contiguous_inputs_work_2d_no_copy_warning(
    rng: np.random.Generator,
) -> None:
    nbits = 6
    shape = (12, 7)
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=shape, dtype=np.uint16)
    ys = rng.integers(0, hi, size=shape, dtype=np.uint16)

    xs_t = xs.T
    ys_t = ys.T
    assert xs_t.flags.f_contiguous
    assert ys_t.flags.f_contiguous

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        idx = hilbert_encode_2d(xs_t, ys_t, nbits=nbits)
        x2, y2 = hilbert_decode_2d(idx, nbits=nbits)

    copy_warnings = [
        w
        for w in rec
        if "could not be flattened to 1D without copying" in str(w.message)
    ]
    assert copy_warnings == []

    np.testing.assert_array_equal(x2, xs_t.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys_t.astype(y2.dtype, copy=False))


def test_transposed_f_contiguous_inputs_work_3d_no_copy_warning(
    rng: np.random.Generator,
) -> None:
    nbits = 5
    shape = (9, 6)
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=shape, dtype=np.uint16)
    ys = rng.integers(0, hi, size=shape, dtype=np.uint16)
    zs = rng.integers(0, hi, size=shape, dtype=np.uint16)

    xs_t = xs.T
    ys_t = ys.T
    zs_t = zs.T
    assert xs_t.flags.f_contiguous
    assert ys_t.flags.f_contiguous
    assert zs_t.flags.f_contiguous

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        idx = hilbert_encode_3d(xs_t, ys_t, zs_t, nbits=nbits)
        x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits)

    copy_warnings = [
        w
        for w in rec
        if "could not be flattened to 1D without copying" in str(w.message)
    ]
    assert copy_warnings == []

    np.testing.assert_array_equal(x2, xs_t.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys_t.astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, zs_t.astype(z2.dtype, copy=False))


def test_strided_transposed_inputs_and_strided_out_work_2d(
    rng: np.random.Generator,
) -> None:
    nbits = 6
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=(13, 10), dtype=np.uint16)
    ys = rng.integers(0, hi, size=(13, 10), dtype=np.uint16)

    xs_view = xs.T[:, ::2]
    ys_view = ys.T[:, ::2]
    assert xs_view.flags.c_contiguous is False
    assert xs_view.flags.f_contiguous is False

    idx = hilbert_encode_2d(xs_view, ys_view, nbits=nbits)
    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, xs_view.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys_view.astype(y2.dtype, copy=False))

    out_base = np.empty((xs_view.shape[0], xs_view.shape[1] * 2), dtype=np.uint32)
    out_view = out_base[:, ::2]
    assert out_view.shape == xs_view.shape
    assert out_view.flags.c_contiguous is False

    idx_out = hilbert_encode_2d(xs_view, ys_view, nbits=nbits, out=out_view)
    assert idx_out is out_view
    np.testing.assert_array_equal(idx_out, idx.astype(out_view.dtype, copy=False))

    out_xs_base = np.empty((xs_view.shape[0], xs_view.shape[1] * 2), dtype=np.uint16)
    out_ys_base = np.empty((xs_view.shape[0], xs_view.shape[1] * 2), dtype=np.uint16)
    out_xs = out_xs_base[:, ::2]
    out_ys = out_ys_base[:, ::2]
    rx, ry = hilbert_decode_2d(idx, nbits=nbits, out_xs=out_xs, out_ys=out_ys)
    assert rx is out_xs
    assert ry is out_ys
    np.testing.assert_array_equal(rx, xs_view.astype(rx.dtype, copy=False))
    np.testing.assert_array_equal(ry, ys_view.astype(ry.dtype, copy=False))


def test_strided_transposed_inputs_and_strided_out_work_3d(
    rng: np.random.Generator,
) -> None:
    nbits = 5
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=(11, 9), dtype=np.uint16)
    ys = rng.integers(0, hi, size=(11, 9), dtype=np.uint16)
    zs = rng.integers(0, hi, size=(11, 9), dtype=np.uint16)

    xs_view = xs.T[:, ::2]
    ys_view = ys.T[:, ::2]
    zs_view = zs.T[:, ::2]
    assert xs_view.flags.c_contiguous is False
    assert xs_view.flags.f_contiguous is False

    idx = hilbert_encode_3d(xs_view, ys_view, zs_view, nbits=nbits)
    x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, xs_view.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys_view.astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, zs_view.astype(z2.dtype, copy=False))

    out_base = np.empty((xs_view.shape[0], xs_view.shape[1] * 2), dtype=np.uint32)
    out_view = out_base[:, ::2]
    assert out_view.shape == xs_view.shape
    assert out_view.flags.c_contiguous is False

    idx_out = hilbert_encode_3d(xs_view, ys_view, zs_view, nbits=nbits, out=out_view)
    assert idx_out is out_view
    np.testing.assert_array_equal(idx_out, idx.astype(out_view.dtype, copy=False))

    out_xs_base = np.empty((xs_view.shape[0], xs_view.shape[1] * 2), dtype=np.uint16)
    out_ys_base = np.empty((xs_view.shape[0], xs_view.shape[1] * 2), dtype=np.uint16)
    out_zs_base = np.empty((xs_view.shape[0], xs_view.shape[1] * 2), dtype=np.uint16)
    out_xs = out_xs_base[:, ::2]
    out_ys = out_ys_base[:, ::2]
    out_zs = out_zs_base[:, ::2]

    rx, ry, rz = hilbert_decode_3d(
        idx,
        nbits=nbits,
        out_xs=out_xs,
        out_ys=out_ys,
        out_zs=out_zs,
    )
    assert rx is out_xs
    assert ry is out_ys
    assert rz is out_zs
    np.testing.assert_array_equal(rx, xs_view.astype(rx.dtype, copy=False))
    np.testing.assert_array_equal(ry, ys_view.astype(ry.dtype, copy=False))
    np.testing.assert_array_equal(rz, zs_view.astype(rz.dtype, copy=False))
