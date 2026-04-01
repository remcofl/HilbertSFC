import warnings

import numpy as np

from hilbertsfc.hilbert2d import hilbert_decode_2d, hilbert_encode_2d
from hilbertsfc.hilbert3d import hilbert_decode_3d, hilbert_encode_3d


def test_batch_accepts_2d_shapes_and_roundtrips_2d(rng: np.random.Generator) -> None:
    nbits = 6
    shape = (32, 17)
    hi = 1 << nbits

    x = rng.integers(0, hi, size=shape, dtype=np.int16)
    y = rng.integers(0, hi, size=shape, dtype=np.int16)

    idx = hilbert_encode_2d(x, y, nbits=nbits)
    assert idx.shape == shape

    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, x.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y.astype(y2.dtype, copy=False))


def test_batch_accepts_2d_shapes_and_roundtrips_3d(rng: np.random.Generator) -> None:
    nbits = 5
    shape = (16, 9)
    hi = 1 << nbits

    x = rng.integers(0, hi, size=shape, dtype=np.int16)
    y = rng.integers(0, hi, size=shape, dtype=np.int16)
    z = rng.integers(0, hi, size=shape, dtype=np.int16)

    idx = hilbert_encode_3d(x, y, z, nbits=nbits)
    assert idx.shape == shape

    x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, x.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y.astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, z.astype(z2.dtype, copy=False))


def test_batch_handles_empty_array() -> None:
    nbits = 4
    x = np.empty((0,), dtype=np.uint8)
    y = np.empty((0,), dtype=np.uint8)

    idx = hilbert_encode_2d(x, y, nbits=nbits)
    assert idx.shape == (0,)

    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    assert x2.shape == (0,)
    assert y2.shape == (0,)


def test_strided_views_work_2d(rng: np.random.Generator) -> None:
    nbits = 4
    hi = 1 << nbits
    x = rng.integers(0, hi, size=100, dtype=np.uint8)[::2]
    y = rng.integers(0, hi, size=100, dtype=np.uint8)[::2]

    assert x.flags.c_contiguous is False

    idx = hilbert_encode_2d(x, y, nbits=nbits)
    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, x.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y.astype(y2.dtype, copy=False))


def test_strided_views_work_on_decode_3d(rng: np.random.Generator) -> None:
    nbits = 3
    hi = 1 << nbits
    x = rng.integers(0, hi, size=100, dtype=np.uint8)
    y = rng.integers(0, hi, size=100, dtype=np.uint8)
    z = rng.integers(0, hi, size=100, dtype=np.uint8)
    idx = hilbert_encode_3d(x, y, z, nbits=nbits)

    idx_view = idx[::2]
    assert idx_view.flags.c_contiguous is False

    x2, y2, z2 = hilbert_decode_3d(idx_view, nbits=nbits)
    np.testing.assert_array_equal(x2, x[::2].astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y[::2].astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, z[::2].astype(z2.dtype, copy=False))


def test_transposed_f_contiguous_inputs_work_2d_no_copy_warning(
    rng: np.random.Generator,
) -> None:
    nbits = 6
    shape = (12, 7)
    hi = 1 << nbits

    x = rng.integers(0, hi, size=shape, dtype=np.uint16)
    y = rng.integers(0, hi, size=shape, dtype=np.uint16)

    x_t = x.T
    y_t = y.T
    assert x_t.flags.f_contiguous
    assert y_t.flags.f_contiguous

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        idx = hilbert_encode_2d(x_t, y_t, nbits=nbits)
        x2, y2 = hilbert_decode_2d(idx, nbits=nbits)

    copy_warnings = [
        w
        for w in rec
        if "could not be flattened to 1D without copying" in str(w.message)
    ]
    assert copy_warnings == []

    np.testing.assert_array_equal(x2, x_t.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y_t.astype(y2.dtype, copy=False))


def test_transposed_f_contiguous_inputs_work_3d_no_copy_warning(
    rng: np.random.Generator,
) -> None:
    nbits = 5
    shape = (9, 6)
    hi = 1 << nbits

    x = rng.integers(0, hi, size=shape, dtype=np.uint16)
    y = rng.integers(0, hi, size=shape, dtype=np.uint16)
    z = rng.integers(0, hi, size=shape, dtype=np.uint16)

    x_t = x.T
    y_t = y.T
    z_t = z.T
    assert x_t.flags.f_contiguous
    assert y_t.flags.f_contiguous
    assert z_t.flags.f_contiguous

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        idx = hilbert_encode_3d(x_t, y_t, z_t, nbits=nbits)
        x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits)

    copy_warnings = [
        w
        for w in rec
        if "could not be flattened to 1D without copying" in str(w.message)
    ]
    assert copy_warnings == []

    np.testing.assert_array_equal(x2, x_t.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y_t.astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, z_t.astype(z2.dtype, copy=False))


def test_strided_transposed_inputs_and_strided_out_work_2d(
    rng: np.random.Generator,
) -> None:
    nbits = 6
    hi = 1 << nbits

    x = rng.integers(0, hi, size=(13, 10), dtype=np.uint16)
    y = rng.integers(0, hi, size=(13, 10), dtype=np.uint16)

    x_view = x.T[:, ::2]
    y_view = y.T[:, ::2]
    assert x_view.flags.c_contiguous is False
    assert x_view.flags.f_contiguous is False

    idx = hilbert_encode_2d(x_view, y_view, nbits=nbits)
    x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, x_view.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y_view.astype(y2.dtype, copy=False))

    out_base = np.empty((x_view.shape[0], x_view.shape[1] * 2), dtype=np.uint32)
    out_view = out_base[:, ::2]
    assert out_view.shape == x_view.shape
    assert out_view.flags.c_contiguous is False

    idx_out = hilbert_encode_2d(x_view, y_view, nbits=nbits, out=out_view)
    assert idx_out is out_view
    np.testing.assert_array_equal(idx_out, idx.astype(out_view.dtype, copy=False))

    out_x_base = np.empty((x_view.shape[0], x_view.shape[1] * 2), dtype=np.uint16)
    out_y_base = np.empty((x_view.shape[0], x_view.shape[1] * 2), dtype=np.uint16)
    out_x = out_x_base[:, ::2]
    out_y = out_y_base[:, ::2]
    rx, ry = hilbert_decode_2d(idx, nbits=nbits, out_x=out_x, out_y=out_y)
    assert rx is out_x
    assert ry is out_y
    np.testing.assert_array_equal(rx, x_view.astype(rx.dtype, copy=False))
    np.testing.assert_array_equal(ry, y_view.astype(ry.dtype, copy=False))


def test_strided_transposed_inputs_and_strided_out_work_3d(
    rng: np.random.Generator,
) -> None:
    nbits = 5
    hi = 1 << nbits

    x = rng.integers(0, hi, size=(11, 9), dtype=np.uint16)
    y = rng.integers(0, hi, size=(11, 9), dtype=np.uint16)
    z = rng.integers(0, hi, size=(11, 9), dtype=np.uint16)

    x_view = x.T[:, ::2]
    y_view = y.T[:, ::2]
    z_view = z.T[:, ::2]
    assert x_view.flags.c_contiguous is False
    assert x_view.flags.f_contiguous is False

    idx = hilbert_encode_3d(x_view, y_view, z_view, nbits=nbits)
    x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits)
    np.testing.assert_array_equal(x2, x_view.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y_view.astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, z_view.astype(z2.dtype, copy=False))

    out_base = np.empty((x_view.shape[0], x_view.shape[1] * 2), dtype=np.uint32)
    out_view = out_base[:, ::2]
    assert out_view.shape == x_view.shape
    assert out_view.flags.c_contiguous is False

    idx_out = hilbert_encode_3d(x_view, y_view, z_view, nbits=nbits, out=out_view)
    assert idx_out is out_view
    np.testing.assert_array_equal(idx_out, idx.astype(out_view.dtype, copy=False))

    out_x_base = np.empty((x_view.shape[0], x_view.shape[1] * 2), dtype=np.uint16)
    out_y_base = np.empty((x_view.shape[0], x_view.shape[1] * 2), dtype=np.uint16)
    out_z_base = np.empty((x_view.shape[0], x_view.shape[1] * 2), dtype=np.uint16)
    out_x = out_x_base[:, ::2]
    out_y = out_y_base[:, ::2]
    out_z = out_z_base[:, ::2]

    rx, ry, rz = hilbert_decode_3d(
        idx,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
    )
    assert rx is out_x
    assert ry is out_y
    assert rz is out_z
    np.testing.assert_array_equal(rx, x_view.astype(rx.dtype, copy=False))
    np.testing.assert_array_equal(ry, y_view.astype(ry.dtype, copy=False))
    np.testing.assert_array_equal(rz, z_view.astype(rz.dtype, copy=False))
