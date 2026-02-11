from __future__ import annotations

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
