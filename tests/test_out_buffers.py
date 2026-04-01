import numpy as np
import pytest

from hilbertsfc.hilbert2d import hilbert_decode_2d, hilbert_encode_2d
from hilbertsfc.hilbert3d import hilbert_decode_3d, hilbert_encode_3d


def test_encode_2d_out_buffer_identity_and_written(rng: np.random.Generator) -> None:
    nbits = 9
    n = 256
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=n, dtype=np.int16)
    ys = rng.integers(0, hi, size=n, dtype=np.int16)

    out = np.empty(n, dtype=np.uint32)
    ret = hilbert_encode_2d(xs, ys, nbits=nbits, out=out)

    assert ret is out
    assert np.any(out != 0)  # sanity: should have written something


def test_decode_2d_out_buffers_identity(rng: np.random.Generator) -> None:
    nbits = 8
    n = 256
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=n, dtype=np.uint8)
    ys = rng.integers(0, hi, size=n, dtype=np.uint8)
    idx = hilbert_encode_2d(xs, ys, nbits=nbits)

    out_x = np.empty_like(xs)
    out_y = np.empty_like(ys)

    rx, ry = hilbert_decode_2d(idx, nbits=nbits, out_x=out_x, out_y=out_y)
    assert rx is out_x
    assert ry is out_y


def test_encode_3d_out_buffer_identity(rng: np.random.Generator) -> None:
    nbits = 6
    n = 256
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=n, dtype=np.int16)
    ys = rng.integers(0, hi, size=n, dtype=np.int16)
    zs = rng.integers(0, hi, size=n, dtype=np.int16)

    out = np.empty(n, dtype=np.uint32)
    ret = hilbert_encode_3d(xs, ys, zs, nbits=nbits, out=out)
    assert ret is out


def test_decode_3d_out_buffers_identity(rng: np.random.Generator) -> None:
    nbits = 6
    n = 128
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=n, dtype=np.uint16)
    ys = rng.integers(0, hi, size=n, dtype=np.uint16)
    zs = rng.integers(0, hi, size=n, dtype=np.uint16)
    idx = hilbert_encode_3d(xs, ys, zs, nbits=nbits)

    out_x = np.empty_like(xs)
    out_y = np.empty_like(ys)
    out_z = np.empty_like(zs)

    rx, ry, rz = hilbert_decode_3d(
        idx, nbits=nbits, out_x=out_x, out_y=out_y, out_z=out_z
    )
    assert rx is out_x
    assert ry is out_y
    assert rz is out_z


def test_encode_out_shape_mismatch_raises(rng: np.random.Generator) -> None:
    nbits = 4
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=(8, 4), dtype=np.uint8)
    ys = rng.integers(0, hi, size=(8, 4), dtype=np.uint8)
    zs = rng.integers(0, hi, size=(8, 4), dtype=np.uint8)

    out2d = np.empty((32,), dtype=np.uint16)
    with pytest.raises(ValueError, match="same shape"):
        hilbert_encode_2d(xs, ys, nbits=nbits, out=out2d)

    out3d = np.empty((32,), dtype=np.uint16)
    with pytest.raises(ValueError, match="same shape"):
        hilbert_encode_3d(xs, ys, zs, nbits=nbits, out=out3d)


def test_decode_out_shape_mismatch_raises(rng: np.random.Generator) -> None:
    nbits = 4
    hi = 1 << nbits
    xs = rng.integers(0, hi, size=32, dtype=np.uint8)
    ys = rng.integers(0, hi, size=32, dtype=np.uint8)
    zs = rng.integers(0, hi, size=32, dtype=np.uint8)

    idx2d = hilbert_encode_2d(xs, ys, nbits=nbits)
    out_x = np.empty((16,), dtype=np.uint8)
    out_y = np.empty((32,), dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape"):
        hilbert_decode_2d(idx2d, nbits=nbits, out_x=out_x, out_y=out_y)

    idx3d = hilbert_encode_3d(xs, ys, zs, nbits=nbits)
    out_x3 = np.empty((32,), dtype=np.uint8)
    out_y3 = np.empty((16,), dtype=np.uint8)
    out_z3 = np.empty((32,), dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape"):
        hilbert_decode_3d(idx3d, nbits=nbits, out_x=out_x3, out_y=out_y3, out_z=out_z3)
