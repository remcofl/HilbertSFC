import numpy as np
import pytest

from hilbertsfc.hilbert3d import (
    get_hilbert_decode_3d_kernel,
    get_hilbert_encode_3d_kernel,
    hilbert_decode_3d,
    hilbert_encode_3d,
)


def test_3d_scalar_roundtrip_bruteforce(small_nbits_3d: tuple[int, ...]) -> None:
    for nbits in small_nbits_3d:
        size = 1 << nbits
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    idx = hilbert_encode_3d(x, y, z, nbits=nbits)
                    assert isinstance(idx, int)
                    assert 0 <= idx < (1 << (3 * nbits))

                    x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits)
                    assert (x2, y2, z2) == (x, y, z)


def test_3d_scalar_invalid_nbits_raises() -> None:
    with pytest.raises(ValueError):
        hilbert_encode_3d(0, 0, 0, nbits=0)
    with pytest.raises(ValueError):
        hilbert_decode_3d(0, nbits=22)


def test_3d_scalar_negative_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hilbert_encode_3d(-1, 2, 3)
    with pytest.raises(ValueError, match="non-negative"):
        hilbert_encode_3d(1, -2, 3)
    with pytest.raises(ValueError, match="non-negative"):
        hilbert_encode_3d(1, 2, -3)
    with pytest.raises(ValueError, match="non-negative"):
        hilbert_decode_3d(-1)


def test_3d_scalar_out_rejected_and_parallel_warns() -> None:
    out = np.empty(1, dtype=np.uint8)
    with pytest.raises(TypeError, match="out is only valid"):
        hilbert_encode_3d(0, 0, 0, nbits=2, out=out)  # type: ignore[arg-type]

    with pytest.warns(UserWarning, match="parallel=True has no effect"):
        hilbert_encode_3d(0, 0, 0, nbits=2, parallel=True)  # type: ignore[reportArgumentType]

    with pytest.warns(UserWarning, match="parallel=True has no effect"):
        hilbert_decode_3d(0, nbits=2, parallel=True)  # type: ignore[reportArgumentType]


def test_3d_batch_roundtrip_with_lut_dtype_and_parallel(
    rng: np.random.Generator,
) -> None:
    nbits = 6
    n = 1024
    hi = 1 << nbits

    x = rng.integers(0, hi, size=n, dtype=np.int16)
    y = rng.integers(0, hi, size=n, dtype=np.int16)
    z = rng.integers(0, hi, size=n, dtype=np.int16)

    idx = hilbert_encode_3d(x, y, z, nbits=nbits, parallel=True)
    assert idx.shape == (n,)
    # 3*6=18 bits -> uint32
    assert idx.dtype == np.dtype(np.uint32)

    x2, y2, z2 = hilbert_decode_3d(idx, nbits=nbits, parallel=True)
    np.testing.assert_array_equal(x2, x.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, y.astype(y2.dtype, copy=False))
    np.testing.assert_array_equal(z2, z.astype(z2.dtype, copy=False))


def test_3d_decode_batch_out_triple_rule(rng: np.random.Generator) -> None:
    nbits = 3
    n = 10
    hi = 1 << nbits
    x = rng.integers(0, hi, size=n, dtype=np.uint8)
    y = rng.integers(0, hi, size=n, dtype=np.uint8)
    z = rng.integers(0, hi, size=n, dtype=np.uint8)
    idx = hilbert_encode_3d(x, y, z, nbits=nbits)

    out_x = np.empty(n, dtype=np.uint8)
    out_y = np.empty(n, dtype=np.uint8)

    with pytest.raises(ValueError, match="provided together"):
        hilbert_decode_3d(idx, nbits=nbits, out_x=out_x, out_y=out_y)  # type: ignore


def test_3d_batch_coord_dtype_validation() -> None:
    x = np.zeros(4, dtype=np.uint8)
    y = np.zeros(4, dtype=np.uint8)
    z = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="coordinate dtypes"):
        hilbert_encode_3d(x, y, z, nbits=9)


def test_3d_decode_batch_index_dtype_validation() -> None:
    # uint16 indices only support up to floor(16/3)=5 bits/coord in 3D.
    idx = np.zeros(4, dtype=np.uint16)
    with pytest.raises(ValueError, match="index dtype"):
        hilbert_decode_3d(idx, nbits=6)


def test_3d_decode_batch_out_coord_dtype_validation() -> None:
    idx = np.zeros(4, dtype=np.uint64)
    out_x = np.zeros(4, dtype=np.uint8)
    out_y = np.zeros(4, dtype=np.uint8)
    out_z = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="out_x/out_y dtypes"):
        hilbert_decode_3d(idx, nbits=9, out_x=out_x, out_y=out_y, out_z=out_z)


def test_3d_batch_infers_nbits_warns_when_capped() -> None:
    x = np.zeros(4, dtype=np.uint64)
    y = np.zeros(4, dtype=np.uint64)
    z = np.zeros(4, dtype=np.uint64)
    with pytest.warns(UserWarning, match="exceeds the algorithm maximum"):
        idx = hilbert_encode_3d(x, y, z)
    assert idx.dtype == np.dtype(np.uint64)


def test_3d_kernel_getters_support_lut_dtype() -> None:
    enc = get_hilbert_encode_3d_kernel(4, lut_dtype=np.uint32)
    dec = get_hilbert_decode_3d_kernel(4, lut_dtype=np.uint32)

    idx = int(enc(2, 7, 3))
    x2, y2, z2 = dec(idx)
    assert (int(x2), int(y2), int(z2)) == (2, 7, 3)
