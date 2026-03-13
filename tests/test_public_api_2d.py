import numpy as np
import pytest

import hilbertsfc
from hilbertsfc.hilbert2d import (
    get_hilbert_decode_2d_kernel,
    get_hilbert_encode_2d_kernel,
    hilbert_decode_2d,
    hilbert_encode_2d,
)


def test_package_exports_smoke() -> None:
    # Keep this very light; mostly ensures top-level re-exports exist.
    assert hasattr(hilbertsfc, "hilbert_encode_2d")
    assert hasattr(hilbertsfc, "hilbert_decode_2d")


def test_2d_scalar_roundtrip_bruteforce(small_nbits_2d: tuple[int, ...]) -> None:
    for nbits in small_nbits_2d:
        size = 1 << nbits
        for x in range(size):
            for y in range(size):
                idx = hilbert_encode_2d(x, y, nbits=nbits)
                assert isinstance(idx, int)
                assert 0 <= idx < (1 << (2 * nbits))

                x2, y2 = hilbert_decode_2d(idx, nbits=nbits)
                assert (x2, y2) == (x, y)


def test_2d_scalar_invalid_nbits_raises() -> None:
    with pytest.raises(ValueError):
        hilbert_encode_2d(0, 0, nbits=0)
    with pytest.raises(ValueError):
        hilbert_decode_2d(0, nbits=33)


def test_2d_scalar_out_rejected_and_parallel_warns() -> None:
    out = np.empty(1, dtype=np.uint8)
    with pytest.raises(TypeError, match="out is only valid"):
        hilbert_encode_2d(0, 0, nbits=4, out=out)  # type: ignore[arg-type]

    with pytest.warns(UserWarning, match="parallel=True has no effect"):
        hilbert_encode_2d(0, 0, nbits=4, parallel=True)  # type: ignore[reportArgumentType]

    with pytest.warns(UserWarning, match="parallel=True has no effect"):
        hilbert_decode_2d(0, nbits=4, parallel=True)  # type: ignore[reportArgumentType]


def test_2d_batch_roundtrip_and_parallel(rng: np.random.Generator) -> None:
    nbits = 8
    n = 2048
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=n, dtype=np.int16)
    ys = rng.integers(0, hi, size=n, dtype=np.int16)

    idx = hilbert_encode_2d(xs, ys, nbits=nbits)
    assert idx.shape == (n,)
    assert idx.dtype == np.dtype(np.uint16)  # 2*8=16 bits

    x2, y2 = hilbert_decode_2d(idx, nbits=nbits, parallel=True)
    np.testing.assert_array_equal(x2, xs.astype(x2.dtype, copy=False))
    np.testing.assert_array_equal(y2, ys.astype(y2.dtype, copy=False))


def test_2d_batch_shape_validation(rng: np.random.Generator) -> None:
    xs = rng.integers(0, 16, size=10, dtype=np.uint8)
    ys = rng.integers(0, 16, size=11, dtype=np.uint8)
    with pytest.raises(ValueError, match="same shape"):
        hilbert_encode_2d(xs, ys, nbits=4)


def test_2d_batch_out_dtype_validation(rng: np.random.Generator) -> None:
    nbits = 9  # requires uint32 for index in 2D
    n = 10
    hi = 1 << nbits
    xs = rng.integers(0, hi, size=n, dtype=np.uint16)
    ys = rng.integers(0, hi, size=n, dtype=np.uint16)

    out = np.empty(n, dtype=np.uint16)  # too small
    with pytest.raises(ValueError, match="out dtype"):
        hilbert_encode_2d(xs, ys, nbits=nbits, out=out)


def test_2d_batch_coord_dtype_validation() -> None:
    # Validation is based on dtype metadata (no scan of values).
    xs = np.zeros(4, dtype=np.uint8)
    ys = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="coordinate dtypes"):
        hilbert_encode_2d(xs, ys, nbits=9)


def test_2d_decode_batch_index_dtype_validation() -> None:
    # uint16 indices only support up to nbits=8 in 2D.
    idx = np.zeros(4, dtype=np.uint16)
    with pytest.raises(ValueError, match="index dtype"):
        hilbert_decode_2d(idx, nbits=9)


def test_2d_batch_infers_nbits_warns_when_capped() -> None:
    xs = np.zeros(4, dtype=np.uint64)
    ys = np.zeros(4, dtype=np.uint64)
    with pytest.warns(UserWarning, match="exceeds the algorithm maximum"):
        idx = hilbert_encode_2d(xs, ys)
    assert idx.dtype == np.dtype(np.uint64)


def test_2d_decode_batch_out_pair_rule(rng: np.random.Generator) -> None:
    nbits = 4
    n = 10
    hi = 1 << nbits
    xs = rng.integers(0, hi, size=n, dtype=np.uint8)
    ys = rng.integers(0, hi, size=n, dtype=np.uint8)
    idx = hilbert_encode_2d(xs, ys, nbits=nbits)

    out_xs = np.empty(n, dtype=np.uint8)
    with pytest.raises(ValueError, match="provided together"):
        hilbert_decode_2d(idx, nbits=nbits, out_xs=out_xs)  # type: ignore


def test_2d_kernel_getters_return_callables() -> None:
    enc = get_hilbert_encode_2d_kernel(4)
    dec = get_hilbert_decode_2d_kernel(4)

    idx = int(enc(3, 5))
    x2, y2 = dec(idx)
    assert (int(x2), int(y2)) == (3, 5)
