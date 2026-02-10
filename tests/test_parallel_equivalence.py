from __future__ import annotations

import numpy as np

from hilbertsfc.hilbert2d import hilbert_decode_2d, hilbert_encode_2d
from hilbertsfc.hilbert3d import hilbert_decode_3d, hilbert_encode_3d


def test_parallel_equivalence_2d(rng: np.random.Generator) -> None:
    nbits = 8
    n = 4096
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=n, dtype=np.int16)
    ys = rng.integers(0, hi, size=n, dtype=np.int16)

    idx_seq = hilbert_encode_2d(xs, ys, nbits, parallel=False)
    idx_par = hilbert_encode_2d(xs, ys, nbits, parallel=True)
    np.testing.assert_array_equal(idx_par, idx_seq)

    x_seq, y_seq = hilbert_decode_2d(idx_seq, nbits, parallel=False)
    x_par, y_par = hilbert_decode_2d(idx_seq, nbits, parallel=True)
    np.testing.assert_array_equal(x_par, x_seq)
    np.testing.assert_array_equal(y_par, y_seq)


def test_parallel_equivalence_3d(rng: np.random.Generator) -> None:
    nbits = 6
    n = 2048
    hi = 1 << nbits

    xs = rng.integers(0, hi, size=n, dtype=np.int16)
    ys = rng.integers(0, hi, size=n, dtype=np.int16)
    zs = rng.integers(0, hi, size=n, dtype=np.int16)

    idx_seq = hilbert_encode_3d(xs, ys, zs, nbits, parallel=False)
    idx_par = hilbert_encode_3d(xs, ys, zs, nbits, parallel=True)
    np.testing.assert_array_equal(idx_par, idx_seq)

    x_seq, y_seq, z_seq = hilbert_decode_3d(idx_seq, nbits, parallel=False)
    x_par, y_par, z_par = hilbert_decode_3d(idx_seq, nbits, parallel=True)
    np.testing.assert_array_equal(x_par, x_seq)
    np.testing.assert_array_equal(y_par, y_seq)
    np.testing.assert_array_equal(z_par, z_seq)
