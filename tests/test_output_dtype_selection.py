import numpy as np

from hilbertsfc.hilbert2d import hilbert_encode_2d
from hilbertsfc.hilbert3d import hilbert_encode_3d


def test_output_dtype_selection_2d() -> None:
    xs = np.zeros(4, dtype=np.uint32)
    ys = np.zeros(4, dtype=np.uint32)

    assert hilbert_encode_2d(xs, ys, nbits=3).dtype == np.dtype(np.uint8)
    assert hilbert_encode_2d(xs, ys, nbits=8).dtype == np.dtype(np.uint16)
    assert hilbert_encode_2d(xs, ys, nbits=9).dtype == np.dtype(np.uint32)
    assert hilbert_encode_2d(xs, ys, nbits=17).dtype == np.dtype(np.uint64)


def test_output_dtype_selection_3d() -> None:
    xs = np.zeros(4, dtype=np.uint32)
    ys = np.zeros(4, dtype=np.uint32)
    zs = np.zeros(4, dtype=np.uint32)

    assert hilbert_encode_3d(xs, ys, zs, nbits=5).dtype == np.dtype(np.uint16)
    assert hilbert_encode_3d(xs, ys, zs, nbits=6).dtype == np.dtype(np.uint32)
    assert hilbert_encode_3d(xs, ys, zs, nbits=11).dtype == np.dtype(np.uint64)
