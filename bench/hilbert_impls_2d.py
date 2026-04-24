"""Pluggable 2D Hilbert implementations for ``bench/bench_cli.py``.

Usage:
    python bench/bench_cli.py --impl bench/hilbert_impls_2d.py:IMPLS --ndim 2 --nbits 16 --n 200000
"""

import numpy as np
from hilbert_bench import HilbertImplementation


def _encode_hilbertsfc(xs, ys, out, *, nbits):
    from hilbertsfc import hilbert_encode_2d

    return hilbert_encode_2d(xs, ys, nbits=nbits, parallel=True, out=out)


def _decode_hilbertsfc(idx, xs, ys, *, nbits):
    from hilbertsfc import hilbert_decode_2d

    return hilbert_decode_2d(idx, nbits=nbits, parallel=True, out_x=xs, out_y=ys)


def _encode_hilbertsfc_morton(xs, ys, out, *, nbits):
    from hilbertsfc import morton_encode_2d

    return morton_encode_2d(xs, ys, nbits=nbits, parallel=True, out=out)


def _decode_hilbertsfc_morton(idx, xs, ys, *, nbits):
    from hilbertsfc import morton_decode_2d

    return morton_decode_2d(idx, nbits=nbits, parallel=True, out_x=xs, out_y=ys)


def _encode_numpy_hilbert_curve(xs, ys, out, *, nbits):
    from hilbert import encode

    coords = np.stack([xs, ys], axis=-1)
    out[:] = encode(coords, num_dims=2, num_bits=nbits)
    return out


def _decode_numpy_hilbert_curve(idx, xs, ys, *, nbits):
    from hilbert import decode

    res = decode(idx, num_dims=2, num_bits=nbits)
    xs[:] = res[:, 0]
    ys[:] = res[:, 1]


def _to_big_endian_bytes(arr: np.ndarray) -> np.ndarray:
    dt = arr.dtype
    be = arr.astype(arr.dtype.newbyteorder(">"), copy=False)
    return be.view(np.uint8).reshape(arr.shape + (dt.itemsize,))


def _bytes_to_uint(arr_bytes: np.ndarray, byteorder: str = ">") -> np.ndarray:
    width = int(arr_bytes.shape[-1])
    dt = np.dtype(f"{byteorder}u{width}")
    native = np.dtype(f"u{width}")
    return arr_bytes.view(dt).astype(native)[..., -1]


def _encode_hilbert_bytes(xs, ys, out, *, nbits):
    from hilbert_bytes import encode

    del nbits
    coords = np.stack([xs, ys], axis=-1)
    coords_bytes = _to_big_endian_bytes(coords)
    idx_bytes = encode(coords_bytes)
    out[:] = _bytes_to_uint(idx_bytes)
    return out


def _decode_hilbert_bytes(idx, xs, ys, *, nbits):
    from hilbert_bytes import decode

    del nbits
    idx_bytes = _to_big_endian_bytes(idx)
    res = decode(idx_bytes, ndim=2)
    x = _bytes_to_uint(np.ascontiguousarray(res[:, 0]))
    y = _bytes_to_uint(np.ascontiguousarray(res[:, 1]))
    xs[:] = x
    ys[:] = y


def _encode_hilbertcurve(xs, ys, out, *, nbits):
    from hilbertcurve.hilbertcurve import HilbertCurve

    coords = np.stack([xs, ys], axis=-1)
    hc = HilbertCurve(int(nbits), 2)
    out[:] = np.asarray(hc.distances_from_points(coords))
    return out


def _decode_hilbertcurve(idx, xs, ys, *, nbits):
    from hilbertcurve.hilbertcurve import HilbertCurve

    hc = HilbertCurve(int(nbits), 2)
    coords = np.asarray(hc.points_from_distances(idx))
    xs[:] = coords[:, 0]
    ys[:] = coords[:, 1]


HILBERTSFC_2D = HilbertImplementation(
    name="hilbertsfc/2d",
    encode=_encode_hilbertsfc,
    decode=_decode_hilbertsfc,
)

HILBERTSFC_MORTON_2D = HilbertImplementation(
    name="hilbertsfc-morton/2d",
    encode=_encode_hilbertsfc_morton,
    decode=_decode_hilbertsfc_morton,
)

NUMPY_HILBERT_CURVE_2D = HilbertImplementation(
    name="numpy-hilbert-curve/2d",
    encode=_encode_numpy_hilbert_curve,
    decode=_decode_numpy_hilbert_curve,
)

HILBERT_BYTES_2D = HilbertImplementation(
    name="hilbert-bytes/2d",
    encode=_encode_hilbert_bytes,
    decode=_decode_hilbert_bytes,
)

HILBERTCURVE_2D = HilbertImplementation(
    name="hilbertcurve/2d",
    encode=_encode_hilbertcurve,
    decode=_decode_hilbertcurve,
)

IMPLS: dict[str, HilbertImplementation] = {
    impl.name: impl
    for impl in (
        HILBERTSFC_2D,
        HILBERTSFC_MORTON_2D,
        HILBERT_BYTES_2D,
        NUMPY_HILBERT_CURVE_2D,
        HILBERTCURVE_2D,
    )
}
