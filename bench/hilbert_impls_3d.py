"""Pluggable 3D Hilbert implementations for ``bench/bench_cli.py``.

Usage:
    python bench/bench_cli.py --impl bench/hilbert_impls_3d.py:IMPLS --ndim 3 --nbits 16 --n 200000
"""

from __future__ import annotations

import numpy as np
from hilbert_bench import HilbertImplementation


def _encode_hilbertsfc(xs, ys, zs, out, *, nbits):
    from hilbertsfc import hilbert_encode_3d

    return hilbert_encode_3d(xs, ys, zs, nbits=nbits, parallel=True, out=out)


def _decode_hilbertsfc(idx, xs, ys, zs, *, nbits):
    from hilbertsfc import hilbert_decode_3d

    return hilbert_decode_3d(
        idx, nbits=nbits, parallel=True, out_xs=xs, out_ys=ys, out_zs=zs
    )


def _encode_numpy_hilbert_curve(xs, ys, zs, out, *, nbits):
    from hilbert import encode

    coords = np.stack([xs, ys, zs], axis=-1)
    out[:] = encode(coords, num_dims=3, num_bits=nbits)
    return out


def _decode_numpy_hilbert_curve(idx, xs, ys, zs, *, nbits):
    from hilbert import decode

    res = decode(idx, num_dims=3, num_bits=nbits)
    xs[:] = res[:, 0]
    ys[:] = res[:, 1]
    zs[:] = res[:, 2]


def _to_big_endian_bytes(arr: np.ndarray) -> np.ndarray:
    dt = arr.dtype
    be = arr.astype(arr.dtype.newbyteorder(">"), copy=False)
    return be.view(np.uint8).reshape(arr.shape + (dt.itemsize,))


def _bytes_to_u32(arr_bytes: np.ndarray) -> np.ndarray:
    p = int(arr_bytes.shape[-1])
    arr_bytes = np.ascontiguousarray(arr_bytes)
    if p == 1:
        return arr_bytes[:, 0].astype(np.uint32)
    if p == 2:
        return arr_bytes.view(">u2").reshape(-1).astype(np.uint32)
    tmp = np.empty((arr_bytes.shape[0], 4), dtype=np.uint8)
    tmp[:, 0] = 0
    tmp[:, 1:] = arr_bytes
    return tmp.view(">u4").reshape(-1).astype(np.uint32)


def _idx_bytes_to_u64(idx_bytes: np.ndarray) -> np.ndarray:
    idx_bytes = np.ascontiguousarray(idx_bytes)
    dp = int(idx_bytes.shape[-1])
    if dp >= 8:
        b8 = idx_bytes[:, -8:]
    else:
        b8 = np.empty((idx_bytes.shape[0], 8), dtype=np.uint8)
        b8[:, : 8 - dp] = 0
        b8[:, 8 - dp :] = idx_bytes
    return b8.view(">u8").reshape(-1).astype(np.uint64)


def _u64_to_idx_bytes(idx: np.ndarray, idx_nbytes: int) -> np.ndarray:
    b8 = (
        np.ascontiguousarray(idx.astype(">u8", copy=False))
        .view(np.uint8)
        .reshape((-1, 8))
    )
    if idx_nbytes == 9:
        out = np.empty((b8.shape[0], 9), dtype=np.uint8)
        out[:, 0] = 0
        out[:, 1:] = b8
        return out
    return b8[:, -idx_nbytes:].copy()


def _encode_hilbert_bytes(xs, ys, zs, out, *, nbits):
    from hilbert_bytes import encode

    coord_nbytes = (int(nbits) + 7) // 8
    coords = np.stack([xs, ys, zs], axis=-1)
    coords_bytes_full = _to_big_endian_bytes(coords)
    coords_bytes = np.ascontiguousarray(coords_bytes_full[..., -coord_nbytes:]).copy()
    idx_bytes = encode(coords_bytes)
    out[:] = _idx_bytes_to_u64(idx_bytes)
    return out


def _decode_hilbert_bytes(idx, xs, ys, zs, *, nbits):
    from hilbert_bytes import decode

    coord_nbytes = (int(nbits) + 7) // 8
    idx_nbytes = 3 * coord_nbytes
    idx_bytes = _u64_to_idx_bytes(np.asarray(idx), idx_nbytes=idx_nbytes)
    res = decode(idx_bytes, ndim=3)
    x = _bytes_to_u32(np.ascontiguousarray(res[:, 0]))
    y = _bytes_to_u32(np.ascontiguousarray(res[:, 1]))
    z = _bytes_to_u32(np.ascontiguousarray(res[:, 2]))
    xs[:] = x
    ys[:] = y
    zs[:] = z


def _encode_hilbertcurve(xs, ys, zs, out, *, nbits):
    from hilbertcurve.hilbertcurve import HilbertCurve

    coords = np.stack([xs, ys, zs], axis=-1)
    hc = HilbertCurve(int(nbits), 3)
    out[:] = np.asarray(hc.distances_from_points(coords))
    return out


def _decode_hilbertcurve(idx, xs, ys, zs, *, nbits):
    from hilbertcurve.hilbertcurve import HilbertCurve

    hc = HilbertCurve(int(nbits), 3)
    coords = np.asarray(hc.points_from_distances(idx))
    xs[:] = coords[:, 0]
    ys[:] = coords[:, 1]
    zs[:] = coords[:, 2]


HILBERTSFC_3D = HilbertImplementation(
    name="hilbertsfc/3d",
    encode=_encode_hilbertsfc,
    decode=_decode_hilbertsfc,
)

NUMPY_HILBERT_CURVE_3D = HilbertImplementation(
    name="numpy-hilbert-curve/3d",
    encode=_encode_numpy_hilbert_curve,
    decode=_decode_numpy_hilbert_curve,
)

HILBERT_BYTES_3D = HilbertImplementation(
    name="hilbert-bytes/3d",
    encode=_encode_hilbert_bytes,
    decode=_decode_hilbert_bytes,
)

HILBERTCURVE_3D = HilbertImplementation(
    name="hilbertcurve/3d",
    encode=_encode_hilbertcurve,
    decode=_decode_hilbertcurve,
)

IMPLS: dict[str, HilbertImplementation] = {
    impl.name: impl
    for impl in (
        HILBERTSFC_3D,
        HILBERT_BYTES_3D,
        NUMPY_HILBERT_CURVE_3D,
        HILBERTCURVE_3D,
    )
}
