"""Pluggable 3d hilbert implementations for scripts/hilbert/bench_cli.py.

Usage:
  python scripts/hilbert/bench_cli.py --impl scripts/hilbert/hilbert_impls_3d.py:<impl> --nbits 16 --n 200000
"""

from __future__ import annotations

import numpy as np
from hilbert_bench import HilbertImplementation


def get_hilbertsfc_impl_3d(nbits: int):
    from hilbertsfc import hilbert_decode_3d, hilbert_encode_3d

    def enc(xs, ys, zs, out):
        return hilbert_encode_3d(xs, ys, zs, nbits=nbits, parallel=True, out=out)

    def dec(idx, xs, ys, zs):
        return hilbert_decode_3d(
            idx, nbits=nbits, parallel=True, out_xs=xs, out_ys=ys, out_zs=zs
        )

    name = f"hilbertsfc/3d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}


def get_numpy_hilbert_curve_impl_3d(nbits: int):
    from hilbert import decode, encode

    def enc(xs, ys, zs):
        coords = np.stack([xs, ys, zs], axis=-1)
        return encode(coords, num_dims=3, num_bits=nbits)

    def dec(idx):
        res = decode(idx, num_dims=3, num_bits=nbits)
        return res[:, 0], res[:, 1], res[:, 2]

    name = f"numpy-hilbert-curve/3d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}


def get_hilbert_bytes_impl_3d(nbits: int):
    from hilbert_bytes import decode, encode

    coord_nbytes = (int(nbits) + 7) // 8
    idx_nbytes = 3 * coord_nbytes

    def to_big_endian_bytes(arr: np.ndarray) -> np.ndarray:
        dt = arr.dtype
        be = arr.astype(arr.dtype.newbyteorder(">"), copy=False)
        return be.view(np.uint8).reshape(arr.shape + (dt.itemsize,))

    def _bytes_to_u32(arr_bytes: np.ndarray) -> np.ndarray:
        """Convert big-endian bytes (n, p) to uint32.

        With nbits<=21, p=coord_nbytes is 1..3.
        """
        p = int(arr_bytes.shape[-1])
        arr_bytes = np.ascontiguousarray(arr_bytes)
        if p == 1:
            return arr_bytes[:, 0].astype(np.uint32)
        if p == 2:
            return arr_bytes.view(">u2").reshape(-1).astype(np.uint32)
        # p == 3: left-pad to 4 bytes then view >u4.
        tmp = np.empty((arr_bytes.shape[0], 4), dtype=np.uint8)
        tmp[:, 0] = 0
        tmp[:, 1:] = arr_bytes
        return tmp.view(">u4").reshape(-1).astype(np.uint32)

    def _idx_bytes_to_u64(idx_bytes: np.ndarray) -> np.ndarray:
        """Convert big-endian index bytes (n, dp) to uint64.

        With nbits<=21, dp=idx_nbytes is 3, 6, or 9. For dp==9 the leading
        byte is always 0 because 3*nbits <= 63.
        """
        idx_bytes = np.ascontiguousarray(idx_bytes)
        dp = int(idx_bytes.shape[-1])
        if dp >= 8:
            b8 = idx_bytes[:, -8:]
        else:
            b8 = np.empty((idx_bytes.shape[0], 8), dtype=np.uint8)
            b8[:, : 8 - dp] = 0
            b8[:, 8 - dp :] = idx_bytes
        return b8.view(">u8").reshape(-1).astype(np.uint64)

    def _u64_to_idx_bytes(idx: np.ndarray) -> np.ndarray:
        """Convert uint64 indices (n,) to big-endian bytes (n, dp)."""
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

    def enc(xs, ys, zs):
        coords = np.stack([xs, ys, zs], axis=-1)
        coords_bytes_full = to_big_endian_bytes(coords)
        # Use exactly ceil(nbits/8) bytes per coordinate.
        coords_bytes = np.ascontiguousarray(
            coords_bytes_full[..., -coord_nbytes:]
        ).copy()
        idx_bytes = encode(coords_bytes)
        return _idx_bytes_to_u64(idx_bytes)

    def dec(idx):
        # Provide exactly 3*ceil(nbits/8) bytes per index so decode uses the
        # same bytes-per-dimension as encode.
        idx_bytes = _u64_to_idx_bytes(np.asarray(idx))
        res = decode(idx_bytes, ndim=3)
        x = np.ascontiguousarray(res[:, 0])
        y = np.ascontiguousarray(res[:, 1])
        z = np.ascontiguousarray(res[:, 2])
        x = _bytes_to_u32(x)
        y = _bytes_to_u32(y)
        z = _bytes_to_u32(z)
        return x, y, z

    name = f"hilbert-bytes/3d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}


def get_hilbertcurve_impl_3d(nbits: int):
    from hilbertcurve.hilbertcurve import HilbertCurve

    def enc(xs, ys, zs):
        coords = np.stack([xs, ys, zs], axis=-1)
        hc = HilbertCurve(nbits, 3)
        idx = hc.distances_from_points(coords)
        return np.array(idx)

    def dec(idx):
        hc = HilbertCurve(nbits, 3)
        coords = hc.points_from_distances(idx)
        coords = np.array(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        return x, y, z

    name = f"hilbertcurve/3d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}
