"""Example pluggable implementations for scripts/hilbert/bench_cli.py.

Usage:
  python scripts/hilbert/bench_cli.py --builtin pack --nbits 16 --n 200000 --threads 8

Or as a plugin module:
  python scripts/hilbert/bench_cli.py --impl scripts/hilbert/hilbert_impls_example.py:IMPLS --nbits 16 --n 200000

This file is intentionally NOT Hilbert; it's just a template showing how to
export a `HilbertImplementation` or dict of them.
"""

from __future__ import annotations

import numpy as np
from hilbert_bench import HilbertImplementation


def get_hilbertsfc_impl(nbits: int):
    from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

    def enc(xs, ys, out):
        return hilbert_encode_2d(xs, ys, nbits=nbits, parallel=True, out=out)

    def dec(idx, xs, ys):
        return hilbert_decode_2d(idx, nbits=nbits, parallel=True, out_xs=xs, out_ys=ys)

    name = f"hilbertsfc/2d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}


def get_numpy_hilbert_curve_impl(nbits: int):
    from hilbert import decode, encode

    def enc(xs, ys):
        coords = np.stack([xs, ys], axis=-1)
        return encode(coords, num_dims=2, num_bits=nbits)

    def dec(idx):
        res = decode(idx, num_dims=2, num_bits=nbits)
        return res[:, 0], res[:, 1]

    name = f"numpy-hilbert-curve/2d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}


def get_hilbert_bytes_impl(nbits: int):
    from hilbert_bytes import decode, encode

    def enc(xs, ys):
        coords = np.stack([xs, ys], axis=-1)
        coords_bytes = coords[..., None].astype(">u8").view("u1")
        idx_bytes = encode(coords_bytes)
        return idx_bytes.view(">u8").astype("u8")[..., -1]

    def dec(idx):
        idx_bytes = idx[..., None].astype(">u8").view("u1")
        res = decode(idx_bytes, ndim=2)
        x = np.ascontiguousarray(res[:, 0]).view(">u4").astype("u4")[..., 0]
        y = np.ascontiguousarray(res[:, 1]).view(">u4").astype("u4")[..., 0]
        return x, y

    name = f"hilbert-bytes/2d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}


def get_hilbertcurve_impl(nbits: int):
    from hilbertcurve.hilbertcurve import HilbertCurve

    def enc(xs, ys):
        coords = np.stack([xs, ys], axis=-1)
        hc = HilbertCurve(nbits, 2)
        idx = hc.distances_from_points(coords)
        return np.array(idx)

    def dec(idx):
        hc = HilbertCurve(nbits, 2)
        coords = hc.points_from_distances(idx)
        coords = np.array(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        return x, y

    name = f"hilbertcurve/2d/nbits{int(nbits)}"
    return {name: HilbertImplementation(name=name, encode=enc, decode=dec)}
