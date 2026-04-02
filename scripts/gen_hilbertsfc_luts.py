# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.4.2",
# ]
# ///
"""Generate all hilbertsfc lookup tables.

This is a development-time script that precomputes small lookup tables (LUTs)
and writes them as `.npy` files into the package data directory
`src/hilbertsfc/_data`.

It is intended for the runtime library to lazily load these `.npy` resources on demand.

Usage
-----
Run with defaults (recommended):
    uv run scripts/gen_hilbertsfc_luts.py

Optionally choose a different output directory:
    uv run scripts/gen_hilbertsfc_luts.py --out path/to/dir

For a specifc set of 2D tile sizes (in bits) between 1 and 7:
    uv run scripts/gen_hilbertsfc_luts.py --2d-nbits 4 6

Choose which 2D LUT encoding to generate:
    uv run scripts/gen_hilbertsfc_luts.py --2d-kind all
    uv run scripts/gen_hilbertsfc_luts.py --2d-kind compacted
    uv run scripts/gen_hilbertsfc_luts.py --2d-kind flat
"""

from pathlib import Path

import numpy as np
from hilbertsfc_gen.lut_2dnb import (
    generate_luts_2dnb_compacted,
    generate_luts_2dnb_flat,
)
from hilbertsfc_gen.lut_3d2b import generate_luts_3d2b

LUT_3D2B_SB_SO_U16_NPY = "lut_3d2b_sb_so_u16.npy"
LUT_3D2B_SO_SB_U16_NPY = "lut_3d2b_so_sb_u16.npy"


def _default_out_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "src" / "hilbertsfc" / "_data"


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all hilbertsfc lookup tables"
    )
    parser.add_argument(
        "--2d-n",
        "--2d-nbits",
        dest="tile_nbits_2d",
        nargs="+",
        type=int,
        default=[4, 7],
        help="2D tile sizes in bits (iterations per lookup). Default: 4 7",
    )
    parser.add_argument(
        "--2d-kind",
        type=str,
        dest="kind_2d",
        default="all",
        choices=["all", "compacted", "flat"],
        help=(
            "Which 2D LUT encoding to generate. "
            "'compacted' (aka 'stateless') packs 4 state lanes into uint64; "
            "'flat' uses uint16 with explicit (state | symbol) indexing; "
            "'all' generates both. Default: all"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_out_dir(),
        help="Output directory for .npy LUT files. Default: src/hilbertsfc/_data",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2D LUTs
    kind_2d = args.kind_2d
    nbits_2d = list(dict.fromkeys(args.tile_nbits_2d))  # stable unique
    for n in nbits_2d:
        if n < 1 or n > 7:
            raise SystemExit(f"--2d-n values must be in [1, 7]; got {n}")

    written: list[Path] = []
    for n in nbits_2d:
        if kind_2d in ("all", "compacted"):
            lut_b_qs_u64, lut_q_bs_u64 = generate_luts_2dnb_compacted(n)

            p_b_qs = out_dir / f"lut_2d{n}b_b_qs_u64.npy"
            p_q_bs = out_dir / f"lut_2d{n}b_q_bs_u64.npy"
            np.save(p_b_qs, lut_b_qs_u64, allow_pickle=False)
            np.save(p_q_bs, lut_q_bs_u64, allow_pickle=False)
            written.extend([p_b_qs, p_q_bs])

        if kind_2d in ("all", "flat"):
            lut_sb_sq_u16, lut_sq_sb_u16 = generate_luts_2dnb_flat(n)

            p_sb_sq = out_dir / f"lut_2d{n}b_sb_sq_u16.npy"
            p_sq_sb = out_dir / f"lut_2d{n}b_sq_sb_u16.npy"
            np.save(p_sb_sq, lut_sb_sq_u16, allow_pickle=False)
            np.save(p_sq_sb, lut_sq_sb_u16, allow_pickle=False)
            written.extend([p_sb_sq, p_sq_sb])

    # 3D LUTs
    lut_3d2b_sb_so_u16, lut_3d2b_so_sb_u16 = generate_luts_3d2b()

    p3 = out_dir / LUT_3D2B_SB_SO_U16_NPY
    p4 = out_dir / LUT_3D2B_SO_SB_U16_NPY
    np.save(p3, lut_3d2b_sb_so_u16, allow_pickle=False)
    np.save(p4, lut_3d2b_so_sb_u16, allow_pickle=False)
    written.extend([p3, p4])

    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
