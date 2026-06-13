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

For a specific set of 2D tile sizes (in bits) between 1 and 7:
    uv run scripts/gen_hilbertsfc_luts.py --2d-nbits 4 6

For a specific set of 3D tile sizes (in bits) between 1 and 4:
    uv run scripts/gen_hilbertsfc_luts.py --3d-nbits 2 3

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
from hilbertsfc_gen.lut_3dnb import generate_luts_3dnb_flat


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
        "--3d-n",
        "--3d-nbits",
        dest="tile_nbits_3d",
        nargs="+",
        type=int,
        default=[2, 3],
        help="3D tile sizes in bits (iterations per lookup). Default: 2 3",
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
    nbits_3d = list(dict.fromkeys(args.tile_nbits_3d))  # stable unique
    for n in nbits_2d:
        if n < 1 or n > 7:
            raise SystemExit(f"--2d-n values must be in [1, 7]; got {n}")
    for n in nbits_3d:
        if n < 1 or n > 4:
            raise SystemExit(f"--3d-n values must be in [1, 4]; got {n}")

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
    for n in nbits_3d:
        lut_sb_so, lut_so_sb = generate_luts_3dnb_flat(n)
        dtype_name = f"u{lut_sb_so.dtype.itemsize * 8}"
        p_sb_so = out_dir / f"lut_3d{n}b_sb_so_{dtype_name}.npy"
        p_so_sb = out_dir / f"lut_3d{n}b_so_sb_{dtype_name}.npy"
        np.save(p_sb_so, lut_sb_so, allow_pickle=False)
        np.save(p_so_sb, lut_so_sb, allow_pickle=False)
        written.extend([p_sb_so, p_so_sb])

    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
