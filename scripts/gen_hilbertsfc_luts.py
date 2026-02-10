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
"""

from pathlib import Path

import numpy as np
from hilbertsfc_gen.lut_2d4b import generate_luts_2d4b_compacted
from hilbertsfc_gen.lut_3d2b import generate_luts_3d2b

LUT_2D4B_B_QS_U64_NPY = "lut_2d4b_b_qs_u64.npy"
LUT_2D4B_Q_BS_U64_NPY = "lut_2d4b_q_bs_u64.npy"
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
        "--out",
        type=Path,
        default=_default_out_dir(),
        help="Output directory for .npy LUT files. Default: src/hilbertsfc/_data",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    lut_2d4b_b_qs_u64, lut_2d4b_q_bs_u64 = generate_luts_2d4b_compacted()
    lut_3d2b_sb_so_u16, lut_3d2b_so_sb_u16 = generate_luts_3d2b()

    p1 = out_dir / LUT_2D4B_B_QS_U64_NPY
    p2 = out_dir / LUT_2D4B_Q_BS_U64_NPY
    p3 = out_dir / LUT_3D2B_SB_SO_U16_NPY
    p4 = out_dir / LUT_3D2B_SO_SB_U16_NPY

    np.save(p1, lut_2d4b_b_qs_u64, allow_pickle=False)
    np.save(p2, lut_2d4b_q_bs_u64, allow_pickle=False)
    np.save(p3, lut_3d2b_sb_so_u16, allow_pickle=False)
    np.save(p4, lut_3d2b_so_sb_u16, allow_pickle=False)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")
    print(f"Wrote {p4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
