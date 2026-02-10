from __future__ import annotations

import numpy as np

from hilbertsfc import clear_lut_caches
from hilbertsfc._luts import (
    lut_2d4b_b_qs_u64,
    lut_2d4b_q_bs_u64,
    lut_3d2b_sb_so,
    lut_3d2b_so_sb,
)


def test_lut_2d4b_encode_decode_load_and_readonly() -> None:
    enc = lut_2d4b_b_qs_u64()
    dec = lut_2d4b_q_bs_u64()

    assert enc.dtype == np.dtype(np.uint64)
    assert dec.dtype == np.dtype(np.uint64)

    # These are numpy arrays loaded from .npy; should be non-writeable.
    assert enc.flags.writeable is False
    assert dec.flags.writeable is False


def test_lut_3d2b_dtype_widening_and_caching() -> None:
    a16 = lut_3d2b_sb_so(np.uint16)
    b16 = lut_3d2b_sb_so(np.uint16)
    assert a16 is b16
    assert a16.dtype == np.dtype(np.uint16)

    a32 = lut_3d2b_sb_so(np.uint32)
    b32 = lut_3d2b_sb_so(np.uint32)
    assert a32 is b32
    assert a32.dtype == np.dtype(np.uint32)

    assert a16 is not a32


def test_lut_caches_can_be_cleared() -> None:
    a = lut_2d4b_b_qs_u64()
    clear_lut_caches()
    b = lut_2d4b_b_qs_u64()

    # Cache clear should force a reload (new ndarray instance).
    assert a is not b


def test_lut_3d2b_pair_shapes_match() -> None:
    sb_so = lut_3d2b_sb_so(np.uint16)
    so_sb = lut_3d2b_so_sb(np.uint16)

    assert sb_so.shape == so_sb.shape
