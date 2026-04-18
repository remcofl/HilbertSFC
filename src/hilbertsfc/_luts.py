"""Lazy-loading LUT accessors.

The LUTs are shipped as ``.npy`` files under ``hilbertsfc._data``.
Access through these functions to keep imports cheap and to ensure a single
shared array instance per process.
"""

from importlib.resources import as_file, files

import numpy as np
from numpy.typing import NDArray

from . import _data
from ._cache import lut_cache
from .types import LutUIntDTypeLike, UIntArray


def _validate_lut_3d2b_uint_dtype(
    dtype: LutUIntDTypeLike,
) -> np.dtype[np.unsignedinteger]:
    req = np.dtype(dtype)
    if req not in (np.dtype(np.uint16), np.dtype(np.uint32), np.dtype(np.uint64)):
        raise ValueError("dtype must be one of: np.uint16, np.uint32, np.uint64")
    return req


def _load_npy(filename: str) -> NDArray[np.generic]:
    data_file = files(_data) / filename
    with as_file(data_file) as path:
        arr = np.load(path)
    try:
        arr.setflags(write=False)
    except Exception:
        # Not all ndarray-like objects support flags; best-effort.
        pass
    return arr


@lut_cache
def lut_2d4b_b_qs_u64() -> NDArray[np.uint64]:
    """Encode LUT: bbbb -> packed (qqqq, next_state) lanes."""
    return _load_npy("lut_2d4b_b_qs_u64.npy").astype(np.uint64, copy=False)


@lut_cache
def lut_2d4b_q_bs_u64() -> NDArray[np.uint64]:
    """Decode LUT: qqqq -> packed (bbbb, next_state) lanes."""
    return _load_npy("lut_2d4b_q_bs_u64.npy").astype(np.uint64, copy=False)


@lut_cache
def lut_2d4b_sb_sq_u16() -> NDArray[np.uint16]:
    """Encode LUT: packed (state, bbbb) -> packed (state, qqqq) lanes."""
    return _load_npy("lut_2d4b_sb_sq_u16.npy").astype(np.uint16, copy=False)


@lut_cache
def lut_2d4b_sq_sb_u16() -> NDArray[np.uint16]:
    """Decode LUT: packed (state, qqqq) -> packed (state, bbbb) lanes."""
    return _load_npy("lut_2d4b_sq_sb_u16.npy").astype(np.uint16, copy=False)


@lut_cache
def lut_2d7b_b_qs_u64() -> NDArray[np.uint64]:
    """Encode LUT: bbbbbbb -> packed (qqqqqqq, next_state) lanes."""
    return _load_npy("lut_2d7b_b_qs_u64.npy").astype(np.uint64, copy=False)


@lut_cache
def lut_2d7b_q_bs_u64() -> NDArray[np.uint64]:
    """Decode LUT: qqqqqqq -> packed (bbbbbbb, next_state) lanes."""
    return _load_npy("lut_2d7b_q_bs_u64.npy").astype(np.uint64, copy=False)


@lut_cache
def _lut_3d2b_sb_so_u16_base() -> NDArray[np.uint16]:
    return _load_npy("lut_3d2b_sb_so_u16.npy").astype(np.uint16, copy=False)


@lut_cache
def _lut_3d2b_so_sb_u16_base() -> NDArray[np.uint16]:
    return _load_npy("lut_3d2b_so_sb_u16.npy").astype(np.uint16, copy=False)


@lut_cache
def lut_3d2b_sb_so(dtype: LutUIntDTypeLike = np.uint16) -> UIntArray:
    """3D 2-bit LUT: (state, bb) -> packed (next_state, oo).

    Parameters
    ----------
    dtype:
        LUT element dtype to return. Defaults to ``np.uint16`` (storage dtype).
        Pass ``np.uint32`` or ``np.uint64`` to widen loads (useful for some
        fused-kernel scenarios).
    """

    req = _validate_lut_3d2b_uint_dtype(dtype)
    base = _lut_3d2b_sb_so_u16_base()
    return base.astype(req, copy=False)  # dtype change will allocate; cached per dtype


@lut_cache
def lut_3d2b_so_sb(dtype: LutUIntDTypeLike = np.uint16) -> UIntArray:
    """3D 2-bit LUT: (state, oo) -> packed (next_state, bb).

    See ``lut_3d2b_sb_so`` for dtype behavior.
    """

    req = _validate_lut_3d2b_uint_dtype(dtype)
    base = _lut_3d2b_so_sb_u16_base()
    return base.astype(req, copy=False)
