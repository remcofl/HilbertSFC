"""hilbertsfc package.

This package is intended to host Hilbert space-filling curve kernels and their
lookup tables as lazily-loaded package resources.

Public API lives in:
- ``hilbertsfc.hilbert2d``
- ``hilbertsfc.hilbert3d``
"""

from __future__ import annotations

from ._cache import clear_all_caches, clear_kernel_caches, clear_lut_caches
from .hilbert2d import (
    get_hilbert_decode_2d_kernel,
    get_hilbert_encode_2d_kernel,
    hilbert_decode_2d,
    hilbert_encode_2d,
)
from .hilbert3d import (
    get_hilbert_decode_3d_kernel,
    get_hilbert_encode_3d_kernel,
    hilbert_decode_3d,
    hilbert_encode_3d,
)

__all__ = [
    "clear_all_caches",
    "clear_kernel_caches",
    "clear_lut_caches",
    "get_hilbert_decode_2d_kernel",
    "get_hilbert_encode_2d_kernel",
    "get_hilbert_decode_3d_kernel",
    "get_hilbert_encode_3d_kernel",
    "hilbert_decode_2d",
    "hilbert_decode_3d",
    "hilbert_encode_2d",
    "hilbert_encode_3d",
]
