"""Hilbert space-filling curve kernels.

The top-level `hilbertsfc` package exposes NumPy/Numba-backed 2D/3D Hilbert
encode/decode functions, plus matching Morton/z-order helpers, kernel accessors,
and cache management utilities.

For GPU acceleration and `torch.Tensor` inputs, use the optional PyTorch frontend:
[`hilbertsfc.torch`][hilbertsfc.torch].

"""

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
from .morton2d import (
    get_morton_decode_2d_kernel,
    get_morton_encode_2d_kernel,
    morton_decode_2d,
    morton_encode_2d,
)
from .morton3d import (
    get_morton_decode_3d_kernel,
    get_morton_encode_3d_kernel,
    morton_decode_3d,
    morton_encode_3d,
)

__all__ = [
    "clear_all_caches",
    "clear_kernel_caches",
    "clear_lut_caches",
    "get_hilbert_decode_2d_kernel",
    "get_hilbert_encode_2d_kernel",
    "get_hilbert_decode_3d_kernel",
    "get_hilbert_encode_3d_kernel",
    "get_morton_decode_2d_kernel",
    "get_morton_encode_2d_kernel",
    "get_morton_decode_3d_kernel",
    "get_morton_encode_3d_kernel",
    "hilbert_decode_2d",
    "hilbert_decode_3d",
    "hilbert_encode_2d",
    "hilbert_encode_3d",
    "morton_decode_2d",
    "morton_decode_3d",
    "morton_encode_2d",
    "morton_encode_3d",
]
