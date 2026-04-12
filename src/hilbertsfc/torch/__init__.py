"""PyTorch-API for HilbertSFC.

This subpackage provides 2D/3D Hilbert encode/decode functions that operate on
integer `torch.Tensor` inputs.
"""

from importlib.util import find_spec

if find_spec("torch") is None:
    raise ModuleNotFoundError(
        "Optional dependency 'torch' is required for 'hilbertsfc.torch'. "
        "Install it with: pip install 'hilbertsfc[torch]'; or install PyTorch"
        "separately and ensure it is available in your environment."
    ) from None

from ._luts import clear_torch_lut_caches, precache_compile_luts
from .hilbert2d import hilbert_decode_2d, hilbert_encode_2d
from .hilbert3d import hilbert_decode_3d, hilbert_encode_3d

__all__ = [
    "clear_torch_lut_caches",
    "hilbert_decode_2d",
    "hilbert_decode_3d",
    "hilbert_encode_2d",
    "hilbert_encode_3d",
    "precache_compile_luts",
]
