"""Compatibility re-exports for shared public API wrappers."""

from ._public_api_adapters import (
    Decode2DAdapter,
    Decode3DAdapter,
    Encode2DAdapter,
    Encode3DAdapter,
)
from ._public_api_shared_2d import decode_2d_api, encode_2d_api
from ._public_api_shared_3d import decode_3d_api, encode_3d_api

__all__ = [
    "Decode2DAdapter",
    "Decode3DAdapter",
    "Encode2DAdapter",
    "Encode3DAdapter",
    "decode_2d_api",
    "decode_3d_api",
    "encode_2d_api",
    "encode_3d_api",
]
