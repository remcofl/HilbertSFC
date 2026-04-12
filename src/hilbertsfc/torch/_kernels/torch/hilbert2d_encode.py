"""Plain torch 2D Hilbert kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.types import TileNBits2D

from ..._luts import (
    TorchCacheMode,
    lut_2d4b_b_qs_i64,
    lut_2d7b_b_qs_i64,
)


def _hilbert_encode_2d_4bit_compacted_qs(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, nbits: int, lut: torch.Tensor
) -> None:
    out.fill_(0)
    state = torch.zeros_like(x, dtype=torch.int8)

    start_bit = (nbits - 1) & ~0x3
    drop_bits = start_bit - nbits + 4
    if drop_bits > 0:
        mask = (1 << nbits) - 1
        x = x & mask
        y = y & mask

    # Minimal dtype for indexing
    if x.dtype.itemsize < 4:
        x = x.to(torch.int32)
    if y.dtype.itemsize < 4:
        y = y.to(torch.int32)

    for bit in range(start_bit, -1, -4):
        b_x = (x >> bit) & 0xF
        b_y = (y >> bit) & 0xF
        b = (b_x << 4) | b_y
        qs = lut[b] >> state
        q = (qs & 0xFF00) >> 8
        out |= q << (bit << 1)
        state = qs & 0xFF


def _hilbert_encode_2d_7bit_compacted_qs(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, nbits: int, lut: torch.Tensor
) -> None:
    out.fill_(0)
    start_bit = (nbits - 1) // 7 * 7
    state = torch.full_like(x, (start_bit + 7) & 0x1, dtype=torch.int8)

    drop_bits = start_bit - nbits + 7
    if drop_bits > 0:
        mask = (1 << nbits) - 1
        x = x & mask
        y = y & mask

    # Minimal dtype for indexing
    if x.dtype.itemsize < 4:
        x = x.to(torch.int32)
    if y.dtype.itemsize < 4:
        y = y.to(torch.int32)

    for bit in range(start_bit, -1, -7):
        b_x = (x >> bit) & 0x7F
        b_y = (y >> bit) & 0x7F
        b = (b_x << 7) | b_y
        qs = lut[b] >> (state << 4)
        q = (qs & 0xFFFC) >> 2
        out |= q << (bit << 1)
        state = qs & 0x3


def _auto_tile_nbits_2d(nbits: int) -> TileNBits2D:
    """Determine the best tile size for 2D *encode* kernels given ``nbits``.

    Policy ``ceil(nbits/4) == ceil(nbits/7)`` -> 4, else 7. This means:
    - ``4`` for very small domains (``nbits <= 4``) and for ``nbits == 8``.
    - ``7`` otherwise.
    """

    if nbits <= 4 or nbits == 8:
        return 4
    return 7


def hilbert_encode_2d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
) -> torch.Tensor:
    """Internal plain-torch 2D Hilbert encoder."""

    validate_nbits_2d(nbits)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    tile = 4 if torch.compiler.is_compiling() else _auto_tile_nbits_2d(nbits)

    if tile == 4:
        lut = lut_2d4b_b_qs_i64(device=x.device, cache=lut_cache)
        _hilbert_encode_2d_4bit_compacted_qs(x, y, out, nbits=nbits, lut=lut)
    else:
        lut = lut_2d7b_b_qs_i64(device=x.device, cache=lut_cache)
        _hilbert_encode_2d_7bit_compacted_qs(x, y, out, nbits=nbits, lut=lut)

    return out
