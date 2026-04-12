"""Plain torch 2D Hilbert kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.types import TileNBits2D

from ..._luts import (
    TorchCacheMode,
    lut_2d4b_q_bs_i64,
    lut_2d7b_q_bs_i64,
)


def _hilbert_decode_2d_4bit_compacted_bs(
    idx: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    nbits: int,
    lut: torch.Tensor,
) -> None:
    out_x.fill_(0)
    out_y.fill_(0)
    state = torch.zeros_like(out_x, dtype=torch.int8)

    start_bit = (nbits - 1) & ~0x3
    drop_bits = start_bit - nbits + 4
    if drop_bits > 0:
        mask = (1 << (nbits << 1)) - 1
        idx = idx & mask

    if idx.dtype.itemsize < 4:
        idx = idx.to(torch.int32)

    for bit in range(start_bit, -1, -4):
        q = (idx >> (bit << 1)) & 0xFF
        bs = lut[q] >> state

        b_x = (bs & 0xF000) >> 12
        b_y = (bs & 0x0F00) >> 8
        out_x |= b_x << bit
        out_y |= b_y << bit

        state = bs & 0xFF


def _hilbert_decode_2d_7bit_compacted_bs(
    idx: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    nbits: int,
    lut: torch.Tensor,
) -> None:
    out_x.fill_(0)
    out_y.fill_(0)
    start_bit = (nbits - 1) // 7 * 7
    state = torch.full_like(out_x, (start_bit + 7) & 0x1, dtype=torch.int8)

    drop_bits = start_bit - nbits + 7
    if drop_bits > 0:
        mask = (1 << (nbits << 1)) - 1
        idx = idx & mask

    if idx.dtype.itemsize < 4:
        idx = idx.to(torch.int32)

    for bit in range(start_bit, -1, -7):
        q = (idx >> (bit << 1)) & 0x3FFF
        bs = lut[q] >> (state << 4)

        b_x = (bs & 0xFE00) >> 9
        b_y = (bs & 0x01FC) >> 2
        out_x |= b_x << bit
        out_y |= b_y << bit

        state = bs & 0x3


def _auto_tile_nbits_2d(nbits: int) -> TileNBits2D:
    """Determine the best tile size for 2D *decode* kernels given ``nbits``.

    Policy ``ceil(nbits/4) == ceil(nbits/7)`` -> 4, else 7. This means:
    - ``4`` for very small domains (``nbits <= 4``) and for ``nbits == 8``.
    - ``7`` otherwise.
    """

    if nbits <= 4 or nbits == 8:
        return 4
    return 7


def hilbert_decode_2d_torch(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal plain-torch 2D Hilbert decoder."""

    validate_nbits_2d(nbits)

    if out_x is None:
        out_x = torch.empty_like(index, dtype=torch.int64)
    if out_y is None:
        out_y = torch.empty_like(index, dtype=torch.int64)

    tile = 4 if torch.compiler.is_compiling() else _auto_tile_nbits_2d(nbits)

    if tile == 4:
        lut = lut_2d4b_q_bs_i64(device=index.device, cache=lut_cache)
        _hilbert_decode_2d_4bit_compacted_bs(index, out_x, out_y, nbits=nbits, lut=lut)
    else:
        lut = lut_2d7b_q_bs_i64(device=index.device, cache=lut_cache)
        _hilbert_decode_2d_7bit_compacted_bs(index, out_x, out_y, nbits=nbits, lut=lut)

    return out_x, out_y
