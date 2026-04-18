"""Plain torch 3D Hilbert kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_3d

from ..._luts import TorchCacheMode, lut_3d2b_so_sb_i16

_LOW_MASK_3D_INDEX_BITS: tuple[int, ...] = tuple((1 << (3 * n)) - 1 for n in range(22))


def _hilbert_decode_3d_2bit_sb(
    idx: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    out_z: torch.Tensor,
    nbits: int,
    lut: torch.Tensor,
) -> None:
    out_x.fill_(0)
    out_y.fill_(0)
    out_z.fill_(0)

    state = torch.zeros_like(out_x, dtype=torch.int32)

    start_bit = ((nbits - 1) // 2) * 2
    drop_bits = start_bit - nbits + 2
    if drop_bits > 0:
        mask = _LOW_MASK_3D_INDEX_BITS[nbits]
        idx = idx & mask

    if idx.dtype.itemsize < 4:
        idx = idx.to(torch.int32)

    for bit in range(start_bit, -1, -2):
        o = (idx >> (3 * bit)) & 0x3F
        sb = lut[(state | o)]

        b_x = (sb & 0x30) >> 4
        b_y = (sb & 0x0C) >> 2
        b_z = sb & 0x03

        out_x |= b_x.to(out_x.dtype) << bit
        out_y |= b_y.to(out_y.dtype) << bit
        out_z |= b_z.to(out_z.dtype) << bit

        state = sb & 0x7C0


def hilbert_decode_3d_torch(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_z: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal plain-torch 3D Hilbert decoder."""

    validate_nbits_3d(nbits)

    if out_x is None:
        out_x = torch.empty_like(index, dtype=torch.int64)
    if out_y is None:
        out_y = torch.empty_like(index, dtype=torch.int64)
    if out_z is None:
        out_z = torch.empty_like(index, dtype=torch.int64)

    lut = lut_3d2b_so_sb_i16(device=index.device, cache=lut_cache)
    _hilbert_decode_3d_2bit_sb(index, out_x, out_y, out_z, nbits=nbits, lut=lut)

    return out_x, out_y, out_z
