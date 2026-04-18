"""Plain torch 3D Hilbert kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_3d

from ..._luts import TorchCacheMode, lut_3d2b_sb_so_i16

_LOW_MASK_3D_COORD_BITS: tuple[int, ...] = tuple((1 << n) - 1 for n in range(22))


def _hilbert_encode_3d_2bit_so(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    out: torch.Tensor,
    nbits: int,
    lut: torch.Tensor,
) -> None:
    out.fill_(0)
    state = torch.zeros_like(x, dtype=torch.int32)

    start_bit = ((nbits - 1) // 2) * 2
    drop_bits = start_bit - nbits + 2
    if drop_bits > 0:
        mask = _LOW_MASK_3D_COORD_BITS[nbits]
        x = x & mask
        y = y & mask
        z = z & mask

    # Minimal dtype for indexing (packing requires int16 minimal)
    if x.dtype.itemsize < 4:
        x = x.to(torch.int32)
    if y.dtype.itemsize < 4:
        y = y.to(torch.int32)
    if z.dtype.itemsize < 4:
        z = z.to(torch.int32)

    for bit in range(start_bit, -1, -2):
        b_x = (x >> bit) & 0x3
        b_y = (y >> bit) & 0x3
        b_z = (z >> bit) & 0x3
        b = (b_x << 4) | (b_y << 2) | b_z

        so = lut[(state | b)]
        o = so & 0x3F
        out |= o.to(out.dtype) << (3 * bit)
        state = so & 0x7C0


def hilbert_encode_3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
) -> torch.Tensor:
    """Internal plain-torch 3D Hilbert encoder."""

    validate_nbits_3d(nbits)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    lut = lut_3d2b_sb_so_i16(device=x.device, cache=lut_cache)
    _hilbert_encode_3d_2bit_so(x, y, z, out, nbits=nbits, lut=lut)

    return out
