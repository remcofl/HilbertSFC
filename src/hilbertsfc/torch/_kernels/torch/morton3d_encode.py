"""Plain torch 3D Morton kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_3d

_LOW_MASK_3D_COORD_BITS: tuple[int, ...] = tuple((1 << n) - 1 for n in range(22))


def _part1by2_i32(x: torch.Tensor, nbits: int) -> torch.Tensor:
    if nbits > 5:
        x = (x | (x << 16)) & 0x030000FF
    if nbits > 3:
        x = (x | (x << 8)) & 0x0300F00F
    if nbits > 2:
        x = (x | (x << 4)) & 0x030C30C3
    if nbits > 1:
        x = (x | (x << 2)) & 0x09249249
    return x


def _part1by2_i64(x: torch.Tensor, nbits: int) -> torch.Tensor:
    if nbits > 10:
        x = (x | (x << 32)) & 0x001F00000000FFFF
    if nbits > 5:
        x = (x | (x << 16)) & 0x001F0000FF0000FF
    if nbits > 3:
        x = (x | (x << 8)) & 0x100F00F00F00F00F
    if nbits > 2:
        x = (x | (x << 4)) & 0x10C30C30C30C30C3
    if nbits > 1:
        x = (x | (x << 2)) & 0x1249249249249249
    return x


def _morton_encode_3d_i32(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    out: torch.Tensor,
    nbits: int,
) -> None:
    mask = _LOW_MASK_3D_COORD_BITS[nbits]
    x = (x.to(torch.int32) if x.dtype.itemsize != 4 else x) & mask
    y = (y.to(torch.int32) if y.dtype.itemsize != 4 else y) & mask
    z = (z.to(torch.int32) if z.dtype.itemsize != 4 else z) & mask

    x_part = _part1by2_i32(x, nbits)
    y_part = _part1by2_i32(y, nbits)
    z_part = _part1by2_i32(z, nbits)
    if out.dtype.itemsize > 4:
        out.fill_(0)
        out |= x_part.to(out.dtype)
        out |= y_part.to(out.dtype) << 1
        out |= z_part.to(out.dtype) << 2
    else:
        out.copy_(x_part | (y_part << 1) | (z_part << 2))


def _morton_encode_3d_i64(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    out: torch.Tensor,
    nbits: int,
) -> None:
    if x.dtype.itemsize < 8:
        x = x.to(torch.int64)
    if y.dtype.itemsize < 8:
        y = y.to(torch.int64)
    if z.dtype.itemsize < 8:
        z = z.to(torch.int64)

    mask = _LOW_MASK_3D_COORD_BITS[nbits]
    x = x & mask
    y = y & mask
    z = z & mask

    idx = (
        _part1by2_i64(x, nbits)
        | (_part1by2_i64(y, nbits) << 1)
        | (_part1by2_i64(z, nbits) << 2)
    )
    out.copy_(idx)


def morton_encode_3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal plain-torch 3D Morton encoder."""

    validate_nbits_3d(nbits)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    if nbits <= 10:
        _morton_encode_3d_i32(x, y, z, out, nbits)
    else:
        _morton_encode_3d_i64(x, y, z, out, nbits)

    return out
