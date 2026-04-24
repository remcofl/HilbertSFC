"""Plain torch 2D Morton kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_2d

_LOW_MASK_2D_COORD_BITS: tuple[int, ...] = tuple((1 << n) - 1 for n in range(33))


def _part1by1_i32(x: torch.Tensor, nbits: int) -> torch.Tensor:
    if nbits > 8:
        x = (x | (x << 8)) & 0x00FF00FF
    if nbits > 4:
        x = (x | (x << 4)) & 0x0F0F0F0F
    if nbits > 2:
        x = (x | (x << 2)) & 0x33333333
    if nbits > 1:
        x = (x | (x << 1)) & 0x55555555
    return x


def _part1by1_i64(x: torch.Tensor, nbits: int) -> torch.Tensor:
    if nbits > 16:
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    if nbits > 8:
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    if nbits > 4:
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    if nbits > 2:
        x = (x | (x << 2)) & 0x3333333333333333
    if nbits > 1:
        x = (x | (x << 1)) & 0x5555555555555555
    return x


def _morton_encode_2d_i32(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, nbits: int
) -> None:
    mask = _LOW_MASK_2D_COORD_BITS[nbits]
    x = (x.to(torch.int32) if x.dtype.itemsize != 4 else x) & mask
    y = (y.to(torch.int32) if y.dtype.itemsize != 4 else y) & mask

    x_part = _part1by1_i32(x, nbits)
    y_part = _part1by1_i32(y, nbits)
    if out.dtype.itemsize > 4:
        out.fill_(0)
        out |= x_part.to(out.dtype)
        out |= y_part.to(out.dtype) << 1
    else:
        out.copy_(x_part | (y_part << 1))


def _morton_encode_2d_i64(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, nbits: int
) -> None:
    if x.dtype.itemsize < 8:
        x = x.to(torch.int64)
    if y.dtype.itemsize < 8:
        y = y.to(torch.int64)

    mask = _LOW_MASK_2D_COORD_BITS[nbits]
    x = x & mask
    y = y & mask

    idx = _part1by1_i64(x, nbits) | (_part1by1_i64(y, nbits) << 1)
    out.copy_(idx)


def morton_encode_2d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Internal plain-torch 2D Morton encoder."""

    validate_nbits_2d(nbits)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    if nbits <= 16:
        _morton_encode_2d_i32(x, y, out, nbits)
    else:
        _morton_encode_2d_i64(x, y, out, nbits)

    return out
