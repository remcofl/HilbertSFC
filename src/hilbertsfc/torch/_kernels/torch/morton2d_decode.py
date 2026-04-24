"""Plain torch 2D Morton decode kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_2d

_LOW_MASK_2D_INDEX_BITS: tuple[int, ...] = tuple(
    ((1 << (n << 1)) - 1) if n < 32 else -1 for n in range(33)
)


def _compact1by1_i32(x: torch.Tensor, nbits: int) -> torch.Tensor:
    x = x & 0x55555555
    if nbits > 1:
        x = (x | (x >> 1)) & 0x33333333
    if nbits > 2:
        x = (x | (x >> 2)) & 0x0F0F0F0F
    if nbits > 4:
        x = (x | (x >> 4)) & 0x00FF00FF
    if nbits > 8:
        x = (x | (x >> 8)) & 0x0000FFFF
    return x


def _compact1by1_i64(x: torch.Tensor, nbits: int) -> torch.Tensor:
    x = x & 0x5555555555555555
    if nbits > 1:
        x = (x | (x >> 1)) & 0x3333333333333333
    if nbits > 2:
        x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    if nbits > 4:
        x = (x | (x >> 4)) & 0x00FF00FF00FF00FF
    if nbits > 8:
        x = (x | (x >> 8)) & 0x0000FFFF0000FFFF
    if nbits > 16:
        x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    return x


def _morton_decode_2d_i32(
    index: torch.Tensor, out_x: torch.Tensor, out_y: torch.Tensor, nbits: int
) -> None:
    if nbits < 16:
        index = index & _LOW_MASK_2D_INDEX_BITS[nbits]
    elif index.dtype.itemsize > 4:
        index = index & 0xFFFFFFFF

    if index.dtype.itemsize != 4:
        index = index.to(torch.int32)

    out_x.copy_(_compact1by1_i32(index, nbits))
    out_y.copy_(_compact1by1_i32(index >> 1, nbits))


def _morton_decode_2d_i64(
    index: torch.Tensor, out_x: torch.Tensor, out_y: torch.Tensor, nbits: int
) -> None:
    if index.dtype.itemsize < 8:
        index = index.to(torch.int64)

    if nbits < 32:
        index = index & _LOW_MASK_2D_INDEX_BITS[nbits]

    out_x.copy_(_compact1by1_i64(index, nbits))
    out_y.copy_(_compact1by1_i64(index >> 1, nbits))


def morton_decode_2d_torch(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal plain-torch 2D Morton decoder."""

    validate_nbits_2d(nbits)

    if out_x is None:
        out_x = torch.empty_like(index, dtype=torch.int64)
    if out_y is None:
        out_y = torch.empty_like(index, dtype=torch.int64)

    if nbits <= 16:
        _morton_decode_2d_i32(index, out_x, out_y, nbits)
    else:
        _morton_decode_2d_i64(index, out_x, out_y, nbits)

    return out_x, out_y
