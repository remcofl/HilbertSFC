"""Plain torch 3D Morton decode kernels."""

import torch

from hilbertsfc._nbits import validate_nbits_3d

_LOW_MASK_3D_INDEX_BITS: tuple[int, ...] = tuple(
    ((1 << (3 * n)) - 1) if n < 21 else -1 for n in range(22)
)


def _compact1by2_i32(x: torch.Tensor, nbits: int) -> torch.Tensor:
    x = x & 0x09249249
    if nbits > 1:
        x = (x ^ (x >> 2)) & 0x030C30C3
    if nbits > 2:
        x = (x ^ (x >> 4)) & 0x0300F00F
    if nbits > 3:
        x = (x ^ (x >> 8)) & 0x030000FF
    if nbits > 5:
        x = (x ^ (x >> 16)) & 0x000003FF
    return x


def _compact1by2_i64(x: torch.Tensor, nbits: int) -> torch.Tensor:
    x = x & 0x1249249249249249
    if nbits > 1:
        x = (x ^ (x >> 2)) & 0x10C30C30C30C30C3
    if nbits > 2:
        x = (x ^ (x >> 4)) & 0x100F00F00F00F00F
    if nbits > 3:
        x = (x ^ (x >> 8)) & 0x001F0000FF0000FF
    if nbits > 5:
        x = (x ^ (x >> 16)) & 0x001F00000000FFFF
    if nbits > 10:
        x = (x ^ (x >> 32)) & 0x00000000001FFFFF
    return x


def _morton_decode_3d_i32(
    index: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    out_z: torch.Tensor,
    nbits: int,
) -> None:
    if nbits < 10:
        index = index & _LOW_MASK_3D_INDEX_BITS[nbits]
    elif index.dtype.itemsize > 4:
        index = index & 0x3FFFFFFF

    if index.dtype.itemsize != 4:
        index = index.to(torch.int32)

    out_x.copy_(_compact1by2_i32(index, nbits))
    out_y.copy_(_compact1by2_i32(index >> 1, nbits))
    out_z.copy_(_compact1by2_i32(index >> 2, nbits))


def _morton_decode_3d_i64(
    index: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    out_z: torch.Tensor,
    nbits: int,
) -> None:
    if index.dtype.itemsize < 8:
        index = index.to(torch.int64)

    if nbits < 21:
        index = index & _LOW_MASK_3D_INDEX_BITS[nbits]

    out_x.copy_(_compact1by2_i64(index, nbits))
    out_y.copy_(_compact1by2_i64(index >> 1, nbits))
    out_z.copy_(_compact1by2_i64(index >> 2, nbits))


def morton_decode_3d_torch(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal plain-torch 3D Morton decoder."""

    validate_nbits_3d(nbits)

    if out_x is None:
        out_x = torch.empty_like(index, dtype=torch.int64)
    if out_y is None:
        out_y = torch.empty_like(index, dtype=torch.int64)
    if out_z is None:
        out_z = torch.empty_like(index, dtype=torch.int64)

    if nbits <= 10:
        _morton_decode_3d_i32(index, out_x, out_y, out_z, nbits)
    else:
        _morton_decode_3d_i64(index, out_x, out_y, out_z, nbits)

    return out_x, out_y, out_z
