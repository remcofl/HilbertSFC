# ruff: noqa: N803, N806
"""Triton 3D Morton encoder."""

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_3d

from ._tuning import (
    TritonTuningMode,
    autotune_key_for_elements,
    triton_autotune_configs,
    validate_tuning_mode,
)


@triton.jit
def _part1by2_u32(x: tl.tensor, NBITS: tl.constexpr) -> tl.tensor:
    if NBITS > 5:
        x = (x | (x << 16)) & 0x030000FF
    if NBITS > 3:
        x = (x | (x << 8)) & 0x0300F00F
    if NBITS > 2:
        x = (x | (x << 4)) & 0x030C30C3
    if NBITS > 1:
        x = (x | (x << 2)) & 0x09249249
    return x


@triton.jit
def _part1by2_u64(x: tl.tensor, NBITS: tl.constexpr) -> tl.tensor:
    if NBITS > 10:
        x = (x | (x << 32)) & 0x001F00000000FFFF
    if NBITS > 5:
        x = (x | (x << 16)) & 0x001F0000FF0000FF
    if NBITS > 3:
        x = (x | (x << 8)) & 0x100F00F00F00F00F
    if NBITS > 2:
        x = (x | (x << 4)) & 0x10C30C30C30C30C3
    if NBITS > 1:
        x = (x | (x << 2)) & 0x1249249249249249
    return x


@triton.jit
def morton_encode_3d_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    z_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    AUTOTUNE_KEY: int,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
):
    COORD_MASK: tl.constexpr = (1 << NBITS) - 1  # type: ignore[reportAssignmentType, reportOperatorIssue]
    OUT_DTYPE: tl.constexpr = out_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0) & COORD_MASK
    y = tl.load(y_ptr + offsets, mask=mask, other=0) & COORD_MASK
    z = tl.load(z_ptr + offsets, mask=mask, other=0) & COORD_MASK

    if NBITS <= 10:
        x = x.to(tl.uint32)
        y = y.to(tl.uint32)
        z = z.to(tl.uint32)
        idx = (
            _part1by2_u32(x, NBITS)
            | (_part1by2_u32(y, NBITS) << 1)
            | (_part1by2_u32(z, NBITS) << 2)
        )
        tl.store(out_ptr + offsets, idx, mask=mask)
    else:
        x = x.to(tl.uint64)
        y = y.to(tl.uint64)
        z = z.to(tl.uint64)
        idx = tl.zeros([BLOCK_SIZE], dtype=OUT_DTYPE)
        idx |= _part1by2_u64(x, NBITS).to(OUT_DTYPE)
        idx |= (_part1by2_u64(y, NBITS) << 1).to(OUT_DTYPE)
        idx |= (_part1by2_u64(z, NBITS) << 2).to(OUT_DTYPE)
        tl.store(out_ptr + offsets, idx, mask=mask)


morton_encode_3d_kernel_autotuned = triton.autotune(
    configs=triton_autotune_configs(),
    key=["AUTOTUNE_KEY", "NBITS"],
)(morton_encode_3d_kernel)


def _choose_launch_config(n_elements: int) -> tuple[int, int]:
    if n_elements <= 2 << 22:
        return 256, 4
    return 512, 4


def morton_encode_3d_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
    triton_tuning: TritonTuningMode = "heuristic",
) -> torch.Tensor:
    """Encode (x, y, z) to Morton indices using Triton."""

    validate_nbits_3d(nbits)
    triton_tuning = validate_tuning_mode(triton_tuning)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    n_elements = out.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if triton_tuning == "heuristic":
        block_size, num_warps = _choose_launch_config(n_elements)
        morton_encode_3d_kernel[grid](  # type: ignore[reportIndexIssue]
            x,
            y,
            z,
            out,
            n_elements,
            AUTOTUNE_KEY=0,
            BLOCK_SIZE=block_size,
            NBITS=nbits,
            num_warps=num_warps,  # type: ignore[reportCallIssue]
        )
        return out

    autotune_key = autotune_key_for_elements(n_elements, tuning=triton_tuning)
    morton_encode_3d_kernel_autotuned[grid](  # type: ignore[reportIndexIssue]
        x,
        y,
        z,
        out,
        n_elements,
        AUTOTUNE_KEY=autotune_key,
        NBITS=nbits,
    )

    return out
