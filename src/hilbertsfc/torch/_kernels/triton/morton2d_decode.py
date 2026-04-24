# ruff: noqa: N803, N806
"""Triton 2D Morton decoder."""

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_2d

from ._tuning import (
    TritonTuningMode,
    autotune_key_for_elements,
    triton_autotune_configs,
    validate_tuning_mode,
)


@triton.jit
def _compact1by1_u32(x: tl.tensor, NBITS: tl.constexpr) -> tl.tensor:
    x = x & 0x55555555
    if NBITS > 1:
        x = (x | (x >> 1)) & 0x33333333
    if NBITS > 2:
        x = (x | (x >> 2)) & 0x0F0F0F0F
    if NBITS > 4:
        x = (x | (x >> 4)) & 0x00FF00FF
    if NBITS > 8:
        x = (x | (x >> 8)) & 0x0000FFFF
    return x


@triton.jit
def _compact1by1_u64(x: tl.tensor, NBITS: tl.constexpr) -> tl.tensor:
    x = x & 0x5555555555555555
    if NBITS > 1:
        x = (x | (x >> 1)) & 0x3333333333333333
    if NBITS > 2:
        x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    if NBITS > 4:
        x = (x | (x >> 4)) & 0x00FF00FF00FF00FF
    if NBITS > 8:
        x = (x | (x >> 8)) & 0x0000FFFF0000FFFF
    if NBITS > 16:
        x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    return x


@triton.jit
def morton_decode_2d_kernel(
    idx_ptr: tl.tensor,
    out_x_ptr: tl.tensor,
    out_y_ptr: tl.tensor,
    n_elements: int,
    AUTOTUNE_KEY: int,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
):
    IDX_MASK: tl.constexpr = (1 << (2 * NBITS)) - 1 if NBITS < 32 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = tl.load(idx_ptr + offsets, mask=mask, other=0)

    if NBITS <= 16:
        if NBITS < 16:
            idx = idx & IDX_MASK
        idx = idx.to(tl.uint32)
        x = _compact1by1_u32(idx, NBITS)
        y = _compact1by1_u32(idx >> 1, NBITS)
    else:
        if NBITS < 32:
            idx = idx & IDX_MASK
        idx = idx.to(tl.uint64)
        x = _compact1by1_u64(idx, NBITS)
        y = _compact1by1_u64(idx >> 1, NBITS)

    tl.store(out_x_ptr + offsets, x, mask=mask)
    tl.store(out_y_ptr + offsets, y, mask=mask)


morton_decode_2d_kernel_autotuned = triton.autotune(
    configs=triton_autotune_configs(),
    key=["AUTOTUNE_KEY", "NBITS"],
)(morton_decode_2d_kernel)


def _choose_launch_config(n_elements: int) -> tuple[int, int]:
    if n_elements <= 2 << 22:
        return 256, 4
    return 512, 4


def morton_decode_2d_triton(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    triton_tuning: TritonTuningMode = "heuristic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode Morton indices to (x, y) using Triton."""

    validate_nbits_2d(nbits)
    triton_tuning = validate_tuning_mode(triton_tuning)

    if out_x is None:
        out_x = torch.empty_like(index, dtype=torch.int64)
    if out_y is None:
        out_y = torch.empty_like(index, dtype=torch.int64)

    n_elements = index.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if triton_tuning == "heuristic":
        block_size, num_warps = _choose_launch_config(n_elements)
        morton_decode_2d_kernel[grid](  # type: ignore[reportIndexIssue]
            index,
            out_x,
            out_y,
            n_elements,
            AUTOTUNE_KEY=0,
            BLOCK_SIZE=block_size,
            NBITS=nbits,
            num_warps=num_warps,  # type: ignore[reportCallIssue]
        )
        return out_x, out_y

    autotune_key = autotune_key_for_elements(n_elements, tuning=triton_tuning)
    morton_decode_2d_kernel_autotuned[grid](  # type: ignore[reportIndexIssue]
        index,
        out_x,
        out_y,
        n_elements,
        AUTOTUNE_KEY=autotune_key,
        NBITS=nbits,
    )

    return out_x, out_y
