# ruff: noqa: N803, N806
"""Triton 2D Hilbert encoder (4-bit state-aware LUT)."""

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_2d
from hilbertsfc.torch._luts import TorchCacheMode, lut_2d4b_sb_sq_i16

from ._tuning import (
    TritonTuningMode,
    autotune_key_for_elements,
    triton_autotune_configs,
    validate_tuning_mode,
)


@triton.jit
def hilbert_encode_2d_4bit_sq(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    AUTOTUNE_KEY: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Encode 2D coordinates (x, y) to a Hilbert index."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x3
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 4
    COORD_MASK: tl.constexpr = (1 << NBITS) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]
    OUT_DTYPE: tl.constexpr = out_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    # Optional LUT preload.
    if SHMEM_LUT:
        lut_local = tl.load(lut_ptr + tl.arange(0, 1024), eviction_policy="evict_last")

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    if not SHMEM_LUT:
        # 8 bit tile packing (x and y) can result in negative values when using int8 inputs,
        # so we convert to uint8.
        if x.dtype == tl.int8:
            x = x.to(tl.uint8)
        if y.dtype == tl.int8:
            y = y.to(tl.uint8)

    if DROP_BITS > 0:
        x = x & COORD_MASK
        y = y & COORD_MASK

    idx = tl.zeros([BLOCK_SIZE], dtype=OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -4):
        b_x = (x >> bit) & 0xF
        b_y = (y >> bit) & 0xF
        b = (b_x << 4) | b_y

        lut_idx = state | b

        if SHMEM_LUT:
            sq = tl.gather(lut_local, lut_idx, axis=0)
        else:
            sq = tl.load(lut_ptr + lut_idx, eviction_policy="evict_last")

        # Keep in int32 for masks/shifts; this is faster than narrow integer ops.
        sq = sq.to(tl.int32)

        q = sq & 0xFF
        idx |= q.to(OUT_DTYPE) << (bit << 1)
        state = sq & 0x300

    tl.store(out_ptr + offsets, idx, mask=mask)


hilbert_encode_2d_4bit_sq_autotuned = triton.autotune(
    configs=triton_autotune_configs(),
    key=["AUTOTUNE_KEY", "NBITS", "SHMEM_LUT"],
)(hilbert_encode_2d_4bit_sq)


def _choose_launch_config(n_elements: int, shmem_lut: bool) -> tuple[int, int]:
    """Choose a reasonable default without autotune.

    Returns
    -------
    block_size: int
    num_warps: int
    """
    if not shmem_lut:
        if n_elements <= 2 << 22:
            return 128, 2
        return 256, 1

    if n_elements <= 2 << 16:
        return 256, 4
    if n_elements <= 2 << 20:
        return 512, 8
    if n_elements <= 2 << 22:
        return 1024, 8
    return 2048, 8


def hilbert_encode_2d_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
    triton_tuning: TritonTuningMode = "heuristic",
) -> torch.Tensor:
    """Encode (x, y) to Hilbert indices.

    - Does not require x/y to be uint64; any integer dtype is accepted.
    - Returns minimal uint dtype that can hold the index (based on 2*nbits).
    """
    validate_nbits_2d(nbits)
    triton_tuning = validate_tuning_mode(triton_tuning)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    n_elements = out.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    lut = lut_2d4b_sb_sq_i16(device=x.device, cache=lut_cache)

    # Triton < 3.3.0 does not have tl.gather, fall back to LUT in global memory.
    # This is still performant due to caching.
    load_lut_into_shared_memory = True if hasattr(tl, "gather") else False
    if triton_tuning == "heuristic":
        block_size, num_warps = _choose_launch_config(
            n_elements,
            shmem_lut=load_lut_into_shared_memory,
        )
        hilbert_encode_2d_4bit_sq[grid](  # type: ignore[reportIndexIssue]
            x,
            y,
            out,
            n_elements,
            AUTOTUNE_KEY=0,
            lut_ptr=lut,
            BLOCK_SIZE=block_size,
            NBITS=nbits,
            SHMEM_LUT=load_lut_into_shared_memory,
            num_warps=num_warps,  # type: ignore[reportCallIssue]
        )
        return out

    autotune_key = autotune_key_for_elements(n_elements, tuning=triton_tuning)
    hilbert_encode_2d_4bit_sq_autotuned[grid](  # type: ignore[reportIndexIssue]
        x,
        y,
        out,
        n_elements,
        AUTOTUNE_KEY=autotune_key,
        lut_ptr=lut,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
    )

    return out
