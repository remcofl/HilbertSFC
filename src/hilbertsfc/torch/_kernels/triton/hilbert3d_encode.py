# ruff: noqa: N803, N806
"""Triton 3D Hilbert encoder (2-bit LUT)."""

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_3d
from hilbertsfc.torch._luts import TorchCacheMode, lut_3d2b_sb_so_i16

from ._tuning import (
    TritonTuningMode,
    autotune_key_for_elements,
    triton_autotune_configs,
    validate_tuning_mode,
)


@triton.jit
def hilbert_encode_3d_2bit_so(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    z_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    AUTOTUNE_KEY: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Encode 3D coordinates (x, y, z) to a Hilbert index using a 2-bit LUT."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x1
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 2
    COORD_MASK: tl.constexpr = (1 << NBITS) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]
    OUT_DTYPE: tl.constexpr = out_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    if SHMEM_LUT:
        lut_idx = tl.arange(0, 2048)
        lut_mask = lut_idx < 1536
        lut_local = tl.load(
            lut_ptr + lut_idx, mask=lut_mask, eviction_policy="evict_last"
        )

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0)

    if DROP_BITS > 0:
        x = x & COORD_MASK
        y = y & COORD_MASK
        z = z & COORD_MASK

    idx = tl.zeros([BLOCK_SIZE], dtype=OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -2):
        b_x = (x >> bit) & 0x3
        b_y = (y >> bit) & 0x3
        b_z = (z >> bit) & 0x3
        b = (b_x << 4) | (b_y << 2) | b_z

        lut_idx = state | b

        if SHMEM_LUT:
            so = tl.gather(lut_local, lut_idx, axis=0)  # type: ignore[reportAttributeAccessIssue]
        else:
            so = tl.load(lut_ptr + lut_idx, cache_modifier=".ca")

        # LUT is stored as int16 in torch; widen for bitwise ops.
        so = so.to(tl.int32)

        o = so & 0x3F
        idx |= o.to(OUT_DTYPE) << (3 * bit)
        state = so & 0x7C0

    tl.store(out_ptr + offsets, idx, mask=mask)


hilbert_encode_3d_2bit_so_autotuned = triton.autotune(
    configs=triton_autotune_configs(),
    key=["AUTOTUNE_KEY", "NBITS", "SHMEM_LUT"],
)(hilbert_encode_3d_2bit_so)


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
    return 1024, 8


def hilbert_encode_3d_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nbits: int,
    out: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
    load_lut_into_shared_memory: bool = True,
    triton_tuning: TritonTuningMode = "heuristic",
) -> torch.Tensor:
    """Encode (x, y, z) to Hilbert indices using Triton."""

    validate_nbits_3d(nbits)
    triton_tuning = validate_tuning_mode(triton_tuning)

    if out is None:
        out = torch.empty_like(x, dtype=torch.int64)

    n_elements = out.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    lut = lut_3d2b_sb_so_i16(device=x.device, cache=lut_cache)

    # Triton < 3.3.0 does not have tl.gather, fall back to LUT in global memory.
    # This is still performant due to caching.
    load_lut_into_shared_memory = True if hasattr(tl, "gather") else False
    if triton_tuning == "heuristic":
        block_size, num_warps = _choose_launch_config(
            n_elements,
            shmem_lut=load_lut_into_shared_memory,
        )
        hilbert_encode_3d_2bit_so[grid](  # type: ignore[reportIndexIssue]
            x,
            y,
            z,
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
    hilbert_encode_3d_2bit_so_autotuned[grid](  # type: ignore[reportIndexIssue]
        x,
        y,
        z,
        out,
        n_elements,
        AUTOTUNE_KEY=autotune_key,
        lut_ptr=lut,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
    )

    return out
