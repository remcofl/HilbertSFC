# ruff: noqa: N803, N806
"""Triton 3D Hilbert decoder (2-bit LUT)."""

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_3d
from hilbertsfc.torch._luts import TorchCacheMode, lut_3d2b_so_sb_i16

from ._tuning import (
    TritonTuningMode,
    autotune_key_for_elements,
    triton_autotune_configs,
    validate_tuning_mode,
)


@triton.jit
def hilbert_decode_3d_2bit_sb(
    idx_ptr: tl.tensor,
    out_x_ptr: tl.tensor,
    out_y_ptr: tl.tensor,
    out_z_ptr: tl.tensor,
    n_elements: int,
    AUTOTUNE_KEY: int,
    lut_ptr: tl.const,
    BLOCK_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    SHMEM_LUT: tl.constexpr,
):
    """Decode Hilbert indices to (x, y, z) using a 2-bit LUT."""

    START_BIT: tl.constexpr = (NBITS - 1) & ~0x1
    DROP_BITS: tl.constexpr = START_BIT - NBITS + 2
    IDX_MASK: tl.constexpr = (1 << (3 * NBITS)) - 1 if DROP_BITS > 0 else 0  # type: ignore[reportAssignmentType, reportOperatorIssue]

    X_OUT_DTYPE: tl.constexpr = out_x_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]
    Y_OUT_DTYPE: tl.constexpr = out_y_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]
    Z_OUT_DTYPE: tl.constexpr = out_z_ptr.dtype.element_ty  # type: ignore[reportAttributeAccessIssue]

    if SHMEM_LUT:
        lut_idx = tl.arange(0, 2048)
        lut_mask = lut_idx < 1536
        lut_local = tl.load(
            lut_ptr + lut_idx, mask=lut_mask, eviction_policy="evict_last"
        )

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = tl.load(idx_ptr + offsets, mask=mask, other=0)

    if DROP_BITS > 0:
        idx = idx & IDX_MASK

    x = tl.zeros([BLOCK_SIZE], dtype=X_OUT_DTYPE)
    y = tl.zeros([BLOCK_SIZE], dtype=Y_OUT_DTYPE)
    z = tl.zeros([BLOCK_SIZE], dtype=Z_OUT_DTYPE)
    state = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for bit in tl.static_range(START_BIT, -1, -2):
        o = (idx >> (3 * bit)) & 0x3F

        lut_idx = state | o

        if SHMEM_LUT:
            sb = tl.gather(lut_local, lut_idx, axis=0)  # type: ignore[reportAttributeAccessIssue]
        else:
            sb = tl.load(lut_ptr + lut_idx, eviction_policy="evict_last")

        # LUT is stored as int16 in torch; widen for bitwise ops.
        sb = sb.to(tl.int32)

        b_x = (sb & 0x30) >> 4
        b_y = (sb & 0x0C) >> 2
        b_z = sb & 0x03

        x |= b_x.to(X_OUT_DTYPE) << bit
        y |= b_y.to(Y_OUT_DTYPE) << bit
        z |= b_z.to(Z_OUT_DTYPE) << bit

        state = sb & 0x7C0

    tl.store(out_x_ptr + offsets, x, mask=mask)
    tl.store(out_y_ptr + offsets, y, mask=mask)
    tl.store(out_z_ptr + offsets, z, mask=mask)


hilbert_decode_3d_2bit_sb_autotuned = triton.autotune(
    configs=triton_autotune_configs(),
    key=["AUTOTUNE_KEY", "NBITS", "SHMEM_LUT"],
)(hilbert_decode_3d_2bit_sb)


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


def hilbert_decode_3d_triton(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_z: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
    triton_tuning: TritonTuningMode = "heuristic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode indices to (x, y, z) using Triton."""

    validate_nbits_3d(nbits)
    triton_tuning = validate_tuning_mode(triton_tuning)

    if out_x is None:
        out_x = torch.empty_like(index, dtype=torch.int64)
    if out_y is None:
        out_y = torch.empty_like(index, dtype=torch.int64)
    if out_z is None:
        out_z = torch.empty_like(index, dtype=torch.int64)

    n_elements = index.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    lut = lut_3d2b_so_sb_i16(device=index.device, cache=lut_cache)

    # Triton < 3.3.0 does not have tl.gather, fall back to LUT in global memory.
    # This is still performant due to caching.
    load_lut_into_shared_memory = True if hasattr(tl, "gather") else False
    if triton_tuning == "heuristic":
        block_size, num_warps = _choose_launch_config(
            n_elements,
            shmem_lut=load_lut_into_shared_memory,
        )
        hilbert_decode_3d_2bit_sb[grid](  # type: ignore[reportIndexIssue]
            index,
            out_x,
            out_y,
            out_z,
            n_elements,
            AUTOTUNE_KEY=0,
            lut_ptr=lut,
            BLOCK_SIZE=block_size,
            NBITS=nbits,
            SHMEM_LUT=load_lut_into_shared_memory,
            num_warps=num_warps,  # type: ignore[reportCallIssue]
        )
        return out_x, out_y, out_z

    autotune_key = autotune_key_for_elements(n_elements, tuning=triton_tuning)
    hilbert_decode_3d_2bit_sb_autotuned[grid](  # type: ignore[reportIndexIssue]
        index,
        out_x,
        out_y,
        out_z,
        n_elements,
        AUTOTUNE_KEY=autotune_key,
        lut_ptr=lut,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
    )

    return out_x, out_y, out_z
