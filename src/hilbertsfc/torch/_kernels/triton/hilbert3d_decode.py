# ruff: noqa: N803, N806
"""Triton 3D Hilbert decoder (2-bit LUT)."""

import os

import torch
import triton  # type: ignore[reportMissingImports]
import triton.language as tl  # type: ignore[reportMissingImports]

from hilbertsfc._nbits import validate_nbits_3d
from hilbertsfc.torch._luts import TorchCacheMode, lut_3d2b_so_sb_i16

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]
_printed_all_timing_keys: set[tuple[str, int, int, bool]] = set()


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_elements", "NBITS", "SHMEM_LUT"],
)
@triton.jit
def hilbert_decode_3d_2bit_compacted_sb(
    idx_ptr: tl.tensor,
    out_x_ptr: tl.tensor,
    out_y_ptr: tl.tensor,
    out_z_ptr: tl.tensor,
    n_elements: int,
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


def hilbert_decode_3d_triton(
    index: torch.Tensor,
    *,
    nbits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_z: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode indices to (x, y, z) using Triton."""

    validate_nbits_3d(nbits)

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

    hilbert_decode_3d_2bit_compacted_sb[grid](  # type: ignore[reportIndexIssue]
        index,
        out_x,
        out_y,
        out_z,
        n_elements,
        lut,
        NBITS=nbits,
        SHMEM_LUT=load_lut_into_shared_memory,
    )

    if os.getenv("HILBERTSFC_TRITON_PRINT_ALL_AUTOTUNE_CONFIGS", "0") == "1":
        key = (
            str(index.device),
            int(n_elements),
            int(nbits),
            bool(load_lut_into_shared_memory),
        )
        if key not in _printed_all_timing_keys:
            timings = getattr(hilbert_decode_3d_2bit_compacted_sb, "configs_timings", None)
            if timings:
                _printed_all_timing_keys.add(key)
                print(
                    "[triton.autotune] all configs hilbert_decode_3d_2bit_compacted_sb "
                    f"device={index.device} n_elements={n_elements} nbits={nbits} "
                    f"shmem_lut={load_lut_into_shared_memory}"
                )

                def _median_ms(v: object) -> float:
                    if isinstance(v, (tuple, list)) and len(v) > 0:
                        return float(v[0])
                    return float(v)

                bytes_per_point = (
                    index.element_size()
                    + out_x.element_size()
                    + out_y.element_size()
                    + out_z.element_size()
                )
                ordered = sorted(timings.items(), key=lambda item: _median_ms(item[1]))
                best_ms = _median_ms(ordered[0][1])

                print("  median_us    delta      mpts/s      gb/s  config")

                for cfg, t in ordered:
                    t_ms = _median_ms(t)
                    delta_pct = 0.0 if best_ms <= 0 else ((t_ms / best_ms) - 1.0) * 100.0
                    t_s = t_ms * 1e-3
                    mpts_s = float("inf") if t_s <= 0 else (n_elements / t_s) / 1e6
                    gb_s = (
                        float("inf")
                        if t_s <= 0
                        else (n_elements * bytes_per_point) / t_s / 1e9
                    )
                    print(
                        f"  {t_ms * 1e3:9.2f}  {delta_pct:+7.2f}%  "
                        f"{mpts_s:10.2f}  {gb_s:8.2f}  {cfg}"
                    )

    return out_x, out_y, out_z
