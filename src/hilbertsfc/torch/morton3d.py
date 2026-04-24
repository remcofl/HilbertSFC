"""Torch 3D Morton API dispatch layer."""

import torch

from ._dispatch import (
    get_morton_decode_3d_numba,
    get_morton_decode_3d_triton,
    get_morton_encode_3d_numba,
    get_morton_encode_3d_triton,
)
from ._dispatch_common import CPUBackend, GPUBackend
from ._kernels.torch.morton3d_decode import morton_decode_3d_torch
from ._kernels.torch.morton3d_encode import morton_encode_3d_torch
from ._public_api_shared_3d import decode_3d_api, encode_3d_api
from ._tuning_mode import TritonTuningMode


def morton_encode_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nbits: int | None = None,
    out: torch.Tensor | None = None,
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> torch.Tensor:
    """Encode 3D integer coordinate tensors to Morton (Z-order) indices.

    API semantics for parameters, returns, and errors match
    [`hilbert_encode_3d`][hilbertsfc.torch.hilbert3d.hilbert_encode_3d],
    except that Morton kernels do not use lookup tables and therefore do not
    accept ``lut_cache``.
    """

    return encode_3d_api(
        x,
        y,
        z,
        nbits=nbits,
        out=out,
        cpu_parallel=cpu_parallel,
        cpu_backend=cpu_backend,
        gpu_backend=gpu_backend,
        triton_tuning=triton_tuning,
        torch_kernel=morton_encode_3d_torch,
        get_numba=get_morton_encode_3d_numba,
        get_triton=get_morton_encode_3d_triton,
    )


def morton_decode_3d(
    index: torch.Tensor,
    *,
    nbits: int | None = None,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_z: torch.Tensor | None = None,
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode Morton (Z-order) index tensors to 3D integer coordinates.

    API semantics for parameters, returns, and errors match
    [`hilbert_decode_3d`][hilbertsfc.torch.hilbert3d.hilbert_decode_3d],
    except that Morton kernels do not use lookup tables and therefore do not
    accept ``lut_cache``.
    """

    return decode_3d_api(
        index,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        cpu_parallel=cpu_parallel,
        cpu_backend=cpu_backend,
        gpu_backend=gpu_backend,
        triton_tuning=triton_tuning,
        torch_kernel=morton_decode_3d_torch,
        get_numba=get_morton_decode_3d_numba,
        get_triton=get_morton_decode_3d_triton,
    )
