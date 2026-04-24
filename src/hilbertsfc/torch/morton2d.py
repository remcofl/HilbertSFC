"""Torch 2D Morton API dispatch layer."""

import torch

from ._dispatch import (
    get_morton_decode_2d_numba,
    get_morton_decode_2d_triton,
    get_morton_encode_2d_numba,
    get_morton_encode_2d_triton,
)
from ._dispatch_common import CPUBackend, GPUBackend
from ._kernels.torch.morton2d_decode import morton_decode_2d_torch
from ._kernels.torch.morton2d_encode import morton_encode_2d_torch
from ._public_api_shared_2d import decode_2d_api, encode_2d_api
from ._tuning_mode import TritonTuningMode


def morton_encode_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    nbits: int | None = None,
    out: torch.Tensor | None = None,
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> torch.Tensor:
    """Encode 2D integer coordinate tensors to Morton (Z-order) indices.

    API semantics for parameters, returns, and errors match
    [`hilbert_encode_2d`][hilbertsfc.torch.hilbert2d.hilbert_encode_2d],
    except that Morton kernels do not use lookup tables and therefore do not
    accept ``lut_cache``.
    """

    return encode_2d_api(
        x,
        y,
        nbits=nbits,
        out=out,
        cpu_parallel=cpu_parallel,
        cpu_backend=cpu_backend,
        gpu_backend=gpu_backend,
        triton_tuning=triton_tuning,
        torch_kernel=morton_encode_2d_torch,
        get_numba=get_morton_encode_2d_numba,
        get_triton=get_morton_encode_2d_triton,
    )


def morton_decode_2d(
    index: torch.Tensor,
    *,
    nbits: int | None = None,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode Morton (Z-order) index tensors to 2D integer coordinates.

    API semantics for parameters, returns, and errors match
    [`hilbert_decode_2d`][hilbertsfc.torch.hilbert2d.hilbert_decode_2d],
    except that Morton kernels do not use lookup tables and therefore do not
    accept ``lut_cache``.
    """

    return decode_2d_api(
        index,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        cpu_parallel=cpu_parallel,
        cpu_backend=cpu_backend,
        gpu_backend=gpu_backend,
        triton_tuning=triton_tuning,
        torch_kernel=morton_decode_2d_torch,
        get_numba=get_morton_decode_2d_numba,
        get_triton=get_morton_decode_2d_triton,
    )
