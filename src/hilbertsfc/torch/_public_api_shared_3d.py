"""Shared 3D torch public API plumbing."""

import warnings
from collections.abc import Callable
from typing import Any, cast

import torch

from .._nbits import MAX_NBITS_3D, validate_nbits_3d
from ._dispatch_common import (
    CPUBackend,
    GPUBackend,
    attempt_run_triton,
    choose_coord_torch_dtype,
    choose_index_torch_dtype,
    effective_bits_torch_dtype,
    max_nbits_for_torch_index_dtype,
    resolve_cpu_parallel,
    validate_cpu_backend,
    validate_gpu_backend,
)
from ._numpy_interop import int_tensor_to_numpy_view
from ._tensor_int import (
    int_tensor_to_signed_view,
    is_uint_torch_dtype,
    require_int_tensor,
)
from ._tuning_mode import TritonTuningMode, validate_triton_tuning_mode


def encode_3d_api(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nbits: int | None,
    out: torch.Tensor | None,
    cpu_parallel: bool | None,
    cpu_backend: CPUBackend,
    gpu_backend: GPUBackend,
    triton_tuning: TritonTuningMode,
    torch_kernel: Callable[..., torch.Tensor],
    get_numba: Callable[[], Callable[..., Any]],
    get_triton: Callable[[], Callable[..., Any]],
) -> torch.Tensor:
    cpu_backend = validate_cpu_backend(cpu_backend)
    gpu_backend = validate_gpu_backend(gpu_backend)
    triton_tuning = validate_triton_tuning_mode(triton_tuning)

    if x.device != y.device or x.device != z.device:
        raise ValueError(
            "x, y, z must be on the same device; "
            f"got {x.device=}, {y.device=}, {z.device=}"
        )
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"x, y, z must have the same shape; got {x.shape=}, {y.shape=}, {z.shape=}"
        )

    require_int_tensor(x, "x")
    require_int_tensor(y, "y")
    require_int_tensor(z, "z")

    max_coord_nbits = max(
        effective_bits_torch_dtype(x.dtype),
        effective_bits_torch_dtype(y.dtype),
        effective_bits_torch_dtype(z.dtype),
    )
    if nbits is None:
        nbits = max_coord_nbits
        if nbits > MAX_NBITS_3D:
            warnings.warn(
                f"The maximum effective bits of the coordinate dtype is {nbits}, "
                f"which exceeds the algorithm maximum of {MAX_NBITS_3D}. "
                f"Using nbits={MAX_NBITS_3D} instead. This means that excess bits in the input coordinates "
                f"will be ignored. To silence this warning, explicitly set nbits<={MAX_NBITS_3D}.",
                UserWarning,
                stacklevel=2,
            )
            nbits = MAX_NBITS_3D
    else:
        validate_nbits_3d(nbits)
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in coordinate dtypes; "
                f"got {x.dtype=} with {effective_bits_torch_dtype(x.dtype)} effective bits, "
                f"{y.dtype=} with {effective_bits_torch_dtype(y.dtype)} effective bits, "
                f"{z.dtype=} with {effective_bits_torch_dtype(z.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}."
            )

    out_provided = out is not None
    if out is None:
        prefer_uint = (
            is_uint_torch_dtype(x.dtype)
            and is_uint_torch_dtype(y.dtype)
            and is_uint_torch_dtype(z.dtype)
        )
        out = torch.empty(
            x.shape,
            dtype=choose_index_torch_dtype(
                nbits=nbits, dims=3, prefer_unsigned=prefer_uint
            ),
            device=x.device,
        )
    else:
        if out.device != x.device:
            raise ValueError(f"out must be on {x.device}; got {out.device=}")
        if out.shape != x.shape:
            raise ValueError(f"out must have shape {x.shape}; got {out.shape=}")
        require_int_tensor(out, "out")

        max_index_nbits = max_nbits_for_torch_index_dtype(out.dtype, dims=3)
        if nbits > max_index_nbits:
            raise ValueError(
                f"{nbits=} does not fit in out dtype; got {out.dtype=} "
                f"which supports up to nbits={max_index_nbits}."
            )

    is_cpu = x.device.type == "cpu"
    is_cuda = x.device.type == "cuda"
    is_compiling = torch.compiler.is_compiling()

    if (not is_cpu) and (not is_cuda) and gpu_backend == "triton":
        raise RuntimeError(
            f"gpu_backend='triton' requires CUDA tensors; got {x.device.type=}"
        )

    if is_cpu and (
        cpu_backend == "numba" or (cpu_backend == "auto" and not is_compiling)
    ):
        x_np = int_tensor_to_numpy_view(x, "x")
        y_np = int_tensor_to_numpy_view(y, "y")
        z_np = int_tensor_to_numpy_view(z, "z")
        out_np = int_tensor_to_numpy_view(out, "out")

        numba_kernel = get_numba()

        if x.ndim == 0:
            out_np[...] = numba_kernel(x_np, y_np, z_np, nbits=nbits)
            return out

        numba_kernel(
            x_np,
            y_np,
            z_np,
            nbits=nbits,
            out=out_np,
            parallel=resolve_cpu_parallel(cpu_parallel, x.numel()),
        )
        return out

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = (
            x.is_contiguous()
            and y.is_contiguous()
            and z.is_contiguous()
            and out.is_contiguous()
        )
        contig_details = (
            f"{x.is_contiguous()=}, {y.is_contiguous()=}, {z.is_contiguous()=}, {out.is_contiguous()=}"
            if out_provided
            else f"{x.is_contiguous()=}, {y.is_contiguous()=}, {z.is_contiguous()=}"
        )

        def _call() -> None:
            get_triton()(x, y, z, nbits=nbits, out=out, triton_tuning=triton_tuning)

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out

    x_i = int_tensor_to_signed_view(x, "x")
    y_i = int_tensor_to_signed_view(y, "y")
    z_i = int_tensor_to_signed_view(z, "z")
    out_i = int_tensor_to_signed_view(out, "out")

    torch_kernel(x_i, y_i, z_i, nbits=nbits, out=out_i)
    return out


def decode_3d_api(
    index: torch.Tensor,
    *,
    nbits: int | None,
    out_x: torch.Tensor | None,
    out_y: torch.Tensor | None,
    out_z: torch.Tensor | None,
    cpu_parallel: bool | None,
    cpu_backend: CPUBackend,
    gpu_backend: GPUBackend,
    triton_tuning: TritonTuningMode,
    torch_kernel: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    get_numba: Callable[[], Callable[..., Any]],
    get_triton: Callable[[], Callable[..., Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cpu_backend = validate_cpu_backend(cpu_backend)
    gpu_backend = validate_gpu_backend(gpu_backend)
    triton_tuning = validate_triton_tuning_mode(triton_tuning)

    require_int_tensor(index, "index")
    max_index_nbits = max_nbits_for_torch_index_dtype(index.dtype, dims=3)
    if nbits is None:
        nbits = max_index_nbits
    else:
        validate_nbits_3d(nbits)
        if nbits > max_index_nbits:
            raise ValueError(
                f"nbits={nbits} exceeds the effective bits of the index dtype; "
                f"got {index.dtype=} which supports up to {max_index_nbits} bits for 3D coordinates. "
                f"max nbits is {max_index_nbits}."
            )

    n_outs = (out_x is not None) + (out_y is not None) + (out_z is not None)
    if n_outs not in (0, 3):
        raise ValueError("out_x, out_y, out_z must be provided together")

    out_provided = n_outs == 3
    if not out_provided:
        coord_dtype = choose_coord_torch_dtype(
            nbits=nbits,
            prefer_unsigned=is_uint_torch_dtype(index.dtype),
        )

        out_x = torch.empty(index.shape, dtype=coord_dtype, device=index.device)
        out_y = torch.empty(index.shape, dtype=coord_dtype, device=index.device)
        out_z = torch.empty(index.shape, dtype=coord_dtype, device=index.device)
    else:
        out_x = cast(torch.Tensor, out_x)
        out_y = cast(torch.Tensor, out_y)
        out_z = cast(torch.Tensor, out_z)

        if (
            out_x.device != index.device
            or out_y.device != index.device
            or out_z.device != index.device
        ):
            raise ValueError(
                "out_x, out_y, out_z must be on the same device as index; "
                f"got {index.device=}, {out_x.device=}, {out_y.device=}, {out_z.device=}"
            )
        if (
            out_x.shape != index.shape
            or out_y.shape != index.shape
            or out_z.shape != index.shape
        ):
            raise ValueError(
                "out_x, out_y, out_z must have the same shape as index; "
                f"got {index.shape=}, {out_x.shape=}, {out_y.shape=}, {out_z.shape=}"
            )

        require_int_tensor(out_x, "out_x")
        require_int_tensor(out_y, "out_y")
        require_int_tensor(out_z, "out_z")

        max_coord_nbits = min(
            effective_bits_torch_dtype(out_x.dtype),
            effective_bits_torch_dtype(out_y.dtype),
            effective_bits_torch_dtype(out_z.dtype),
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in out_x/out_y/out_z dtypes; "
                f"got {out_x.dtype=} with {effective_bits_torch_dtype(out_x.dtype)} effective bits, "
                f"{out_y.dtype=} with {effective_bits_torch_dtype(out_y.dtype)} effective bits, "
                f"{out_z.dtype=} with {effective_bits_torch_dtype(out_z.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}"
            )

    is_cpu = index.device.type == "cpu"
    is_cuda = index.device.type == "cuda"
    is_compiling = torch.compiler.is_compiling()

    if (not is_cpu) and (not is_cuda) and gpu_backend == "triton":
        raise RuntimeError(
            f"gpu_backend='triton' requires CUDA tensors; got {index.device.type=}"
        )

    if is_cpu and (
        cpu_backend == "numba" or (cpu_backend == "auto" and not is_compiling)
    ):
        index_np = int_tensor_to_numpy_view(index, "index")
        out_x_np = int_tensor_to_numpy_view(out_x, "out_x")
        out_y_np = int_tensor_to_numpy_view(out_y, "out_y")
        out_z_np = int_tensor_to_numpy_view(out_z, "out_z")

        numba_kernel = get_numba()

        if index.ndim == 0:
            x_i, y_i, z_i = numba_kernel(index_np, nbits=nbits)
            out_x_np[...] = x_i
            out_y_np[...] = y_i
            out_z_np[...] = z_i
            return out_x, out_y, out_z

        numba_kernel(
            index_np,
            nbits=nbits,
            out_x=out_x_np,
            out_y=out_y_np,
            out_z=out_z_np,
            parallel=resolve_cpu_parallel(cpu_parallel, index.numel()),
        )
        return out_x, out_y, out_z

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = (
            index.is_contiguous()
            and out_x.is_contiguous()
            and out_y.is_contiguous()
            and out_z.is_contiguous()
        )
        contig_details = (
            f"{index.is_contiguous()=}, {out_x.is_contiguous()=}, {out_y.is_contiguous()=}, {out_z.is_contiguous()=}"
            if out_provided
            else f"{index.is_contiguous()=}"
        )

        def _call() -> None:
            get_triton()(
                index,
                nbits=nbits,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                triton_tuning=triton_tuning,
            )

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out_x, out_y, out_z

    index_i = int_tensor_to_signed_view(index, "index")
    out_x_i = int_tensor_to_signed_view(out_x, "out_x")
    out_y_i = int_tensor_to_signed_view(out_y, "out_y")
    out_z_i = int_tensor_to_signed_view(out_z, "out_z")

    torch_kernel(
        index_i,
        nbits=nbits,
        out_x=out_x_i,
        out_y=out_y_i,
        out_z=out_z_i,
    )
    return out_x, out_y, out_z
