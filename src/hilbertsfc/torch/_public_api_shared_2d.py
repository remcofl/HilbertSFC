"""Shared 2D torch public API plumbing."""

import warnings
from collections.abc import Callable
from typing import Any, cast

import torch

from .._nbits import MAX_NBITS_2D, validate_nbits_2d
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


def encode_2d_api(
    x: torch.Tensor,
    y: torch.Tensor,
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

    if x.device != y.device:
        raise ValueError(
            f"x and y must be on the same device; got {x.device=}, {y.device=}"
        )
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape=}, {y.shape=}"
        )

    require_int_tensor(x, "x")
    require_int_tensor(y, "y")

    max_coord_nbits = max(
        effective_bits_torch_dtype(x.dtype), effective_bits_torch_dtype(y.dtype)
    )
    if nbits is None:
        nbits = max_coord_nbits
        if nbits > MAX_NBITS_2D:
            warnings.warn(
                f"The maximum effective bits of the coordinate dtype is {nbits}, "
                f"which exceeds the algorithm maximum of {MAX_NBITS_2D}. "
                f"Using nbits={MAX_NBITS_2D} instead. This means that excess bits in the input coordinates "
                f"will be ignored. To silence this warning, explicitly set nbits<={MAX_NBITS_2D}.",
                UserWarning,
                stacklevel=2,
            )
            nbits = MAX_NBITS_2D
    else:
        validate_nbits_2d(nbits)
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in coordinate dtypes; "
                f"got {x.dtype=} with {effective_bits_torch_dtype(x.dtype)} effective bits, "
                f"{y.dtype=} with {effective_bits_torch_dtype(y.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}."
            )

    out_provided = out is not None
    if out is None:
        prefer_uint = is_uint_torch_dtype(x.dtype) and is_uint_torch_dtype(y.dtype)
        out = torch.empty(
            x.shape,
            dtype=choose_index_torch_dtype(
                nbits=nbits, dims=2, prefer_unsigned=prefer_uint
            ),
            device=x.device,
        )
    else:
        if out.device != x.device:
            raise ValueError(f"out must be on {x.device}; got {out.device=}")
        if out.shape != x.shape:
            raise ValueError(f"out must have shape {x.shape}; got {out.shape=}")
        require_int_tensor(out, "out")

        max_index_nbits = max_nbits_for_torch_index_dtype(out.dtype, dims=2)
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
        out_np = int_tensor_to_numpy_view(out, "out")

        numba_kernel = get_numba()

        if x.ndim == 0:
            out_np[...] = numba_kernel(x_np, y_np, nbits=nbits)
            return out

        numba_kernel(
            x_np,
            y_np,
            nbits=nbits,
            out=out_np,
            parallel=resolve_cpu_parallel(cpu_parallel, x.numel()),
        )
        return out

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = x.is_contiguous() and y.is_contiguous() and out.is_contiguous()
        contig_details = (
            f"{x.is_contiguous()=}, {y.is_contiguous()=}, {out.is_contiguous()=}"
            if out_provided
            else f"{x.is_contiguous()=}, {y.is_contiguous()=}"
        )

        def _call() -> None:
            # Do not pass a dtype-view alias of `out` into Triton.
            # When in compiled region inductor can hit an internal assertion when a Triton
            # wrapper mutates via `out.view(int64)` and the graph returns the base `out`.
            get_triton()(x, y, nbits=nbits, out=out, triton_tuning=triton_tuning)

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out

    # CPU - auto with compiling, or explicit torch backend
    # Accelerator - auto with non-CUDA or with CUDA not all contiguous,
    #               or explicit torch backend

    # Not strictly needed when is_compiling with inductor.
    x_i = int_tensor_to_signed_view(x, "x")
    y_i = int_tensor_to_signed_view(y, "y")
    out_i = int_tensor_to_signed_view(out, "out")

    torch_kernel(
        x_i,
        y_i,
        nbits=nbits,
        out=out_i,
    )
    return out


def decode_2d_api(
    index: torch.Tensor,
    *,
    nbits: int | None,
    out_x: torch.Tensor | None,
    out_y: torch.Tensor | None,
    cpu_parallel: bool | None,
    cpu_backend: CPUBackend,
    gpu_backend: GPUBackend,
    triton_tuning: TritonTuningMode,
    torch_kernel: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    get_numba: Callable[[], Callable[..., Any]],
    get_triton: Callable[[], Callable[..., Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_backend = validate_cpu_backend(cpu_backend)
    gpu_backend = validate_gpu_backend(gpu_backend)
    triton_tuning = validate_triton_tuning_mode(triton_tuning)

    require_int_tensor(index, "index")
    max_index_nbits = max_nbits_for_torch_index_dtype(index.dtype, dims=2)
    if nbits is None:
        nbits = max_index_nbits
    else:
        validate_nbits_2d(nbits)
        if nbits > max_index_nbits:
            raise ValueError(
                f"nbits={nbits} exceeds the effective bits of the index dtype; "
                f"got {index.dtype=} which supports up to {max_index_nbits} bits for 2D coordinates. "
                f"max nbits is {max_index_nbits}."
            )

    if (out_x is None) != (out_y is None):
        raise ValueError("out_x and out_y must be provided together")

    out_provided = out_x is not None and out_y is not None
    if not out_provided:
        coord_dtype = choose_coord_torch_dtype(
            nbits=nbits,
            prefer_unsigned=is_uint_torch_dtype(index.dtype),
        )

        out_x = torch.empty(index.shape, dtype=coord_dtype, device=index.device)
        out_y = torch.empty(index.shape, dtype=coord_dtype, device=index.device)
    else:
        out_x = cast(torch.Tensor, out_x)
        out_y = cast(torch.Tensor, out_y)

        if out_x.device != index.device or out_y.device != index.device:
            raise ValueError(
                "out_x and out_y must be on the same device as index; "
                f"got {index.device=}, {out_x.device=}, {out_y.device=}"
            )
        if out_x.shape != index.shape or out_y.shape != index.shape:
            raise ValueError(
                "out_x and out_y must have the same shape as index; "
                f"got {index.shape=}, {out_x.shape=}, {out_y.shape=}"
            )

        require_int_tensor(out_x, "out_x")
        require_int_tensor(out_y, "out_y")

        max_coord_nbits = min(
            effective_bits_torch_dtype(out_x.dtype),
            effective_bits_torch_dtype(out_y.dtype),
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in out_x/out_y dtypes; "
                f"got {out_x.dtype=} with {effective_bits_torch_dtype(out_x.dtype)} effective bits, "
                f"{out_y.dtype=} with {effective_bits_torch_dtype(out_y.dtype)} effective bits; "
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

        numba_kernel = get_numba()

        if index.ndim == 0:
            x_i, y_i = numba_kernel(index_np, nbits=nbits)
            out_x_np[...] = x_i
            out_y_np[...] = y_i
            return out_x, out_y

        numba_kernel(
            index_np,
            nbits=nbits,
            out_x=out_x_np,
            out_y=out_y_np,
            parallel=resolve_cpu_parallel(cpu_parallel, index.numel()),
        )
        return out_x, out_y

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = (
            index.is_contiguous() and out_x.is_contiguous() and out_y.is_contiguous()
        )
        contig_details = (
            f"{index.is_contiguous()=}, {out_x.is_contiguous()=}, {out_y.is_contiguous()=}"
            if out_provided
            else f"{index.is_contiguous()=}"
        )

        def _call() -> None:
            get_triton()(
                index,
                nbits=nbits,
                out_x=out_x,
                out_y=out_y,
                triton_tuning=triton_tuning,
            )

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out_x, out_y

    # CPU - auto with compiling, or explicit torch backend
    # Accelerator - auto with non-CUDA or with CUDA not all contiguous,
    #               or explicit torch backend

    # Not strictly needed when is_compiling with inductor.
    index_i = int_tensor_to_signed_view(index, "index")
    out_x_i = int_tensor_to_signed_view(out_x, "out_x")
    out_y_i = int_tensor_to_signed_view(out_y, "out_y")

    torch_kernel(
        index_i,
        nbits=nbits,
        out_x=out_x_i,
        out_y=out_y_i,
    )
    return out_x, out_y
