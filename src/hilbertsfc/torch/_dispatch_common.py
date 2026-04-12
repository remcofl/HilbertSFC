"""Shared helpers for torch frontend dispatch modules."""

import warnings
from collections.abc import Callable
from typing import Literal

import torch

from .._dtype import (
    choose_sint_coord_dtype,
    choose_sint_index_dtype,
    choose_uint_coord_dtype,
    choose_uint_index_dtype,
)
from ._dtypes_int import (
    SIGNED_INT_TORCH_DTYPES,
    UNSIGNED_INT_TORCH_DTYPES,
    numpy_to_torch_dtype_int,
)

GPU_BACKENDS = ("auto", "triton", "torch")
type GPUBackend = Literal["auto", "triton", "torch"]
"""GPU backend options for `hilbertsfc.torch` functions."""

CPU_BACKENDS = ("auto", "numba", "torch")
type CPUBackend = Literal["auto", "numba", "torch"]
"""CPU backend options for `hilbertsfc.torch` functions."""

_UNSIGNED_INT_DTYPES = UNSIGNED_INT_TORCH_DTYPES
_SIGNED_INT_DTYPES = SIGNED_INT_TORCH_DTYPES


def effective_bits_torch_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.bool:
        raise TypeError("boolean tensors are not supported")
    if dtype in _UNSIGNED_INT_DTYPES:
        return dtype.itemsize * 8
    if dtype in _SIGNED_INT_DTYPES:
        return dtype.itemsize * 8 - 1
    raise TypeError(f"expected integer tensor dtype; got {dtype!r}")


def max_nbits_for_torch_index_dtype(dtype: torch.dtype, *, dims: int) -> int:
    """Max coordinate nbits such that Hilbert indices fit in `dtype`."""

    if dims <= 0:
        raise ValueError("dims must be positive")
    return effective_bits_torch_dtype(dtype) // dims


def choose_index_torch_dtype(
    *, nbits: int, dims: int, allow_unsigned: bool = True, prefer_unsigned: bool = False
) -> torch.dtype:
    """Choose return dtype for indices.

    If `prefer_unsigned` is False we still fall back to unsigned when the
    required bit width cannot fit in any signed dtype (e.g. 2D with nbits=32
    requires 64 index bits), unless `allow_unsigned=False`.
    """

    # 1. Try unsigned first if allowed and preferred
    if allow_unsigned and prefer_unsigned:
        np_dtype = choose_uint_index_dtype(nbits=nbits, dims=dims)
        try:
            return numpy_to_torch_dtype_int(np_dtype)
        except TypeError:
            pass  # unsigned not supported in torch

    # 2. Try signed
    try:
        return numpy_to_torch_dtype_int(choose_sint_index_dtype(nbits=nbits, dims=dims))
    except (ValueError, TypeError):
        # 3. Try unsigned if not tried before
        if allow_unsigned and not prefer_unsigned:
            np_dtype = choose_uint_index_dtype(nbits=nbits, dims=dims)
            try:
                return numpy_to_torch_dtype_int(np_dtype)
            except TypeError:
                pass

        raise ValueError(
            f"Required index bit width ({nbits=}, {dims=}) cannot be represented "
            "with the available integer dtypes."
        )


def choose_coord_torch_dtype(
    *, nbits: int, allow_unsigned: bool = True, prefer_unsigned: bool = False
) -> torch.dtype:
    """Choose return dtype for coordinates.

    If `prefer_unsigned` is False we still fall back to unsigned when the
    required bit width cannot fit in any signed dtype, unless
    `allow_unsigned=False`.
    """

    # 1. Try unsigned first if allowed and preferred
    if allow_unsigned and prefer_unsigned:
        np_dtype = choose_uint_coord_dtype(nbits=nbits)
        try:
            return numpy_to_torch_dtype_int(np_dtype)
        except TypeError:
            pass  # unsigned not supported in torch

    # 2. Try signed
    try:
        return numpy_to_torch_dtype_int(choose_sint_coord_dtype(nbits=nbits))
    except (ValueError, TypeError):
        # 3. Try unsigned if not tried before
        if allow_unsigned and not prefer_unsigned:
            np_dtype = choose_uint_coord_dtype(nbits=nbits)
            try:
                return numpy_to_torch_dtype_int(np_dtype)
            except TypeError:
                pass

        raise ValueError(
            f"Required coordinate bit width ({nbits=}) cannot be represented "
            "with the available integer dtypes."
        )


def resolve_cpu_parallel(cpu_parallel: bool | None, numel: int) -> bool:
    if cpu_parallel is not None:
        return cpu_parallel
    return torch.get_num_threads() > 1 and numel >= 1 << 14


def validate_cpu_backend(cpu_backend: str) -> CPUBackend:
    if cpu_backend not in CPU_BACKENDS:
        raise ValueError(
            f"cpu_backend must be one of: {CPU_BACKENDS}; got {cpu_backend!r}"
        )
    return cpu_backend


def validate_gpu_backend(gpu_backend: str) -> GPUBackend:
    if gpu_backend not in GPU_BACKENDS:
        raise ValueError(
            f"gpu_backend must be one of: {GPU_BACKENDS}; got {gpu_backend!r}"
        )
    return gpu_backend


def is_triton_available() -> bool:
    """Return True if Triton can be imported.

    This is a lightweight capability check used for dispatch decisions.
    """

    try:
        import triton  # type: ignore[reportMissingImports] # noqa: F401

        return True
    except Exception:
        return False


def attempt_run_triton(
    *,
    gpu_backend: GPUBackend,
    all_contiguous: bool,
    contiguity_details: str,
    call_triton: Callable[[], None],
) -> bool:
    """Try running a Triton kernel and handle fallback/warnings.

    Returns
    -------
    bool
        True if the Triton kernel ran successfully and the caller should return.
        False if the caller should fall back to the torch backend.
    """

    if not is_triton_available():
        if gpu_backend == "triton":
            raise RuntimeError(
                "gpu_backend='triton' requested, but Triton is unavailable"
            )
        return False

    if not all_contiguous:
        if gpu_backend == "triton":
            raise ValueError(
                "gpu_backend='triton' requires contiguous tensors; "
                f"got {contiguity_details}. "
                "Hint: call `.contiguous()` or use gpu_backend='torch'."
            )

        warnings.warn(
            "Triton backend requires contiguous tensors; falling back to gpu_backend='torch'. "
            "Hint: call `.contiguous()` on passed tensors to enable Triton, "
            "or set gpu_backend='torch' to silence this warning.",
            UserWarning,
            stacklevel=3,
        )
        return False

    try:
        call_triton()
        return True
    except Exception as e:
        if gpu_backend == "triton":
            raise RuntimeError(
                "gpu_backend='triton' requested, but Triton kernel failed at runtime. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        warnings.warn(
            "Triton kernel failed at runtime; falling back to gpu_backend='torch'. "
            "Hint: set gpu_backend='torch' to suppress this warning; "
            "or call this function with gpu_backend='triton' to get a more detailed "
            "error message.",
            UserWarning,
            stacklevel=3,
        )
        return False
