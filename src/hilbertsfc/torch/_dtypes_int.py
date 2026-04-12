"""Integer dtype mapping helpers between NumPy and torch.

This module is intentionally integer-only. Torch unsigned integer dtypes beyond
uint8 may be unavailable depending on the installed torch build.
"""

import numpy as np
import torch

SIGNED_INT_TORCH_DTYPES: tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

UNSIGNED_INT_TORCH_DTYPES: tuple[torch.dtype, ...] = tuple(
    dt
    for dt in (
        torch.uint8,
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
    )
    if dt is not None
)
# np.dtype.num enables dynamo-friendly mapping
_NUMPY_INT_DTYPE_NUM_TO_TORCH_DTYPE: dict[int, torch.dtype] = {}
_TORCH_INT_DTYPE_TO_NUMPY_DTYPE: dict[torch.dtype, np.dtype] = {}


def _register_int_dtype_pair(
    numpy_dtype: np.dtype | type[np.generic], torch_dtype: torch.dtype
) -> None:
    dt = np.dtype(numpy_dtype)
    _NUMPY_INT_DTYPE_NUM_TO_TORCH_DTYPE[dt.num] = torch_dtype
    _TORCH_INT_DTYPE_TO_NUMPY_DTYPE[torch_dtype] = dt


_register_int_dtype_pair(np.int8, torch.int8)
_register_int_dtype_pair(np.int16, torch.int16)
_register_int_dtype_pair(np.int32, torch.int32)
_register_int_dtype_pair(np.int64, torch.int64)
_register_int_dtype_pair(np.uint8, torch.uint8)

# Unsigned dtypes beyond uint8 are optional in torch.
if (u16 := getattr(torch, "uint16", None)) is not None:
    _register_int_dtype_pair(np.uint16, u16)
if (u32 := getattr(torch, "uint32", None)) is not None:
    _register_int_dtype_pair(np.uint32, u32)
if (u64 := getattr(torch, "uint64", None)) is not None:
    _register_int_dtype_pair(np.uint64, u64)


def numpy_to_torch_dtype_int(dtype: np.dtype | type[np.generic]) -> torch.dtype:
    """Map a NumPy integer dtype to the corresponding torch integer dtype."""

    dt = np.dtype(dtype)
    out = _NUMPY_INT_DTYPE_NUM_TO_TORCH_DTYPE.get(dt.num)
    if out is None:
        raise TypeError(f"unsupported integer dtype {dt!r}")
    return out


def torch_to_numpy_dtype_int(dtype: torch.dtype) -> np.dtype:
    """Map a torch integer dtype to the corresponding NumPy integer dtype."""

    out = _TORCH_INT_DTYPE_TO_NUMPY_DTYPE.get(dtype)
    if out is None:
        raise TypeError(f"unsupported integer torch dtype {dtype}")
    return out


def is_int_torch_dtype(dtype: torch.dtype) -> bool:
    return dtype in SIGNED_INT_TORCH_DTYPES or dtype in UNSIGNED_INT_TORCH_DTYPES


def is_sint_torch_dtype(dtype: torch.dtype) -> bool:
    return dtype in SIGNED_INT_TORCH_DTYPES


def is_uint_torch_dtype(dtype: torch.dtype) -> bool:
    return dtype in UNSIGNED_INT_TORCH_DTYPES
