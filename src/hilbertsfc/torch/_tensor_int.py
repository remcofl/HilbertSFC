import torch

from ._dtypes_int import is_int_torch_dtype, is_sint_torch_dtype, is_uint_torch_dtype


def require_int_tensor(x: torch.Tensor, name: str) -> None:
    if not is_int_torch_dtype(x.dtype):
        raise TypeError(f"{name} must be an integer torch.Tensor; got dtype={x.dtype}")


def int_tensor_to_signed_view(x: torch.Tensor, name: str) -> torch.Tensor:
    """Return a signed-integer dtype view of an integer tensor.

    This is a zero-copy reinterpretation (bitcast) intended for kernels that only
    support signed integer dtypes.

    Note: When the input exceeds the max value of the signed dtype,
    the output will be negative. This has no or minimal effect on most bit ops.

    Behavior:
    - Signed integer inputs are returned unchanged.
    - Unsigned integer inputs are reinterpreted as the matching signed dtype
      (e.g. `uint32 -> int32`) without copying.
    - Non-integer inputs raise a TypeError.
    """

    require_int_tensor(x, name)

    if is_sint_torch_dtype(x.dtype):
        return x

    bits = x.dtype.itemsize * 8

    return x.view(getattr(torch, f"int{bits}"))


def int_tensor_to_unsigned_view(x: torch.Tensor, name: str) -> torch.Tensor:
    """Return a unsigned-integer dtype view of an integer tensor.

    This is a zero-copy reinterpretation (bitcast) intended for kernels that only
    support unsigned integer dtypes.

    Note: When the input exceeds the max value of the unsigned dtype,
    the output will be large. This has no or minimal effect on most bit ops.

    Behavior:
    - Signed integer inputs are returned unchanged.
    - Unsigned integer inputs are reinterpreted as the matching unsigned dtype
      (e.g. `uint32 -> uint32`) without copying.
    - Non-integer inputs raise a TypeError.
    """

    require_int_tensor(x, name)

    if is_uint_torch_dtype(x.dtype):
        return x

    bits = x.dtype.itemsize * 8

    return x.view(getattr(torch, f"uint{bits}"))
