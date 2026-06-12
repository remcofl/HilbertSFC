import torch

from ._dtypes_int import is_int_torch_dtype, is_sint_torch_dtype, is_uint_torch_dtype


def require_int_tensor(x: torch.Tensor, name: str) -> None:
    if not is_int_torch_dtype(x.dtype):
        raise TypeError(f"{name} must be an integer torch.Tensor; got dtype={x.dtype}")


def reject_obvious_tensor_memory_overlap(
    output: torch.Tensor,
    output_name: str,
    *others: tuple[torch.Tensor, str],
) -> None:
    """Reject output tensors when overlap is cheaply and definitely detected.

    This is a conservative fast path for eager tensors. It catches identical
    data pointers and overlapping byte ranges for contiguous tensors, but it
    intentionally does not perform full strided overlap analysis.
    """

    # Dynamo cannot trace the overlap predicates, which return Python values.
    if torch.compiler.is_compiling():
        return

    for other, other_name in others:
        if _tensors_have_obvious_memory_overlap(output, other):
            raise ValueError(f"{output_name} must not overlap {other_name}")


def _tensors_have_obvious_memory_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True only when tensor memory overlap is cheap to prove."""

    if a.numel() == 0 or b.numel() == 0 or a.device != b.device:
        return False

    a_start = a.data_ptr()
    b_start = b.data_ptr()

    if a_start == b_start:
        return True

    if a.is_contiguous() and b.is_contiguous():
        a_end = a_start + a.numel() * a.element_size()
        b_end = b_start + b.numel() * b.element_size()
        return a_start < b_end and b_start < a_end

    return False


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
