import numpy as np
import pytest


def _torch_modules():
    torch = pytest.importorskip("torch")
    dtypes_mod = __import__("hilbertsfc.torch._dtypes_int", fromlist=["*"])
    tensor_mod = __import__("hilbertsfc.torch._tensor_int", fromlist=["*"])
    return torch, dtypes_mod, tensor_mod


@pytest.mark.torch
def test_numpy_torch_integer_dtype_roundtrip() -> None:
    torch, dtypes_mod, _ = _torch_modules()

    numpy_dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8]
    if getattr(torch, "uint16", None) is not None:
        numpy_dtypes.append(np.uint16)
    if getattr(torch, "uint32", None) is not None:
        numpy_dtypes.append(np.uint32)
    if getattr(torch, "uint64", None) is not None:
        numpy_dtypes.append(np.uint64)

    for np_dt in numpy_dtypes:
        torch_dt = dtypes_mod.numpy_to_torch_dtype_int(np_dt)
        back = dtypes_mod.torch_to_numpy_dtype_int(torch_dt)
        assert back == np.dtype(np_dt)


@pytest.mark.torch
def test_numpy_to_torch_dtype_int_rejects_non_integer() -> None:
    _, dtypes_mod, _ = _torch_modules()

    with pytest.raises(TypeError, match="unsupported integer dtype"):
        dtypes_mod.numpy_to_torch_dtype_int(np.float32)


@pytest.mark.torch
def test_torch_to_numpy_dtype_int_rejects_non_integer() -> None:
    torch, dtypes_mod, _ = _torch_modules()

    with pytest.raises(TypeError, match="unsupported integer torch dtype"):
        dtypes_mod.torch_to_numpy_dtype_int(torch.float32)


@pytest.mark.torch
def test_dtype_predicates() -> None:
    torch, dtypes_mod, _ = _torch_modules()

    assert dtypes_mod.is_int_torch_dtype(torch.int32)
    assert dtypes_mod.is_sint_torch_dtype(torch.int32)
    assert not dtypes_mod.is_uint_torch_dtype(torch.int32)

    assert dtypes_mod.is_int_torch_dtype(torch.uint8)
    assert dtypes_mod.is_uint_torch_dtype(torch.uint8)
    assert not dtypes_mod.is_sint_torch_dtype(torch.uint8)

    assert not dtypes_mod.is_int_torch_dtype(torch.float32)


@pytest.mark.torch
def test_require_int_tensor_rejects_float() -> None:
    torch, _, tensor_mod = _torch_modules()

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        tensor_mod.require_int_tensor(x, "x")


@pytest.mark.torch
def test_int_tensor_to_signed_view_identity_for_signed() -> None:
    torch, _, tensor_mod = _torch_modules()

    x = torch.tensor([-1, 2, 3], dtype=torch.int32)
    y = tensor_mod.int_tensor_to_signed_view(x, "x")

    assert y is x


@pytest.mark.torch
def test_int_tensor_to_signed_view_reinterprets_unsigned() -> None:
    torch, _, tensor_mod = _torch_modules()

    x = torch.tensor([0, 255], dtype=torch.uint8)
    y = tensor_mod.int_tensor_to_signed_view(x, "x")

    assert y.dtype == torch.int8
    assert y.tolist() == [0, -1]
    assert y.data_ptr() == x.data_ptr()


@pytest.mark.torch
def test_int_tensor_to_unsigned_view_identity_for_unsigned() -> None:
    torch, _, tensor_mod = _torch_modules()

    x = torch.tensor([1, 2, 3], dtype=torch.uint8)
    y = tensor_mod.int_tensor_to_unsigned_view(x, "x")

    assert y is x


@pytest.mark.torch
def test_int_tensor_to_unsigned_view_reinterprets_signed() -> None:
    torch, _, tensor_mod = _torch_modules()

    x = torch.tensor([0, -1], dtype=torch.int8)
    y = tensor_mod.int_tensor_to_unsigned_view(x, "x")

    assert y.dtype == torch.uint8
    assert y.tolist() == [0, 255]
    assert y.data_ptr() == x.data_ptr()


@pytest.mark.torch
def test_signed_unsigned_view_helpers_reject_non_integer() -> None:
    torch, _, tensor_mod = _torch_modules()

    x = torch.tensor([1.5], dtype=torch.float32)
    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        tensor_mod.int_tensor_to_signed_view(x, "x")
    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        tensor_mod.int_tensor_to_unsigned_view(x, "x")
