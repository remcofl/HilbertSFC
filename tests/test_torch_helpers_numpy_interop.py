import warnings

import numpy as np
import pytest


def _torch_and_mod():
    torch = pytest.importorskip("torch")
    mod = __import__("hilbertsfc.torch._numpy_interop", fromlist=["*"])
    return torch, mod


@pytest.mark.torch
def test_int_tensor_to_numpy_view_is_zero_copy() -> None:
    torch, mod = _torch_and_mod()

    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    arr = mod.int_tensor_to_numpy_view(x, "x")

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32
    arr[1] = 77
    assert x[1].item() == 77


@pytest.mark.torch
def test_int_tensor_to_numpy_view_rejects_non_integer_tensor() -> None:
    torch, mod = _torch_and_mod()

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        mod.int_tensor_to_numpy_view(x, "x")


@pytest.mark.torch
def test_int_tensor_to_numpy_view_rejects_non_cpu_device() -> None:
    torch, mod = _torch_and_mod()

    x = torch.empty((2,), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="must be on CPU for NumPy fallback"):
        mod.int_tensor_to_numpy_view(x, "x")


@pytest.mark.torch
def test_int_tensor_to_numpy_view_wraps_numpy_failure() -> None:
    torch, mod = _torch_and_mod()

    indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
    values = torch.tensor([3, 4, 5], dtype=torch.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = torch.sparse_coo_tensor(
            indices,
            values,
            (2, 3),
            check_invariants=False,
        )

    user_warning_messages = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    if user_warning_messages:
        assert any(
            "Sparse invariant checks are implicitly disabled" in msg
            for msg in user_warning_messages
        )

    with pytest.raises(ValueError, match="must support a zero-copy NumPy view"):
        mod.int_tensor_to_numpy_view(x, "x")
