import numpy as np
import pytest

from hilbertsfc._flatten import flatten_nocopy


class _NoCopyKwargArray(np.ndarray):
    """Array subclass that mimics NumPy<2 behavior for reshape(copy=...)."""

    def reshape(self, *shape, **kwargs):
        if "copy" in kwargs:
            raise TypeError("copy keyword unsupported")
        return np.ndarray.reshape(self, *shape, **kwargs)


class _NoCopyWithImplicitCopyArray(np.ndarray):
    """Array subclass forcing fallback path where reshape would copy."""

    def reshape(self, *shape, **kwargs):
        if "copy" in kwargs:
            raise TypeError("copy keyword unsupported")
        return np.array(np.ndarray.reshape(self, *shape, **kwargs), copy=True)


def test_flatten_nocopy_direct_1d_view() -> None:
    arr = np.arange(8, dtype=np.int32)
    out = flatten_nocopy(arr, "arr")

    assert out.shape == (8,)
    assert np.shares_memory(out, arr)


def test_flatten_nocopy_strict_false_returns_original_on_failure() -> None:
    arr = np.arange(12, dtype=np.int32).reshape(3, 4).T
    assert not arr.flags.c_contiguous

    out = flatten_nocopy(arr, "arr", order="C", strict=False)
    assert out is arr


def test_flatten_nocopy_strict_true_raises_with_helpful_message() -> None:
    arr = np.arange(12, dtype=np.int32).reshape(3, 4).T

    with pytest.raises(ValueError, match="must support a zero-copy 1D view"):
        flatten_nocopy(arr, "arr", order="C", strict=True)


def test_flatten_nocopy_numpy_lt2_typeerror_fallback_success() -> None:
    base = np.arange(10, dtype=np.int16)
    arr = base.view(_NoCopyKwargArray)

    out = flatten_nocopy(arr, "arr")
    assert out.shape == (10,)
    assert np.shares_memory(out, base)


def test_flatten_nocopy_numpy_lt2_typeerror_fallback_copy_rejected() -> None:
    base = np.arange(9, dtype=np.int16).reshape(3, 3)
    arr = base.view(_NoCopyWithImplicitCopyArray)

    with pytest.raises(ValueError, match="must support a zero-copy 1D view"):
        flatten_nocopy(arr, "arr", strict=True)


def test_flatten_nocopy_numpy_lt2_typeerror_fallback_copy_allowed() -> None:
    base = np.arange(9, dtype=np.int16).reshape(3, 3)
    arr = base.view(_NoCopyWithImplicitCopyArray)

    out = flatten_nocopy(arr, "arr", strict=False)
    assert out is arr


def test_flatten_nocopy_numpy_lt2_empty_array_returns_reshaped() -> None:
    base = np.array([], dtype=np.int16)
    arr = base.view(_NoCopyKwargArray)

    out = flatten_nocopy(arr, "arr")
    assert out.shape == (0,)
