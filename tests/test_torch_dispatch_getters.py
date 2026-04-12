import sys
import types

import numpy as np
import pytest


def _mod():
    pytest.importorskip("torch")
    return __import__("hilbertsfc.torch._dispatch", fromlist=["*"])


@pytest.mark.torch
def test_get_hilbert_encode_decode_2d_numba_callables_work() -> None:
    mod = _mod()

    enc = mod.get_hilbert_encode_2d_numba()
    dec = mod.get_hilbert_decode_2d_numba()

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)

    idx = enc(x, y, nbits=5)
    xx, yy = dec(idx, nbits=5)

    np.testing.assert_array_equal(xx, x)
    np.testing.assert_array_equal(yy, y)


@pytest.mark.torch
def test_get_hilbert_encode_decode_3d_numba_callables_work() -> None:
    mod = _mod()

    enc = mod.get_hilbert_encode_3d_numba()
    dec = mod.get_hilbert_decode_3d_numba()

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    z = np.array([0, 1, 2], dtype=np.int32)

    idx = enc(x, y, z, nbits=5)
    xx, yy, zz = dec(idx, nbits=5)

    np.testing.assert_array_equal(xx, x)
    np.testing.assert_array_equal(yy, y)
    np.testing.assert_array_equal(zz, z)


@pytest.mark.torch
def test_triton_getters_use_module_symbol_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _mod()

    fake_2d_enc = types.ModuleType("hilbertsfc.torch._kernels.triton.hilbert2d_encode")
    fake_2d_dec = types.ModuleType("hilbertsfc.torch._kernels.triton.hilbert2d_decode")
    fake_3d_enc = types.ModuleType("hilbertsfc.torch._kernels.triton.hilbert3d_encode")
    fake_3d_dec = types.ModuleType("hilbertsfc.torch._kernels.triton.hilbert3d_decode")

    def f1(*args, **kwargs):
        return None

    def f2(*args, **kwargs):
        return None

    def f3(*args, **kwargs):
        return None

    def f4(*args, **kwargs):
        return None

    fake_2d_enc.hilbert_encode_2d_triton = f1  # type: ignore[reportAttributeAccessIssue]
    fake_2d_dec.hilbert_decode_2d_triton = f2  # type: ignore[reportAttributeAccessIssue]
    fake_3d_enc.hilbert_encode_3d_triton = f3  # type: ignore[reportAttributeAccessIssue]
    fake_3d_dec.hilbert_decode_3d_triton = f4  # type: ignore[reportAttributeAccessIssue]

    monkeypatch.setitem(sys.modules, fake_2d_enc.__name__, fake_2d_enc)
    monkeypatch.setitem(sys.modules, fake_2d_dec.__name__, fake_2d_dec)
    monkeypatch.setitem(sys.modules, fake_3d_enc.__name__, fake_3d_enc)
    monkeypatch.setitem(sys.modules, fake_3d_dec.__name__, fake_3d_dec)

    assert mod.get_hilbert_encode_2d_triton() is f1
    assert mod.get_hilbert_decode_2d_triton() is f2
    assert mod.get_hilbert_encode_3d_triton() is f3
    assert mod.get_hilbert_decode_3d_triton() is f4
