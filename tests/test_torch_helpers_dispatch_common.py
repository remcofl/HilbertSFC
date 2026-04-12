import numpy as np
import pytest


def _torch_and_mod():
    torch = pytest.importorskip("torch")
    mod = __import__("hilbertsfc.torch._dispatch_common", fromlist=["*"])
    return torch, mod


@pytest.mark.torch
def test_effective_bits_torch_dtype_integer_and_errors() -> None:
    torch, mod = _torch_and_mod()

    assert mod.effective_bits_torch_dtype(torch.uint8) == 8
    assert mod.effective_bits_torch_dtype(torch.int8) == 7

    with pytest.raises(TypeError, match="boolean tensors are not supported"):
        mod.effective_bits_torch_dtype(torch.bool)
    with pytest.raises(TypeError, match="expected integer tensor dtype"):
        mod.effective_bits_torch_dtype(torch.float32)


@pytest.mark.torch
def test_max_nbits_for_torch_index_dtype_and_dims_validation() -> None:
    torch, mod = _torch_and_mod()

    assert mod.max_nbits_for_torch_index_dtype(torch.int32, dims=2) == 15

    with pytest.raises(ValueError, match="dims must be positive"):
        mod.max_nbits_for_torch_index_dtype(torch.int32, dims=0)


@pytest.mark.torch
def test_choose_index_torch_dtype_prefers_unsigned_when_requested() -> None:
    torch, mod = _torch_and_mod()

    out = mod.choose_index_torch_dtype(nbits=4, dims=2, prefer_unsigned=True)
    assert out == torch.uint8


@pytest.mark.torch
def test_choose_index_torch_dtype_signed_then_unsigned_fallback() -> None:
    torch, mod = _torch_and_mod()

    if getattr(torch, "uint64", None) is None:
        pytest.skip("torch.uint64 is unavailable in this torch build")

    out = mod.choose_index_torch_dtype(nbits=32, dims=2, prefer_unsigned=False)
    assert out == torch.uint64


@pytest.mark.torch
def test_choose_index_torch_dtype_disallow_unsigned_raises() -> None:
    _, mod = _torch_and_mod()

    with pytest.raises(ValueError, match="cannot be represented"):
        mod.choose_index_torch_dtype(
            nbits=32,
            dims=2,
            allow_unsigned=False,
            prefer_unsigned=False,
        )


@pytest.mark.torch
def test_choose_coord_torch_dtype_prefers_unsigned_when_requested() -> None:
    torch, mod = _torch_and_mod()

    out = mod.choose_coord_torch_dtype(nbits=8, prefer_unsigned=True)
    assert out == torch.uint8


@pytest.mark.torch
def test_choose_coord_torch_dtype_signed_then_unsigned_fallback() -> None:
    torch, mod = _torch_and_mod()

    if getattr(torch, "uint64", None) is None:
        pytest.skip("torch.uint64 is unavailable in this torch build")

    out = mod.choose_coord_torch_dtype(nbits=64, prefer_unsigned=False)
    assert out == torch.uint64


@pytest.mark.torch
def test_choose_coord_torch_dtype_disallow_unsigned_raises() -> None:
    _, mod = _torch_and_mod()

    with pytest.raises(ValueError, match="cannot be represented"):
        mod.choose_coord_torch_dtype(
            nbits=64,
            allow_unsigned=False,
            prefer_unsigned=False,
        )


@pytest.mark.torch
def test_choose_index_torch_dtype_prefer_unsigned_mapping_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, mod = _torch_and_mod()

    monkeypatch.setattr(mod, "choose_uint_index_dtype", lambda **kwargs: np.uint8)
    monkeypatch.setattr(mod, "choose_sint_index_dtype", lambda **kwargs: np.int8)

    def _fake_map(dtype):
        if np.dtype(dtype) == np.dtype(np.uint8):
            raise TypeError("unsigned unavailable")
        return torch.int8

    monkeypatch.setattr(mod, "numpy_to_torch_dtype_int", _fake_map)

    out = mod.choose_index_torch_dtype(nbits=2, dims=2, prefer_unsigned=True)
    assert out == torch.int8


@pytest.mark.torch
def test_choose_index_torch_dtype_signed_and_unsigned_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, mod = _torch_and_mod()

    monkeypatch.setattr(
        mod,
        "choose_sint_index_dtype",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("no signed")),
    )
    monkeypatch.setattr(mod, "choose_uint_index_dtype", lambda **kwargs: np.uint64)
    monkeypatch.setattr(
        mod,
        "numpy_to_torch_dtype_int",
        lambda dtype: (_ for _ in ()).throw(TypeError("no unsigned")),
    )

    with pytest.raises(ValueError, match="cannot be represented"):
        mod.choose_index_torch_dtype(nbits=32, dims=2, prefer_unsigned=False)


@pytest.mark.torch
def test_choose_coord_torch_dtype_prefer_unsigned_mapping_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, mod = _torch_and_mod()

    monkeypatch.setattr(mod, "choose_uint_coord_dtype", lambda **kwargs: np.uint8)
    monkeypatch.setattr(mod, "choose_sint_coord_dtype", lambda **kwargs: np.int8)

    def _fake_map(dtype):
        if np.dtype(dtype) == np.dtype(np.uint8):
            raise TypeError("unsigned unavailable")
        return torch.int8

    monkeypatch.setattr(mod, "numpy_to_torch_dtype_int", _fake_map)

    out = mod.choose_coord_torch_dtype(nbits=2, prefer_unsigned=True)
    assert out == torch.int8


@pytest.mark.torch
def test_choose_coord_torch_dtype_signed_and_unsigned_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, mod = _torch_and_mod()

    monkeypatch.setattr(
        mod,
        "choose_sint_coord_dtype",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("no signed")),
    )
    monkeypatch.setattr(mod, "choose_uint_coord_dtype", lambda **kwargs: np.uint64)
    monkeypatch.setattr(
        mod,
        "numpy_to_torch_dtype_int",
        lambda dtype: (_ for _ in ()).throw(TypeError("no unsigned")),
    )

    with pytest.raises(ValueError, match="cannot be represented"):
        mod.choose_coord_torch_dtype(nbits=64, prefer_unsigned=False)


@pytest.mark.torch
def test_resolve_cpu_parallel_explicit_override() -> None:
    _, mod = _torch_and_mod()

    assert mod.resolve_cpu_parallel(True, numel=1) is True
    assert mod.resolve_cpu_parallel(False, numel=10_000_000) is False


@pytest.mark.torch
def test_resolve_cpu_parallel_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    torch, mod = _torch_and_mod()

    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    assert mod.resolve_cpu_parallel(None, numel=(1 << 14) - 1) is False
    assert mod.resolve_cpu_parallel(None, numel=(1 << 14)) is True


@pytest.mark.torch
def test_validate_cpu_backend_values() -> None:
    _, mod = _torch_and_mod()

    assert mod.validate_cpu_backend("auto") == "auto"
    assert mod.validate_cpu_backend("numba") == "numba"
    assert mod.validate_cpu_backend("torch") == "torch"

    with pytest.raises(ValueError, match="cpu_backend must be one of"):
        mod.validate_cpu_backend("bad")


@pytest.mark.torch
def test_validate_gpu_backend_values() -> None:
    _, mod = _torch_and_mod()

    assert mod.validate_gpu_backend("auto") == "auto"
    assert mod.validate_gpu_backend("triton") == "triton"
    assert mod.validate_gpu_backend("torch") == "torch"

    with pytest.raises(ValueError, match="gpu_backend must be one of"):
        mod.validate_gpu_backend("bad")


@pytest.mark.torch
def test_is_triton_available_returns_bool() -> None:
    _, mod = _torch_and_mod()

    assert isinstance(mod.is_triton_available(), bool)


@pytest.mark.torch
def test_is_triton_available_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, mod = _torch_and_mod()

    import builtins

    orig_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("missing triton")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert mod.is_triton_available() is False
