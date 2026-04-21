import pytest


def _torch_pair():
    torch = pytest.importorskip("torch")
    htorch = pytest.importorskip("hilbertsfc.torch")
    h2 = __import__("hilbertsfc.torch.hilbert2d", fromlist=["*"])
    return torch, htorch, h2


@pytest.mark.torch
def test_encode_2d_requires_same_device() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.int32)
    y = torch.empty((2,), dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="must be on the same device"):
        htorch.hilbert_encode_2d(
            x, y, nbits=4, cpu_backend="torch", gpu_backend="torch"
        )


@pytest.mark.torch
def test_encode_2d_requires_same_shape() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.int32)
    y = torch.tensor([1], dtype=torch.int32)

    with pytest.raises(ValueError, match="must have the same shape"):
        htorch.hilbert_encode_2d(
            x, y, nbits=4, cpu_backend="torch", gpu_backend="torch"
        )


@pytest.mark.torch
def test_encode_2d_requires_integer_inputs() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([1, 2], dtype=torch.int32)

    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        htorch.hilbert_encode_2d(
            x, y, nbits=4, cpu_backend="torch", gpu_backend="torch"
        )


@pytest.mark.torch
def test_encode_2d_invalid_triton_tuning_raises() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.int32)
    y = torch.tensor([3, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="triton_tuning must be one of"):
        htorch.hilbert_encode_2d(
            x,
            y,
            nbits=4,
            cpu_backend="torch",
            gpu_backend="torch",
            triton_tuning="bad",  # type: ignore[arg-type]
        )


@pytest.mark.torch
def test_encode_2d_warns_when_default_nbits_exceeds_algorithm_max() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.int64)
    y = torch.tensor([3, 4], dtype=torch.int64)

    with pytest.warns(UserWarning, match="exceeds the algorithm maximum"):
        out = htorch.hilbert_encode_2d(x, y, cpu_backend="torch", gpu_backend="torch")

    assert out.shape == x.shape


@pytest.mark.torch
def test_encode_2d_nbits_must_fit_input_dtype() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.uint8)
    y = torch.tensor([3, 4], dtype=torch.uint8)

    with pytest.raises(ValueError, match="does not fit in coordinate dtypes"):
        htorch.hilbert_encode_2d(
            x, y, nbits=9, cpu_backend="torch", gpu_backend="torch"
        )


@pytest.mark.torch
def test_encode_2d_out_validation_errors() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.int16)
    y = torch.tensor([3, 4], dtype=torch.int16)

    out_meta = torch.empty((2,), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="out must be on"):
        htorch.hilbert_encode_2d(
            x, y, nbits=4, out=out_meta, cpu_backend="torch", gpu_backend="torch"
        )

    out_bad_shape = torch.empty((3,), dtype=torch.int32)
    with pytest.raises(ValueError, match="out must have shape"):
        htorch.hilbert_encode_2d(
            x, y, nbits=4, out=out_bad_shape, cpu_backend="torch", gpu_backend="torch"
        )

    out_float = torch.empty((2,), dtype=torch.float32)
    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        htorch.hilbert_encode_2d(
            x, y, nbits=4, out=out_float, cpu_backend="torch", gpu_backend="torch"
        )


@pytest.mark.torch
def test_encode_2d_out_dtype_too_narrow_raises() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.int32)
    y = torch.tensor([3, 4], dtype=torch.int32)
    out = torch.empty((2,), dtype=torch.uint8)

    with pytest.raises(ValueError, match="does not fit in out dtype"):
        htorch.hilbert_encode_2d(
            x, y, nbits=8, out=out, cpu_backend="torch", gpu_backend="torch"
        )


@pytest.mark.torch
def test_encode_2d_triton_requires_cuda_for_accelerators() -> None:
    torch, htorch, _ = _torch_pair()

    x = torch.empty((2,), dtype=torch.int32, device="meta")
    y = torch.empty((2,), dtype=torch.int32, device="meta")

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        htorch.hilbert_encode_2d(x, y, nbits=4, gpu_backend="triton")


@pytest.mark.torch
def test_encode_2d_auto_backend_while_compiling_avoids_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, htorch, h2 = _torch_pair()

    x = torch.tensor([1, 2], dtype=torch.int32)
    y = torch.tensor([3, 4], dtype=torch.int32)
    htorch.precache_compile_luts(device=x.device, op="hilbert_encode_2d")

    def _boom():
        raise AssertionError("numba getter should not be called while compiling")

    monkeypatch.setattr(h2, "get_hilbert_encode_2d_numba", _boom)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    out = htorch.hilbert_encode_2d(
        x, y, nbits=4, cpu_backend="auto", gpu_backend="torch"
    )
    assert out.shape == x.shape


@pytest.mark.torch
def test_decode_2d_requires_integer_index() -> None:
    torch, htorch, _ = _torch_pair()

    index = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        htorch.hilbert_decode_2d(
            index, nbits=4, cpu_backend="torch", gpu_backend="torch"
        )


@pytest.mark.torch
def test_decode_2d_invalid_cpu_backend_raises() -> None:
    torch, htorch, _ = _torch_pair()

    index = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="cpu_backend must be one of"):
        htorch.hilbert_decode_2d(index, nbits=4, cpu_backend="bad")  # type: ignore[arg-type]


@pytest.mark.torch
def test_decode_2d_out_dtype_must_be_integer() -> None:
    torch, htorch, _ = _torch_pair()

    index = torch.tensor([1, 2], dtype=torch.int16)
    out_x = torch.empty((2,), dtype=torch.float32)
    out_y = torch.empty((2,), dtype=torch.int16)

    with pytest.raises(TypeError, match="must be an integer torch.Tensor"):
        htorch.hilbert_decode_2d(
            index,
            nbits=4,
            out_x=out_x,
            out_y=out_y,
            cpu_backend="torch",
            gpu_backend="torch",
        )


@pytest.mark.torch
def test_decode_2d_triton_requires_cuda_for_accelerators() -> None:
    torch, htorch, _ = _torch_pair()

    index = torch.empty((2,), dtype=torch.int32, device="meta")

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        htorch.hilbert_decode_2d(index, nbits=4, gpu_backend="triton")


@pytest.mark.torch
def test_decode_2d_auto_backend_while_compiling_avoids_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, htorch, h2 = _torch_pair()

    index = torch.tensor([1, 2], dtype=torch.int32)
    htorch.precache_compile_luts(device=index.device, op="hilbert_decode_2d")

    def _boom():
        raise AssertionError("numba getter should not be called while compiling")

    monkeypatch.setattr(h2, "get_hilbert_decode_2d_numba", _boom)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    out_x, out_y = htorch.hilbert_decode_2d(
        index, nbits=4, cpu_backend="auto", gpu_backend="torch"
    )
    assert out_x.shape == index.shape
    assert out_y.shape == index.shape
