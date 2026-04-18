import numpy as np
import pytest

from hilbertsfc import hilbert_decode_2d as np_hilbert_decode_2d
from hilbertsfc import hilbert_decode_3d as np_hilbert_decode_3d
from hilbertsfc import hilbert_encode_2d as np_hilbert_encode_2d
from hilbertsfc import hilbert_encode_3d as np_hilbert_encode_3d


def _require_torch_cuda_triton():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    htorch = pytest.importorskip("hilbertsfc.torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    return torch, htorch


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
def test_triton_encode_decode_2d_matches_numpy(rng: np.random.Generator) -> None:
    torch, htorch = _require_torch_cuda_triton()

    nbits = 12
    n = 4096
    hi = 1 << nbits

    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)

    x = torch.from_numpy(x_np).to(device="cuda")
    y = torch.from_numpy(y_np).to(device="cuda")

    idx = htorch.hilbert_encode_2d(x, y, nbits=nbits, gpu_backend="triton")
    idx_np = idx.detach().cpu().numpy()

    ref_idx = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
    np.testing.assert_array_equal(idx_np, ref_idx)

    out_x, out_y = htorch.hilbert_decode_2d(idx, nbits=nbits, gpu_backend="triton")
    np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
def test_triton_auto_non_contiguous_falls_back_to_torch(
    rng: np.random.Generator,
) -> None:
    torch, htorch = _require_torch_cuda_triton()

    nbits = 9
    n = 2048
    hi = 1 << nbits

    x_base = torch.from_numpy(rng.integers(0, hi, size=(n, 2), dtype=np.int32)).to(
        device="cuda"
    )
    y_base = torch.from_numpy(rng.integers(0, hi, size=(n, 2), dtype=np.int32)).to(
        device="cuda"
    )

    x = x_base[:, 0]
    y = y_base[:, 0]

    assert not x.is_contiguous()
    assert not y.is_contiguous()

    with pytest.warns(UserWarning, match="falling back to gpu_backend='torch'"):
        idx_auto = htorch.hilbert_encode_2d(x, y, nbits=nbits, gpu_backend="auto")

    idx_torch = htorch.hilbert_encode_2d(x, y, nbits=nbits, gpu_backend="torch")
    np.testing.assert_array_equal(
        idx_auto.detach().cpu().numpy(),
        idx_torch.detach().cpu().numpy(),
    )


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
def test_triton_forced_non_contiguous_raises(rng: np.random.Generator) -> None:
    torch, htorch = _require_torch_cuda_triton()

    nbits = 6
    n = 512
    hi = 1 << nbits

    x_base = torch.from_numpy(rng.integers(0, hi, size=(n, 2), dtype=np.int32)).to(
        device="cuda"
    )
    y_base = torch.from_numpy(rng.integers(0, hi, size=(n, 2), dtype=np.int32)).to(
        device="cuda"
    )

    x = x_base[:, 0]
    y = y_base[:, 0]

    with pytest.raises(ValueError, match="requires contiguous tensors"):
        htorch.hilbert_encode_2d(x, y, nbits=nbits, gpu_backend="triton")


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.compile
@pytest.mark.no_graph_break
@pytest.mark.triton
@pytest.mark.parametrize(
    "op,dim,nbits,n",
    [
        pytest.param("encode", 2, 10, 4096, id="triton-compile-encode-2d"),
        pytest.param("decode", 2, 9, 4096, id="triton-compile-decode-2d"),
        pytest.param("encode", 3, 7, 2048, id="triton-compile-encode-3d"),
        pytest.param("decode", 3, 7, 2048, id="triton-compile-decode-3d"),
    ],
)
def test_cuda_compile_fullgraph_with_triton_matches_numpy(
    rng: np.random.Generator,
    op: str,
    dim: int,
    nbits: int,
    n: int,
) -> None:
    torch, htorch = _require_torch_cuda_triton()

    if dim == 2 and op == "encode":
        hi = 1 << nbits
        x_np = rng.integers(0, hi, size=n, dtype=np.int32)
        y_np = rng.integers(0, hi, size=n, dtype=np.int32)

        x = torch.from_numpy(x_np).to(device="cuda")
        y = torch.from_numpy(y_np).to(device="cuda")

        htorch.precache_compile_luts(device=x.device, op="hilbert_encode_2d")

        def fn(a, b):  # type: ignore[reportRedeclaration]
            return htorch.hilbert_encode_2d(a, b, nbits=nbits, gpu_backend="triton")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out = compiled_fn(x, y)

        ref = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    if dim == 2 and op == "decode":
        idx_np = rng.integers(0, 1 << (2 * nbits), size=n, dtype=np.uint32)
        idx = torch.from_numpy(idx_np).to(device="cuda")

        htorch.precache_compile_luts(device=idx.device, op="hilbert_decode_2d")

        def fn(index):
            return htorch.hilbert_decode_2d(index, nbits=nbits, gpu_backend="triton")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out_x, out_y = compiled_fn(idx)

        ref_x, ref_y = np_hilbert_decode_2d(idx_np, nbits=nbits)
        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), ref_x)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), ref_y)
        return

    if dim == 3 and op == "encode":
        hi = 1 << nbits
        x_np = rng.integers(0, hi, size=n, dtype=np.int32)
        y_np = rng.integers(0, hi, size=n, dtype=np.int32)
        z_np = rng.integers(0, hi, size=n, dtype=np.int32)

        x = torch.from_numpy(x_np).to(device="cuda")
        y = torch.from_numpy(y_np).to(device="cuda")
        z = torch.from_numpy(z_np).to(device="cuda")

        htorch.precache_compile_luts(device=x.device, op="hilbert_encode_3d")

        def fn(a, b, c):  # type: ignore[reportRedeclaration]
            return htorch.hilbert_encode_3d(
                a,
                b,
                c,
                nbits=nbits,
                gpu_backend="triton",
            )

        compiled_fn = torch.compile(fn, fullgraph=True)
        out = compiled_fn(x, y, z)

        ref = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    if dim == 3 and op == "decode":
        idx_np = rng.integers(0, 1 << (3 * nbits), size=n, dtype=np.uint32)
        idx = torch.from_numpy(idx_np).to(device="cuda")

        htorch.precache_compile_luts(device=idx.device, op="hilbert_decode_3d")

        def fn(index):
            return htorch.hilbert_decode_3d(index, nbits=nbits, gpu_backend="triton")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out_x, out_y, out_z = compiled_fn(idx)

        ref_x, ref_y, ref_z = np_hilbert_decode_3d(idx_np, nbits=nbits)
        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), ref_x)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), ref_y)
        np.testing.assert_array_equal(out_z.detach().cpu().numpy(), ref_z)
        return

    raise AssertionError(f"Unsupported combination: {op=}, {dim=}")


_CUDA_TRITON_2D_DTYPE_SWEEP_CASES = [
    pytest.param(np.uint8, 1, id="triton-compile-2d-encode-u8-n1"),
    pytest.param(np.uint8, 4, id="triton-compile-2d-encode-u8-n4"),
    pytest.param(np.int16, 8, id="triton-compile-2d-encode-i16-n8"),
    pytest.param(np.int16, 12, id="triton-compile-2d-encode-i16-n12"),
]


_CUDA_TRITON_2D_DECODE_INDEX_DTYPE_SWEEP_CASES = [
    pytest.param(np.uint8, 1, id="triton-compile-2d-decode-u8-n1"),
    pytest.param(np.uint8, 4, id="triton-compile-2d-decode-u8-n4"),
    pytest.param(np.uint16, 5, id="triton-compile-2d-decode-u16-n5"),
    pytest.param(np.uint16, 8, id="triton-compile-2d-decode-u16-n8"),
]


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.compile
@pytest.mark.no_graph_break
@pytest.mark.triton
@pytest.mark.parametrize("dtype,nbits", _CUDA_TRITON_2D_DTYPE_SWEEP_CASES)
def test_cuda_compile_fullgraph_with_triton_2d_encode_small_dtypes_and_nbits(
    rng: np.random.Generator,
    dtype: np.dtype,
    nbits: int,
) -> None:
    torch, htorch = _require_torch_cuda_triton()

    n = 2048
    hi = 1 << nbits
    x_np = rng.integers(0, hi, size=n, dtype=dtype)
    y_np = rng.integers(0, hi, size=n, dtype=dtype)
    x = torch.from_numpy(x_np).to(device="cuda")
    y = torch.from_numpy(y_np).to(device="cuda")

    htorch.precache_compile_luts(device=x.device, op="hilbert_encode_2d")

    def fn(a, b):
        return htorch.hilbert_encode_2d(a, b, nbits=nbits, gpu_backend="triton")

    compiled_fn = torch.compile(fn, fullgraph=True)
    out = compiled_fn(x, y)

    ref = np_hilbert_encode_2d(
        x_np.astype(np.int64), y_np.astype(np.int64), nbits=nbits
    )
    np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.compile
@pytest.mark.no_graph_break
@pytest.mark.triton
@pytest.mark.parametrize(
    "index_dtype,nbits", _CUDA_TRITON_2D_DECODE_INDEX_DTYPE_SWEEP_CASES
)
def test_cuda_compile_fullgraph_with_triton_2d_decode_small_dtypes_and_nbits(
    rng: np.random.Generator,
    index_dtype: np.dtype,
    nbits: int,
) -> None:
    torch, htorch = _require_torch_cuda_triton()

    n = 2048
    hi = 1 << nbits
    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)
    idx_np = np_hilbert_encode_2d(x_np, y_np, nbits=nbits).astype(index_dtype)
    idx = torch.from_numpy(idx_np).to(device="cuda")

    htorch.precache_compile_luts(device=idx.device, op="hilbert_decode_2d")

    def fn(index):
        return htorch.hilbert_decode_2d(index, nbits=nbits, gpu_backend="triton")

    compiled_fn = torch.compile(fn, fullgraph=True)
    out_x, out_y = compiled_fn(idx)

    np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
def test_gpu_decode_matches_numpy_reference(rng: np.random.Generator) -> None:
    torch, htorch = _require_torch_cuda_triton()

    nbits = 8
    n = 4096

    idx_np = rng.integers(0, 1 << (2 * nbits), size=n, dtype=np.uint32)
    idx = torch.from_numpy(idx_np).to(device="cuda")

    out_x, out_y = htorch.hilbert_decode_2d(idx, nbits=nbits, gpu_backend="auto")
    x_np, y_np = np_hilbert_decode_2d(idx_np, nbits=nbits)

    np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
def test_triton_encode_decode_3d_matches_numpy(rng: np.random.Generator) -> None:
    torch, htorch = _require_torch_cuda_triton()

    nbits = 8
    n = 2048
    hi = 1 << nbits

    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)
    z_np = rng.integers(0, hi, size=n, dtype=np.int32)

    x = torch.from_numpy(x_np).to(device="cuda")
    y = torch.from_numpy(y_np).to(device="cuda")
    z = torch.from_numpy(z_np).to(device="cuda")

    idx = htorch.hilbert_encode_3d(x, y, z, nbits=nbits, gpu_backend="triton")
    idx_np = idx.detach().cpu().numpy()

    ref_idx = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
    np.testing.assert_array_equal(idx_np, ref_idx)

    out_x, out_y, out_z = htorch.hilbert_decode_3d(
        idx,
        nbits=nbits,
        gpu_backend="triton",
    )
    np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
    np.testing.assert_array_equal(out_z.detach().cpu().numpy(), z_np)


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
def test_gpu_decode_3d_matches_numpy_reference(rng: np.random.Generator) -> None:
    torch, htorch = _require_torch_cuda_triton()

    nbits = 6
    n = 2048

    idx_np = rng.integers(0, 1 << (3 * nbits), size=n, dtype=np.uint32)
    idx = torch.from_numpy(idx_np).to(device="cuda")

    out_x, out_y, out_z = htorch.hilbert_decode_3d(
        idx,
        nbits=nbits,
        gpu_backend="auto",
    )
    x_np, y_np, z_np = np_hilbert_decode_3d(idx_np, nbits=nbits)

    np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
    np.testing.assert_array_equal(out_z.detach().cpu().numpy(), z_np)
