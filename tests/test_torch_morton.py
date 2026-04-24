import numpy as np
import pytest

from hilbertsfc import morton_encode_2d as np_morton_encode_2d
from hilbertsfc import morton_encode_3d as np_morton_encode_3d


def _torch_pair():
    torch = pytest.importorskip("torch")
    htorch = pytest.importorskip("hilbertsfc.torch")
    return torch, htorch


def _require_torch_compile():
    torch, htorch = _torch_pair()
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this torch build")
    return torch, htorch


def _require_torch_cuda_triton():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    htorch = pytest.importorskip("hilbertsfc.torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    return torch, htorch


@pytest.mark.torch
@pytest.mark.parametrize("dim,nbits,n", [(2, 12, 2048), (3, 8, 1024)])
def test_torch_morton_cpu_torch_matches_numpy(
    rng: np.random.Generator, dim: int, nbits: int, n: int
) -> None:
    torch, htorch = _torch_pair()

    hi = 1 << nbits
    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    if dim == 2:
        idx = htorch.morton_encode_2d(
            x, y, nbits=nbits, cpu_backend="torch", gpu_backend="torch"
        )
        ref = np_morton_encode_2d(x_np, y_np, nbits=nbits)
        np.testing.assert_array_equal(idx.cpu().numpy(), ref)

        out_x, out_y = htorch.morton_decode_2d(
            idx, nbits=nbits, cpu_backend="torch", gpu_backend="torch"
        )
        np.testing.assert_array_equal(out_x.cpu().numpy(), x_np)
        np.testing.assert_array_equal(out_y.cpu().numpy(), y_np)
        return

    z_np = rng.integers(0, hi, size=n, dtype=np.int32)
    z = torch.from_numpy(z_np)
    idx = htorch.morton_encode_3d(
        x, y, z, nbits=nbits, cpu_backend="torch", gpu_backend="torch"
    )
    ref = np_morton_encode_3d(x_np, y_np, z_np, nbits=nbits)
    np.testing.assert_array_equal(idx.cpu().numpy(), ref)

    out_x, out_y, out_z = htorch.morton_decode_3d(
        idx, nbits=nbits, cpu_backend="torch", gpu_backend="torch"
    )
    np.testing.assert_array_equal(out_x.cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.cpu().numpy(), y_np)
    np.testing.assert_array_equal(out_z.cpu().numpy(), z_np)


@pytest.mark.torch
@pytest.mark.parametrize("dim,nbits", [(2, 16), (2, 17), (3, 10), (3, 11)])
def test_torch_morton_cpu_numba_and_torch_match_boundaries(
    rng: np.random.Generator, dim: int, nbits: int
) -> None:
    torch, htorch = _torch_pair()

    n = 1024
    hi = 1 << nbits
    x = torch.from_numpy(rng.integers(0, hi, size=n, dtype=np.int32))
    y = torch.from_numpy(rng.integers(0, hi, size=n, dtype=np.int32))

    if dim == 2:
        idx_numba = htorch.morton_encode_2d(x, y, nbits=nbits, cpu_backend="numba")
        idx_torch = htorch.morton_encode_2d(x, y, nbits=nbits, cpu_backend="torch")
        np.testing.assert_array_equal(idx_numba.numpy(), idx_torch.numpy())

        xy_numba = htorch.morton_decode_2d(idx_numba, nbits=nbits, cpu_backend="numba")
        xy_torch = htorch.morton_decode_2d(idx_torch, nbits=nbits, cpu_backend="torch")
        np.testing.assert_array_equal(xy_numba[0].numpy(), xy_torch[0].numpy())
        np.testing.assert_array_equal(xy_numba[1].numpy(), xy_torch[1].numpy())
        return

    z = torch.from_numpy(rng.integers(0, hi, size=n, dtype=np.int32))
    idx_numba = htorch.morton_encode_3d(x, y, z, nbits=nbits, cpu_backend="numba")
    idx_torch = htorch.morton_encode_3d(x, y, z, nbits=nbits, cpu_backend="torch")
    np.testing.assert_array_equal(idx_numba.numpy(), idx_torch.numpy())

    xyz_numba = htorch.morton_decode_3d(idx_numba, nbits=nbits, cpu_backend="numba")
    xyz_torch = htorch.morton_decode_3d(idx_torch, nbits=nbits, cpu_backend="torch")
    np.testing.assert_array_equal(xyz_numba[0].numpy(), xyz_torch[0].numpy())
    np.testing.assert_array_equal(xyz_numba[1].numpy(), xyz_torch[1].numpy())
    np.testing.assert_array_equal(xyz_numba[2].numpy(), xyz_torch[2].numpy())


@pytest.mark.torch
def test_torch_morton_dtype_and_out_buffers() -> None:
    torch, htorch = _torch_pair()

    x = torch.tensor([0, 1, 65535], dtype=torch.int32)
    y = torch.tensor([0, 2, 65535], dtype=torch.int32)

    idx = htorch.morton_encode_2d(x, y, nbits=16, cpu_backend="torch")
    assert idx.dtype == torch.int64

    out = torch.empty_like(idx)
    ret = htorch.morton_encode_2d(x, y, nbits=16, out=out, cpu_backend="torch")
    assert ret is out
    np.testing.assert_array_equal(out.numpy(), idx.numpy())

    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    ret_x, ret_y = htorch.morton_decode_2d(
        idx, nbits=16, out_x=out_x, out_y=out_y, cpu_backend="torch"
    )
    assert ret_x is out_x
    assert ret_y is out_y
    np.testing.assert_array_equal(out_x.numpy(), x.numpy())
    np.testing.assert_array_equal(out_y.numpy(), y.numpy())


@pytest.mark.torch
def test_torch_morton_encode_masks_wide_coordinates_before_i32_path() -> None:
    torch, htorch = _torch_pair()

    nbits = 4
    mask = (1 << nbits) - 1

    x_np = np.array([(1 << 40) + 3, -1, (1 << 35) + 5], dtype=np.int64)
    y_np = np.array([(1 << 42) + 4, (1 << 33) + 6, -2], dtype=np.int64)
    z_np = np.array([(1 << 38) + 7, -3, (1 << 36) + 8], dtype=np.int64)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    z = torch.from_numpy(z_np)

    x_low = x_np & mask
    y_low = y_np & mask
    z_low = z_np & mask

    idx_2d = htorch.morton_encode_2d(x, y, nbits=nbits, cpu_backend="torch")
    ref_2d = np_morton_encode_2d(x_low, y_low, nbits=nbits)
    np.testing.assert_array_equal(idx_2d.numpy(), ref_2d)

    out_x, out_y = htorch.morton_decode_2d(idx_2d, nbits=nbits, cpu_backend="torch")
    np.testing.assert_array_equal(out_x.numpy(), x_low)
    np.testing.assert_array_equal(out_y.numpy(), y_low)

    idx_3d = htorch.morton_encode_3d(x, y, z, nbits=nbits, cpu_backend="torch")
    ref_3d = np_morton_encode_3d(x_low, y_low, z_low, nbits=nbits)
    np.testing.assert_array_equal(idx_3d.numpy(), ref_3d)

    out_x, out_y, out_z = htorch.morton_decode_3d(
        idx_3d, nbits=nbits, cpu_backend="torch"
    )
    np.testing.assert_array_equal(out_x.numpy(), x_low)
    np.testing.assert_array_equal(out_y.numpy(), y_low)
    np.testing.assert_array_equal(out_z.numpy(), z_low)


@pytest.mark.torch
def test_torch_morton_validation_errors() -> None:
    torch, htorch = _torch_pair()

    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    y = torch.tensor([1, 2], dtype=torch.int32)

    with pytest.raises(ValueError, match="same shape"):
        htorch.morton_encode_2d(x, y, nbits=4)

    with pytest.raises(ValueError, match="gpu_backend must be one of"):
        htorch.morton_encode_2d(x, x, nbits=4, gpu_backend="bad")  # type: ignore[arg-type]

    index = torch.tensor([1, 2, 3], dtype=torch.uint16)
    out_x = torch.empty_like(index, dtype=torch.uint8)
    with pytest.raises(ValueError, match="out_x and out_y must be provided together"):
        htorch.morton_decode_2d(index, nbits=8, out_x=out_x)

    with pytest.raises(
        ValueError, match="exceeds the effective bits of the index dtype"
    ):
        htorch.morton_decode_2d(index, nbits=9)


@pytest.mark.torch
@pytest.mark.compile
@pytest.mark.no_graph_break
@pytest.mark.parametrize(
    "dim,op,nbits",
    [(2, "encode", 9), (2, "decode", 8), (3, "encode", 7), (3, "decode", 6)],
)
def test_torch_compile_fullgraph_cpu_torch_morton(
    rng: np.random.Generator, dim: int, op: str, nbits: int
) -> None:
    torch, htorch = _require_torch_compile()

    n = 512
    hi = 1 << nbits
    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    if dim == 2 and op == "encode":

        def fn(a, b):
            return htorch.morton_encode_2d(a, b, nbits=nbits, cpu_backend="torch")

        out = torch.compile(fn, fullgraph=True)(x, y)
        ref = np_morton_encode_2d(x_np, y_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    if dim == 2 and op == "decode":
        idx_np = np_morton_encode_2d(x_np, y_np, nbits=nbits)
        idx = torch.from_numpy(idx_np)

        def fn(index):
            return htorch.morton_decode_2d(index, nbits=nbits, cpu_backend="torch")

        out_x, out_y = torch.compile(fn, fullgraph=True)(idx)
        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
        return

    z_np = rng.integers(0, hi, size=n, dtype=np.int32)
    z = torch.from_numpy(z_np)

    if dim == 3 and op == "encode":

        def fn(a, b, c):
            return htorch.morton_encode_3d(a, b, c, nbits=nbits, cpu_backend="torch")

        out = torch.compile(fn, fullgraph=True)(x, y, z)
        ref = np_morton_encode_3d(x_np, y_np, z_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    idx_np = np_morton_encode_3d(x_np, y_np, z_np, nbits=nbits)
    idx = torch.from_numpy(idx_np)

    def fn(index):
        return htorch.morton_decode_3d(index, nbits=nbits, cpu_backend="torch")

    out_x, out_y, out_z = torch.compile(fn, fullgraph=True)(idx)
    np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
    np.testing.assert_array_equal(out_z.detach().cpu().numpy(), z_np)


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.parametrize("dim,nbits", [(2, 12), (3, 8)])
def test_triton_morton_matches_numpy(
    rng: np.random.Generator, dim: int, nbits: int
) -> None:
    torch, htorch = _require_torch_cuda_triton()

    n = 2048
    hi = 1 << nbits
    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)
    x = torch.from_numpy(x_np).to(device="cuda")
    y = torch.from_numpy(y_np).to(device="cuda")

    if dim == 2:
        idx = htorch.morton_encode_2d(x, y, nbits=nbits, gpu_backend="triton")
        ref = np_morton_encode_2d(x_np, y_np, nbits=nbits)
        np.testing.assert_array_equal(idx.detach().cpu().numpy(), ref)

        out_x, out_y = htorch.morton_decode_2d(idx, nbits=nbits, gpu_backend="triton")
        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
        return

    z_np = rng.integers(0, hi, size=n, dtype=np.int32)
    z = torch.from_numpy(z_np).to(device="cuda")
    idx = htorch.morton_encode_3d(x, y, z, nbits=nbits, gpu_backend="triton")
    ref = np_morton_encode_3d(x_np, y_np, z_np, nbits=nbits)
    np.testing.assert_array_equal(idx.detach().cpu().numpy(), ref)

    out_x, out_y, out_z = htorch.morton_decode_3d(
        idx, nbits=nbits, gpu_backend="triton"
    )
    np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
    np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
    np.testing.assert_array_equal(out_z.detach().cpu().numpy(), z_np)


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.triton
def test_triton_morton_auto_non_contiguous_falls_back(
    rng: np.random.Generator,
) -> None:
    torch, htorch = _require_torch_cuda_triton()

    nbits = 9
    n = 1024
    hi = 1 << nbits
    x_base = torch.from_numpy(rng.integers(0, hi, size=(n, 2), dtype=np.int32)).to(
        device="cuda"
    )
    y_base = torch.from_numpy(rng.integers(0, hi, size=(n, 2), dtype=np.int32)).to(
        device="cuda"
    )

    x = x_base[:, 0]
    y = y_base[:, 0]

    with pytest.warns(UserWarning, match="falling back to gpu_backend='torch'"):
        idx_auto = htorch.morton_encode_2d(x, y, nbits=nbits, gpu_backend="auto")

    idx_torch = htorch.morton_encode_2d(x, y, nbits=nbits, gpu_backend="torch")
    np.testing.assert_array_equal(
        idx_auto.detach().cpu().numpy(),
        idx_torch.detach().cpu().numpy(),
    )

    with pytest.raises(ValueError, match="requires contiguous tensors"):
        htorch.morton_encode_2d(x, y, nbits=nbits, gpu_backend="triton")
