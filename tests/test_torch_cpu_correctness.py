import numpy as np
import pytest

from hilbertsfc import (
    hilbert_encode_2d as np_hilbert_encode_2d,
)
from hilbertsfc import (
    hilbert_encode_3d as np_hilbert_encode_3d,
)


def _torch_pair():
    torch = pytest.importorskip("torch")
    htorch = pytest.importorskip("hilbertsfc.torch")
    return torch, htorch


@pytest.mark.torch
def test_torch_encode_decode_2d_matches_numpy_cpu(rng: np.random.Generator) -> None:
    torch, htorch = _torch_pair()

    nbits = 12
    n = 2048
    hi = 1 << nbits

    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    idx_t = htorch.hilbert_encode_2d(
        x,
        y,
        nbits=nbits,
        cpu_backend="torch",
        gpu_backend="torch",
    )

    idx_np = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
    np.testing.assert_array_equal(idx_t.cpu().numpy(), idx_np)

    x_t2, y_t2 = htorch.hilbert_decode_2d(
        idx_t,
        nbits=nbits,
        cpu_backend="torch",
        gpu_backend="torch",
    )
    np.testing.assert_array_equal(x_t2.cpu().numpy(), x_np)
    np.testing.assert_array_equal(y_t2.cpu().numpy(), y_np)


@pytest.mark.torch
def test_torch_encode_decode_3d_matches_numpy_cpu(rng: np.random.Generator) -> None:
    torch, htorch = _torch_pair()

    nbits = 8
    n = 1024
    hi = 1 << nbits

    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)
    z_np = rng.integers(0, hi, size=n, dtype=np.int32)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    z = torch.from_numpy(z_np)

    idx_t = htorch.hilbert_encode_3d(
        x,
        y,
        z,
        nbits=nbits,
        cpu_backend="torch",
        gpu_backend="torch",
    )

    idx_np = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
    np.testing.assert_array_equal(idx_t.cpu().numpy(), idx_np)

    x_t2, y_t2, z_t2 = htorch.hilbert_decode_3d(
        idx_t,
        nbits=nbits,
        cpu_backend="torch",
        gpu_backend="torch",
    )
    np.testing.assert_array_equal(x_t2.cpu().numpy(), x_np)
    np.testing.assert_array_equal(y_t2.cpu().numpy(), y_np)
    np.testing.assert_array_equal(z_t2.cpu().numpy(), z_np)


@pytest.mark.torch
def test_cpu_backend_numba_and_torch_match(rng: np.random.Generator) -> None:
    torch, htorch = _torch_pair()

    nbits = 10
    n = 1024
    hi = 1 << nbits

    x = torch.from_numpy(rng.integers(0, hi, size=n, dtype=np.int32))
    y = torch.from_numpy(rng.integers(0, hi, size=n, dtype=np.int32))

    idx_numba = htorch.hilbert_encode_2d(x, y, nbits=nbits, cpu_backend="numba")
    idx_torch = htorch.hilbert_encode_2d(x, y, nbits=nbits, cpu_backend="torch")

    assert idx_numba.device.type == "cpu"
    assert idx_torch.device.type == "cpu"
    np.testing.assert_array_equal(idx_numba.numpy(), idx_torch.numpy())


@pytest.mark.torch
def test_invalid_gpu_backend_value_raises() -> None:
    torch, htorch = _torch_pair()

    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    y = torch.tensor([4, 5, 6], dtype=torch.int32)

    with pytest.raises(ValueError, match="gpu_backend must be one of"):
        htorch.hilbert_encode_2d(x, y, nbits=4, gpu_backend="bad")  # type: ignore[arg-type]


@pytest.mark.torch
def test_decode_2d_requires_out_pair() -> None:
    torch, htorch = _torch_pair()

    index = torch.tensor([1, 2, 3], dtype=torch.uint16)
    out_x = torch.empty_like(index, dtype=torch.uint8)

    with pytest.raises(ValueError, match="out_x and out_y must be provided together"):
        htorch.hilbert_decode_2d(index, nbits=8, out_x=out_x)


@pytest.mark.torch
def test_decode_2d_nbits_exceeds_index_dtype_raises() -> None:
    torch, htorch = _torch_pair()

    # uint16 supports at most nbits=8 for 2D decode.
    index = torch.tensor([1, 2, 3], dtype=torch.uint16)

    with pytest.raises(
        ValueError, match="exceeds the effective bits of the index dtype"
    ):
        htorch.hilbert_decode_2d(index, nbits=9)


@pytest.mark.torch
def test_decode_2d_out_device_mismatch_raises() -> None:
    torch, htorch = _torch_pair()

    index = torch.tensor([1, 2, 3], dtype=torch.uint16)
    out_x = torch.empty(index.shape, dtype=torch.uint8, device="meta")
    out_y = torch.empty(index.shape, dtype=torch.uint8, device="meta")

    with pytest.raises(ValueError, match="must be on the same device as index"):
        htorch.hilbert_decode_2d(index, nbits=8, out_x=out_x, out_y=out_y)


@pytest.mark.torch
def test_decode_2d_out_shape_mismatch_raises() -> None:
    torch, htorch = _torch_pair()

    index = torch.tensor([1, 2, 3], dtype=torch.uint16)
    out_x = torch.empty((2,), dtype=torch.uint8)
    out_y = torch.empty((2,), dtype=torch.uint8)

    with pytest.raises(ValueError, match="must have the same shape as index"):
        htorch.hilbert_decode_2d(index, nbits=8, out_x=out_x, out_y=out_y)


@pytest.mark.torch
def test_decode_2d_nbits_exceeds_out_coord_dtype_raises() -> None:
    torch, htorch = _torch_pair()

    index = torch.tensor([1, 2, 3], dtype=torch.uint64)
    out_x = torch.empty(index.shape, dtype=torch.uint8)
    out_y = torch.empty(index.shape, dtype=torch.uint8)

    with pytest.raises(ValueError, match="does not fit in out_x/out_y dtypes"):
        htorch.hilbert_decode_2d(index, nbits=9, out_x=out_x, out_y=out_y)
