import numpy as np
import pytest

from hilbertsfc import hilbert_encode_2d as np_hilbert_encode_2d
from hilbertsfc import hilbert_encode_3d as np_hilbert_encode_3d


def _torch_pair():
    torch = pytest.importorskip("torch")
    htorch = pytest.importorskip("hilbertsfc.torch")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this torch build")
    return torch, htorch


def _rand_xyz(
    rng: np.random.Generator,
    *,
    dim: int,
    nbits: int,
    n: int,
) -> tuple[np.ndarray, ...]:
    hi = 1 << nbits
    x_np = rng.integers(0, hi, size=n, dtype=np.int32)
    y_np = rng.integers(0, hi, size=n, dtype=np.int32)
    if dim == 2:
        return (x_np, y_np)
    if dim == 3:
        z_np = rng.integers(0, hi, size=n, dtype=np.int32)
        return (x_np, y_np, z_np)
    raise AssertionError(f"Unsupported {dim=}")


_CPU_TORCH_FULLGRAPH_CASES = [
    pytest.param("encode", 2, 9, 1024, id="cpu-torch-encode-2d"),
    pytest.param("decode", 2, 8, 1024, id="cpu-torch-decode-2d"),
    pytest.param("encode", 3, 7, 1024, id="cpu-torch-encode-3d"),
    pytest.param("decode", 3, 6, 1024, id="cpu-torch-decode-3d"),
]


_CUDA_TORCH_FULLGRAPH_CASES = [
    pytest.param("encode", 2, 10, 2048, id="cuda-torch-encode-2d"),
    pytest.param("decode", 2, 9, 2048, id="cuda-torch-decode-2d"),
    pytest.param("encode", 3, 8, 1024, id="cuda-torch-encode-3d"),
    pytest.param("decode", 3, 7, 1024, id="cuda-torch-decode-3d"),
]


_CPU_NUMBA_CASES = [
    pytest.param("encode", 2, 10, 1024, id="cpu-numba-encode-2d"),
    pytest.param("decode", 2, 10, 1024, id="cpu-numba-decode-2d"),
    pytest.param("encode", 3, 8, 1024, id="cpu-numba-encode-3d"),
    pytest.param("decode", 3, 8, 1024, id="cpu-numba-decode-3d"),
]


@pytest.mark.torch
@pytest.mark.compile
@pytest.mark.parametrize("op,dim,nbits,n", _CPU_NUMBA_CASES)
def test_torch_compile_cpu_numba_backend_allows_graph_breaks(
    rng: np.random.Generator,
    op: str,
    dim: int,
    nbits: int,
    n: int,
) -> None:
    torch, htorch = _torch_pair()

    # The CPU numba backend is not torch.compile-friendly (it is wrapped in
    # `torch._dynamo.disable(...)`), so graph breaks are expected/allowed.
    if dim == 2:
        x_np, y_np = _rand_xyz(rng, dim=2, nbits=nbits, n=n)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        if op == "encode":

            def fn(a, b):  # type: ignore[reportRedeclaration]
                return htorch.hilbert_encode_2d(
                    a,
                    b,
                    nbits=nbits,
                    cpu_backend="numba",
                )

            compiled_fn = torch.compile(fn, fullgraph=False)
            out_idx = compiled_fn(x, y)

            ref_idx = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
            np.testing.assert_array_equal(out_idx.detach().cpu().numpy(), ref_idx)
            return

        if op == "decode":
            idx_np = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
            idx = torch.from_numpy(idx_np)

            def fn(index):
                return htorch.hilbert_decode_2d(
                    index,
                    nbits=nbits,
                    cpu_backend="numba",
                )

            compiled_fn = torch.compile(fn, fullgraph=False)
            out_x, out_y = compiled_fn(idx)

            np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
            np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
            return

        raise AssertionError(f"Unsupported {op=}")

    if dim == 3:
        x_np, y_np, z_np = _rand_xyz(rng, dim=3, nbits=nbits, n=n)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        z = torch.from_numpy(z_np)

        if op == "encode":

            def fn(a, b, c):  # type: ignore[reportRedeclaration]
                return htorch.hilbert_encode_3d(
                    a,
                    b,
                    c,
                    nbits=nbits,
                    cpu_backend="numba",
                )

            compiled_fn = torch.compile(fn, fullgraph=False)
            out_idx = compiled_fn(x, y, z)

            ref_idx = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
            np.testing.assert_array_equal(out_idx.detach().cpu().numpy(), ref_idx)
            return

        if op == "decode":
            idx_np = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
            idx = torch.from_numpy(idx_np)

            def fn(index):
                return htorch.hilbert_decode_3d(
                    index,
                    nbits=nbits,
                    cpu_backend="numba",
                )

            compiled_fn = torch.compile(fn, fullgraph=False)
            out_x, out_y, out_z = compiled_fn(idx)

            np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
            np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
            np.testing.assert_array_equal(out_z.detach().cpu().numpy(), z_np)
            return

        raise AssertionError(f"Unsupported {op=}")

    raise AssertionError(f"Unsupported {dim=}")


@pytest.mark.torch
@pytest.mark.compile
@pytest.mark.no_graph_break
@pytest.mark.parametrize("op,dim,nbits,n", _CPU_TORCH_FULLGRAPH_CASES)
def test_torch_compile_fullgraph_cpu_torch_backend_no_graph_breaks(
    rng: np.random.Generator,
    op: str,
    dim: int,
    nbits: int,
    n: int,
) -> None:
    torch, htorch = _torch_pair()

    if dim == 2 and op == "encode":
        x_np, y_np = _rand_xyz(rng, dim=2, nbits=nbits, n=n)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        htorch.precache_compile_luts(device=x.device, op="hilbert_encode_2d")

        def fn(a, b):  # type: ignore[reportRedeclaration]
            return htorch.hilbert_encode_2d(a, b, nbits=nbits, cpu_backend="torch")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out = compiled_fn(x, y)

        ref = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    if dim == 2 and op == "decode":
        x_np, y_np = _rand_xyz(rng, dim=2, nbits=nbits, n=n)
        idx_np = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
        idx = torch.from_numpy(idx_np)

        htorch.precache_compile_luts(device=idx.device, op="hilbert_decode_2d")

        def fn(index):
            return htorch.hilbert_decode_2d(index, nbits=nbits, cpu_backend="torch")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out_x, out_y = compiled_fn(idx)

        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
        return

    if dim == 3 and op == "encode":
        x_np, y_np, z_np = _rand_xyz(rng, dim=3, nbits=nbits, n=n)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        z = torch.from_numpy(z_np)

        htorch.precache_compile_luts(device=x.device, op="hilbert_encode_3d")

        def fn(a, b, c):  # type: ignore[reportRedeclaration]
            return htorch.hilbert_encode_3d(
                a,
                b,
                c,
                nbits=nbits,
                cpu_backend="torch",
            )

        compiled_fn = torch.compile(fn, fullgraph=True)
        out = compiled_fn(x, y, z)

        ref = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    if dim == 3 and op == "decode":
        x_np, y_np, z_np = _rand_xyz(rng, dim=3, nbits=nbits, n=n)
        idx_np = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
        idx = torch.from_numpy(idx_np)

        htorch.precache_compile_luts(device=idx.device, op="hilbert_decode_3d")

        def fn(index):
            return htorch.hilbert_decode_3d(index, nbits=nbits, cpu_backend="torch")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out_x, out_y, out_z = compiled_fn(idx)

        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
        np.testing.assert_array_equal(out_z.detach().cpu().numpy(), z_np)
        return

    raise AssertionError(f"Unsupported combination: {op=}, {dim=}")


@pytest.mark.torch
@pytest.mark.gpu
@pytest.mark.compile
@pytest.mark.no_graph_break
@pytest.mark.parametrize("op,dim,nbits,n", _CUDA_TORCH_FULLGRAPH_CASES)
def test_torch_compile_fullgraph_cuda_torch_backend_no_graph_breaks(
    rng: np.random.Generator,
    op: str,
    dim: int,
    nbits: int,
    n: int,
) -> None:
    torch, htorch = _torch_pair()

    if dim == 2 and op == "encode":
        x_np, y_np = _rand_xyz(rng, dim=2, nbits=nbits, n=n)
        x = torch.from_numpy(x_np).to(device="cuda")
        y = torch.from_numpy(y_np).to(device="cuda")

        htorch.precache_compile_luts(device=x.device, op="hilbert_encode_2d")

        def fn(a, b):  # type: ignore[reportRedeclaration]
            return htorch.hilbert_encode_2d(a, b, nbits=nbits, gpu_backend="torch")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out = compiled_fn(x, y)

        ref = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    if dim == 2 and op == "decode":
        x_np, y_np = _rand_xyz(rng, dim=2, nbits=nbits, n=n)
        idx_np = np_hilbert_encode_2d(x_np, y_np, nbits=nbits)
        idx = torch.from_numpy(idx_np).to(device="cuda")

        htorch.precache_compile_luts(device=idx.device, op="hilbert_decode_2d")

        def fn(index):
            return htorch.hilbert_decode_2d(index, nbits=nbits, gpu_backend="torch")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out_x, out_y = compiled_fn(idx)

        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
        return

    if dim == 3 and op == "encode":
        x_np, y_np, z_np = _rand_xyz(rng, dim=3, nbits=nbits, n=n)
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
                gpu_backend="torch",
            )

        compiled_fn = torch.compile(fn, fullgraph=True)
        out = compiled_fn(x, y, z)

        ref = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
        np.testing.assert_array_equal(out.detach().cpu().numpy(), ref)
        return

    if dim == 3 and op == "decode":
        x_np, y_np, z_np = _rand_xyz(rng, dim=3, nbits=nbits, n=n)
        idx_np = np_hilbert_encode_3d(x_np, y_np, z_np, nbits=nbits)
        idx = torch.from_numpy(idx_np).to(device="cuda")

        htorch.precache_compile_luts(device=idx.device, op="hilbert_decode_3d")

        def fn(index):
            return htorch.hilbert_decode_3d(index, nbits=nbits, gpu_backend="torch")

        compiled_fn = torch.compile(fn, fullgraph=True)
        out_x, out_y, out_z = compiled_fn(idx)

        np.testing.assert_array_equal(out_x.detach().cpu().numpy(), x_np)
        np.testing.assert_array_equal(out_y.detach().cpu().numpy(), y_np)
        np.testing.assert_array_equal(out_z.detach().cpu().numpy(), z_np)
        return

    raise AssertionError(f"Unsupported combination: {op=}, {dim=}")
