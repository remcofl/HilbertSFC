import pytest


@pytest.mark.torch
def test_torch_package_exports() -> None:
    htorch = pytest.importorskip("hilbertsfc.torch")

    for name in [
        "hilbert_encode_2d",
        "hilbert_decode_2d",
        "hilbert_encode_3d",
        "hilbert_decode_3d",
        "precache_compile_luts",
        "clear_torch_lut_caches",
    ]:
        assert hasattr(htorch, name)


@pytest.mark.torch
def test_torch_types_exports() -> None:
    types_mod = pytest.importorskip("hilbertsfc.torch.types")

    for name in [
        "CPUBackend",
        "GPUBackend",
        "TorchCacheMode",
        "TorchDeviceLike",
        "TorchHilbertOp",
    ]:
        assert hasattr(types_mod, name)
