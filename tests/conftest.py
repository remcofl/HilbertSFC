import importlib.util
from functools import cache

import numpy as np
import pytest


@cache
def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@cache
def has_torch() -> bool:
    return module_available("torch")


@cache
def has_triton() -> bool:
    return module_available("triton")


@cache
def has_torch_compile() -> bool:
    if not has_torch():
        return False

    import torch

    return hasattr(torch, "compile")


@cache
def has_cuda() -> bool:
    if not has_torch():
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    # Some environments report CUDA as available, but the installed PyTorch build
    # does not contain kernels for the present GPU architecture (e.g. newer GPUs
    # than the wheel was built for). In that case, CUDA ops fail at runtime with
    # "no kernel image is available".
    try:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        arch_list = torch.cuda.get_arch_list()

        # If the arch isn't in the build, do a tiny kernel smoke test to see if
        # PTX/JIT still makes it runnable.
        if arch_list and sm not in arch_list:
            try:
                torch.empty(1, device="cuda").fill_(0)
            except RuntimeError:
                return False
    except Exception:
        pass

    return True


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "torch" in item.keywords and not has_torch():
        pytest.skip("torch is not available")

    if "compile" in item.keywords and not has_torch_compile():
        pytest.skip("torch.compile is unavailable in this torch build")

    if "gpu" in item.keywords and not has_cuda():
        pytest.skip("CUDA is not available")

    if "triton" in item.keywords and not (has_cuda() and has_triton()):
        pytest.skip("Triton/CUDA is unavailable")


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def small_nbits_2d() -> tuple[int, ...]:
    # Keep compilation + brute-force loops fast.
    return (1, 2, 3, 4)


@pytest.fixture(scope="session")
def small_nbits_3d() -> tuple[int, ...]:
    return (1, 2, 3)
