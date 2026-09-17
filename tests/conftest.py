import importlib.util
import os
import shutil
from functools import cache

import numpy as np
import pytest

STRICT_OPTIONAL_TESTS = os.environ.get("HILBERTSFC_STRICT_OPTIONAL_TESTS") == "1"


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
def has_torch_compile_cpu_toolchain() -> bool:
    compiler_names = (
        ("cl.exe", "clang-cl.exe", "icx-cl.exe")
        if os.name == "nt"
        else ("c++", "g++", "clang++")
    )
    return any(shutil.which(compiler) is not None for compiler in compiler_names)


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

        return sm in arch_list
    except Exception:
        pass

    return True


def skip_or_fail(reason: str) -> None:
    if STRICT_OPTIONAL_TESTS:
        pytest.fail(reason)
    pytest.skip(reason)


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "torch" in item.keywords and not has_torch():
        skip_or_fail("torch is not available")

    if (
        "compile" in item.keywords
        and "gpu" not in item.keywords
        and not has_torch_compile_cpu_toolchain()
    ):
        reason = "torch.compile CPU tests require an active C++ compiler"
        if os.name == "nt":
            reason += "; run pytest from a Visual Studio Developer shell"
        skip_or_fail(reason)

    if "gpu" in item.keywords and not has_cuda():
        skip_or_fail("CUDA is not available")

    if "triton" in item.keywords and not (has_cuda() and has_triton()):
        skip_or_fail("Triton/CUDA is unavailable")


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
