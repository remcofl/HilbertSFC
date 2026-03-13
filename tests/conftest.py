import numpy as np
import pytest


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
