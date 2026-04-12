import importlib
import importlib.util

import pytest


def test_hilbertsfc_torch_import_requires_torch() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip(
            "torch is installed; optional-dependency import guard not exercised"
        )

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("hilbertsfc.torch")

    msg = str(excinfo.value)
    assert "hilbertsfc.torch" in msg
    assert "hilbertsfc[torch]" in msg
