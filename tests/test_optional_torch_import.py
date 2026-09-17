import importlib
import importlib.util
import sys

import pytest


def test_hilbertsfc_torch_import_requires_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec_without_torch(name: str, *args, **kwargs):
        if name == "torch":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_torch)
    monkeypatch.delitem(sys.modules, "hilbertsfc.torch", raising=False)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("hilbertsfc.torch")

    msg = str(excinfo.value)
    assert "hilbertsfc.torch" in msg
    assert "hilbertsfc[torch]" in msg
