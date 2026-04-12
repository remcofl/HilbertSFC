import pytest


def _dispatch_common_module():
    pytest.importorskip("torch")
    return __import__("hilbertsfc.torch._dispatch_common", fromlist=["*"])


@pytest.mark.torch
def test_attempt_run_triton_unavailable_auto_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _dispatch_common_module()

    monkeypatch.setattr(mod, "is_triton_available", lambda: False)

    called = {"value": False}

    def _call() -> None:
        called["value"] = True

    ok = mod.attempt_run_triton(
        gpu_backend="auto",
        all_contiguous=True,
        contiguity_details="",
        call_triton=_call,
    )

    assert ok is False
    assert called["value"] is False


@pytest.mark.torch
def test_attempt_run_triton_unavailable_forced_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _dispatch_common_module()

    monkeypatch.setattr(mod, "is_triton_available", lambda: False)

    with pytest.raises(RuntimeError, match="requested, but Triton is unavailable"):
        mod.attempt_run_triton(
            gpu_backend="triton",
            all_contiguous=True,
            contiguity_details="",
            call_triton=lambda: None,
        )


@pytest.mark.torch
def test_attempt_run_triton_non_contiguous_auto_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _dispatch_common_module()

    monkeypatch.setattr(mod, "is_triton_available", lambda: True)

    with pytest.warns(UserWarning, match="falling back to gpu_backend='torch'"):
        ok = mod.attempt_run_triton(
            gpu_backend="auto",
            all_contiguous=False,
            contiguity_details="x.is_contiguous()=False",
            call_triton=lambda: None,
        )

    assert ok is False


@pytest.mark.torch
def test_attempt_run_triton_non_contiguous_forced_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _dispatch_common_module()

    monkeypatch.setattr(mod, "is_triton_available", lambda: True)

    with pytest.raises(ValueError, match="requires contiguous tensors"):
        mod.attempt_run_triton(
            gpu_backend="triton",
            all_contiguous=False,
            contiguity_details="x.is_contiguous()=False",
            call_triton=lambda: None,
        )


@pytest.mark.torch
def test_attempt_run_triton_runtime_failure_auto_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _dispatch_common_module()

    monkeypatch.setattr(mod, "is_triton_available", lambda: True)

    def _boom() -> None:
        raise RuntimeError("boom")

    with pytest.warns(UserWarning, match="Triton kernel failed at runtime"):
        ok = mod.attempt_run_triton(
            gpu_backend="auto",
            all_contiguous=True,
            contiguity_details="",
            call_triton=_boom,
        )

    assert ok is False


@pytest.mark.torch
def test_attempt_run_triton_runtime_failure_forced_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _dispatch_common_module()

    monkeypatch.setattr(mod, "is_triton_available", lambda: True)

    def _boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="requested, but Triton kernel failed"):
        mod.attempt_run_triton(
            gpu_backend="triton",
            all_contiguous=True,
            contiguity_details="",
            call_triton=_boom,
        )
