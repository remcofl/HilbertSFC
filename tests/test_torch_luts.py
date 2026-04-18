import pytest


def _torch_luts_module():
    pytest.importorskip("torch")
    return __import__("hilbertsfc.torch._luts", fromlist=["*"])


@pytest.mark.torch
def test_lut_device_cache_reuses_same_tensor() -> None:
    mod = _torch_luts_module()

    mod.clear_torch_lut_caches()
    t1 = mod.lut_2d4b_b_qs_i64(cache="device")
    t2 = mod.lut_2d4b_b_qs_i64(cache="device")

    assert t1 is t2
    assert t1.dtype == t2.dtype


@pytest.mark.torch
def test_lut_host_only_does_not_reuse_tensor() -> None:
    mod = _torch_luts_module()

    t1 = mod.lut_2d4b_b_qs_i64(cache="host_only")
    t2 = mod.lut_2d4b_b_qs_i64(cache="host_only")

    assert t1 is not t2


@pytest.mark.torch
def test_lut_accessors_return_expected_torch_dtypes() -> None:
    mod = _torch_luts_module()

    t2d = mod.lut_2d7b_q_bs_i64(cache="host_only")
    t3d = mod.lut_3d2b_sb_so_i16(cache="host_only")

    torch = pytest.importorskip("torch")
    assert t2d.dtype == torch.int64
    assert t3d.dtype == torch.int16


@pytest.mark.torch
def test_lut_invalid_cache_mode_raises() -> None:
    mod = _torch_luts_module()

    with pytest.raises(ValueError, match="cache must be one of"):
        mod.lut_2d4b_b_qs_i64(cache="bad")


@pytest.mark.torch
def test_clear_torch_lut_caches_filters_by_op() -> None:
    mod = _torch_luts_module()

    mod.clear_torch_lut_caches()
    e1 = mod.lut_2d4b_b_qs_i64(cache="device")
    d1 = mod.lut_2d4b_q_bs_i64(cache="device")

    mod.clear_torch_lut_caches(op="hilbert_encode_2d")

    e2 = mod.lut_2d4b_b_qs_i64(cache="device")
    d2 = mod.lut_2d4b_q_bs_i64(cache="device")

    assert e2 is not e1
    assert d2 is d1


@pytest.mark.torch
def test_clear_torch_lut_caches_unknown_op_raises() -> None:
    mod = _torch_luts_module()

    with pytest.raises(ValueError, match="unknown op"):
        mod.clear_torch_lut_caches(op="bad")


@pytest.mark.torch
def test_precache_compile_luts_unknown_op_raises() -> None:
    mod = _torch_luts_module()

    with pytest.raises(ValueError, match="unknown op"):
        mod.precache_compile_luts(op="bad")


@pytest.mark.torch
def test_precache_compile_luts_populates_expected_entries() -> None:
    mod = _torch_luts_module()

    mod.clear_torch_lut_caches()
    mod.precache_compile_luts(op="hilbert_encode_2d")

    keys = [k[1] for k in mod._DEVICE_LUT_CACHE.keys()]
    assert "lut_2d4b_sb_sq_i16" in keys


@pytest.mark.torch
def test_precache_compile_luts_missing_accessor_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _torch_luts_module()

    names = mod._OP_TO_LUT_NAMES_COMPILE["hilbert_encode_2d"]
    monkeypatch.setitem(
        mod._OP_TO_LUT_NAMES_COMPILE,
        "hilbert_encode_2d",
        ("definitely_missing_accessor",),
    )

    try:
        with pytest.raises(RuntimeError, match="missing LUT accessor"):
            mod.precache_compile_luts(op="hilbert_encode_2d")
    finally:
        monkeypatch.setitem(mod._OP_TO_LUT_NAMES_COMPILE, "hilbert_encode_2d", names)


@pytest.mark.torch
def test_clear_torch_lut_caches_for_single_device_cpu() -> None:
    mod = _torch_luts_module()

    mod.clear_torch_lut_caches()
    t1 = mod.lut_3d2b_so_sb_i16(device="cpu", cache="device")

    mod.clear_torch_lut_caches(device="cpu", op="hilbert_decode_3d")
    t2 = mod.lut_3d2b_so_sb_i16(device="cpu", cache="device")

    assert t2 is not t1
