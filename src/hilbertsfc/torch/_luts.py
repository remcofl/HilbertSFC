"""Torch LUT wrappers with optional per-device caching.

This module bridges the backend-agnostic NumPy LUT accessors in [`hilbertsfc._luts`][hilbertsfc._luts]
with Torch tensors.

Notes
-----
- Root LUTs are always loaded as NumPy arrays (and cached process-wide via
    [`lut_cache`][hilbertsfc._cache.lut_cache]).
- With ``cache='device'``, this module also caches the converted Torch tensor per
    device.
- With ``cache='host_only'``, no Torch-side cache is used; tensors are materialized
  from the cached NumPy arrays on demand.

The compacted 2D LUTs are stored as ``uint64`` on disk but exposed as
``torch.int64`` tensors because the Torch kernels use bitwise shifts/masks and
Torch unsigned integer support is limited.

The 3D 2-bit LUTs are stored as ``uint16`` on disk but exposed as
``torch.int16`` tensors.

For `torch.compile`, building LUT tensors during compilation can cause graph
breaks and extra compile-time overhead. Use [`precache_compile_luts`][hilbertsfc.torch._luts.precache_compile_luts] to
materialize the required LUT(s) before compilation.
"""

import warnings
from collections.abc import Callable
from typing import Literal

import torch

from .. import _luts

type TorchDeviceLike = torch.device | str | None
"""Device specifier with the same semantics as `torch.device`."""

type TorchCacheMode = Literal["device", "host_only"]
"""Cache mode for torch look-up tables (LUTs)."""

TORCH_CACHE_MODES = ("device", "host_only")


def validate_torch_cache_mode(cache: str) -> TorchCacheMode:
    if cache not in TORCH_CACHE_MODES:
        raise ValueError(
            f"lut_cache must be one of: {TORCH_CACHE_MODES}; got {cache!r}"
        )
    return cache  # type: ignore[return-value]


# Public operation names (used for compile pre-caching and targeted cache clears).
type TorchHilbertOp = Literal[
    "hilbert_encode_2d",
    "hilbert_decode_2d",
    "hilbert_encode_3d",
    "hilbert_decode_3d",
    "all",
]
"""Identifiers for Hilbert operations, including "all" option to target all operations."""


# (device_key, lut_name) -> tensor
_DEVICE_LUT_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _resolve_device(device: TorchDeviceLike) -> torch.device:
    if device is None:
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _device_key(device: torch.device) -> str:
    # Canonicalized by _resolve_device. `str(torch.device("cuda")) == "cuda"` is
    # intentionally avoided by resolving "cuda" to an indexed device.
    return str(device)


_OP_TO_LUT_NAMES_ALL: dict[TorchHilbertOp, tuple[str, ...]] = {
    "hilbert_encode_2d": (
        "lut_2d4b_b_qs_i64",
        "lut_2d4b_sb_sq_i16",
        "lut_2d7b_b_qs_i64",
    ),
    "hilbert_decode_2d": (
        "lut_2d4b_q_bs_i64",
        "lut_2d4b_sq_sb_i16",
        "lut_2d7b_q_bs_i64",
    ),
    "hilbert_encode_3d": ("lut_3d2b_sb_so_i16",),
    "hilbert_decode_3d": ("lut_3d2b_so_sb_i16",),
    "all": (
        "lut_2d4b_b_qs_i64",
        "lut_2d4b_q_bs_i64",
        "lut_2d4b_sb_sq_i16",
        "lut_2d4b_sq_sb_i16",
        "lut_2d7b_b_qs_i64",
        "lut_2d7b_q_bs_i64",
        "lut_3d2b_sb_so_i16",
        "lut_3d2b_so_sb_i16",
    ),
}


_OP_TO_LUT_NAMES_COMPILE: dict[TorchHilbertOp, tuple[str, ...]] = {
    # Under torch.compile, the 2D plain-torch kernels force the 4-bit tile.
    "hilbert_encode_2d": ("lut_2d4b_sb_sq_i16",),
    "hilbert_decode_2d": ("lut_2d4b_sq_sb_i16",),
    # 3D kernels always use the 2-bit LUTs.
    "hilbert_encode_3d": ("lut_3d2b_sb_so_i16",),
    "hilbert_decode_3d": ("lut_3d2b_so_sb_i16",),
    "all": (
        "lut_2d4b_sb_sq_i16",
        "lut_2d4b_sq_sb_i16",
        "lut_3d2b_sb_so_i16",
        "lut_3d2b_so_sb_i16",
    ),
}


_COMPILE_LUT_NAME_TO_OP: dict[str, TorchHilbertOp] = {
    name: op
    for op, names in _OP_TO_LUT_NAMES_COMPILE.items()
    if op != "all"
    for name in names
}


def clear_torch_lut_caches(
    device: TorchDeviceLike = None, *, op: TorchHilbertOp = "all"
) -> None:
    """Clear Torch-side LUT caches.

    Parameters
    ----------
    device
        Device whose cached LUT tensors should be cleared.

        If ``None`` (default), clears cached LUT tensors for all devices.
    op
        Operation used to filter which cached LUT tensors are cleared.

        - ``"all"`` (default): clear all cached LUT tensors.
        - Otherwise: clear only the cached LUT tensors used by that
          operation (for example, ``"hilbert_encode_2d"``).


    Notes
    -----
    This does *not* clear the root process-wide LUT cache. Use
        [`clear_lut_caches`][hilbertsfc.clear_lut_caches] for that.
    """

    target_names: set[str] | None
    if op == "all":
        target_names = None
    else:
        try:
            target_names = set(_OP_TO_LUT_NAMES_ALL[op])
        except KeyError as e:
            raise ValueError(f"unknown op {op!r}") from e

    if device is None:
        if target_names is None:
            _DEVICE_LUT_CACHE.clear()
            return

        keys = [k for k in _DEVICE_LUT_CACHE.keys() if k[1] in target_names]
        for k in keys:
            _DEVICE_LUT_CACHE.pop(k, None)
        return

    dev_key = _device_key(_resolve_device(device))
    if target_names is None:
        keys = [k for k in _DEVICE_LUT_CACHE.keys() if k[0] == dev_key]
    else:
        keys = [
            k
            for k in _DEVICE_LUT_CACHE.keys()
            if k[0] == dev_key and k[1] in target_names
        ]
    for k in keys:
        _DEVICE_LUT_CACHE.pop(k, None)


def precache_compile_luts(
    device: TorchDeviceLike = None, *, op: TorchHilbertOp = "all"
) -> None:
    """Pre-cache Torch LUT tensors for use with ``torch.compile``.

    When using HilbertSFC Torch functions with ``torch.compile``, call this before
    compilation to avoid materializing LUT tensors inside the compiled region,
    which can cause graph breaks, extra overhead, and failure with
    ``fullgraph=True``.

    Parameters
    ----------
    device:
        Device for which to cache LUT tensors.

        ``None`` means CPU.

    op
        Operation used to select which LUT tensors are pre-cached.

        - ``"all"`` (default): pre-cache all LUT tensors needed for supported
          operations.
        - Otherwise: pre-cache only the LUT tensors used by that operation.

    Notes
    -----
    It is generally not useful to pre-cache LUT tensors with this function
    when not using ``torch.compile``, as this function materializes LUT tensors
    that may not be used outside compiled regions.
    """

    dev = _resolve_device(device)
    try:
        names = _OP_TO_LUT_NAMES_COMPILE[op]
    except KeyError as e:
        raise ValueError(f"unknown op {op!r}") from e

    for name in names:
        # Call the accessor by name; these are module-level functions.
        fn = globals().get(name)
        if not callable(fn):
            raise RuntimeError(f"internal error: missing LUT accessor {name!r}")
        fn(device=dev, cache="device")


def _compile_cache_miss_message(
    *,
    name: str,
    device: torch.device,
    cache: TorchCacheMode,
) -> str:
    op = _COMPILE_LUT_NAME_TO_OP.get(name, "all")
    precache = f"precache_compile_luts(device={device!r}, op={op!r})"

    if cache == "host_only":
        detail = (
            "`lut_cache='host_only'` rematerializes LUTs on every call, which can "
            "cause a graph break under `torch.compile`. Prefer "
            f"`lut_cache='device'` and call `{precache}` before the first compiled call. "
        )
    else:
        detail = (
            f"Prefer calling `{precache}` before the first compiled call to avoid "
            "cache misses during tracing. "
        )

    return (
        "HilbertSFC hit an uncached LUT while tracing with `torch.compile` "
        f"(op={op!r}, lut={name!r}, device={_device_key(device)!r}). "
        f"{detail}"
        "With `fullgraph=True`, LUTs must be pre-cached."
    )


def _build_with_compile_cache_miss_warning(
    *,
    name: str,
    device: torch.device,
    cache: TorchCacheMode,
    build: Callable[[], torch.Tensor],
) -> torch.Tensor:
    @torch.compiler.disable()
    def _warn_and_build() -> torch.Tensor:
        warnings.warn(
            _compile_cache_miss_message(name=name, device=device, cache=cache),
            RuntimeWarning,
            stacklevel=3,
        )
        return build()

    return _warn_and_build()


def _cached_tensor(
    *,
    name: str,
    device: torch.device,
    cache: TorchCacheMode,
    build: Callable[[], torch.Tensor],
) -> torch.Tensor:
    cache = validate_torch_cache_mode(cache)

    key: tuple[str, str] | None = None
    if cache == "device":
        key = (_device_key(device), name)
        t = _DEVICE_LUT_CACHE.get(key)
        if t is not None:
            return t

    if torch.compiler.is_compiling():
        t = _build_with_compile_cache_miss_warning(
            name=name,
            device=device,
            cache=cache,
            build=build,
        )
    else:
        t = build()

    if key is not None:
        _DEVICE_LUT_CACHE[key] = t
    return t


def _lut_u64_as_i64(arr, *, device: torch.device) -> torch.Tensor:
    # Use int64 to support bitwise ops and indexing; values may exceed int64
    # sign bit, but kernels always mask low bits after shifting.
    # Use `torch.tensor` to force a copy so PyTorch doesn't warn about
    # non-writable NumPy buffers.
    return torch.tensor(arr, device=device, dtype=torch.int64)


def _lut_u16_as_i16(arr, *, device: torch.device) -> torch.Tensor:
    # Stored as uint16 on disk; use int16 in torch for better support.
    return torch.tensor(arr, device=device, dtype=torch.int16)


# --- 2D compacted (uint64 on disk, exposed as int64) ---


def lut_2d4b_b_qs_i64(
    *, device: TorchDeviceLike = None, cache: TorchCacheMode = "device"
) -> torch.Tensor:
    """Torch tensor view of [`lut_2d4b_b_qs_u64`][hilbertsfc._luts.lut_2d4b_b_qs_u64] (as int64)."""

    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u64_as_i64(_luts.lut_2d4b_b_qs_u64(), device=dev)

    return _cached_tensor(
        name="lut_2d4b_b_qs_i64",
        device=dev,
        cache=cache,
        build=_build,
    )


def lut_2d4b_q_bs_i64(
    *, device: TorchDeviceLike = None, cache: TorchCacheMode = "device"
) -> torch.Tensor:
    """Torch tensor view of [`lut_2d4b_q_bs_u64`][hilbertsfc._luts.lut_2d4b_q_bs_u64] (as int64)."""

    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u64_as_i64(_luts.lut_2d4b_q_bs_u64(), device=dev)

    return _cached_tensor(
        name="lut_2d4b_q_bs_i64",
        device=dev,
        cache=cache,
        build=_build,
    )


def lut_2d7b_b_qs_i64(
    *, device: TorchDeviceLike = None, cache: TorchCacheMode = "device"
) -> torch.Tensor:
    """Torch tensor view of [`lut_2d7b_b_qs_u64`][hilbertsfc._luts.lut_2d7b_b_qs_u64] (as int64)."""

    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u64_as_i64(_luts.lut_2d7b_b_qs_u64(), device=dev)

    return _cached_tensor(
        name="lut_2d7b_b_qs_i64",
        device=dev,
        cache=cache,
        build=_build,
    )


def lut_2d7b_q_bs_i64(
    *, device: TorchDeviceLike = None, cache: TorchCacheMode = "device"
) -> torch.Tensor:
    """Torch tensor view of [`lut_2d7b_q_bs_u64`][hilbertsfc._luts.lut_2d7b_q_bs_u64] (as int64)."""

    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u64_as_i64(_luts.lut_2d7b_q_bs_u64(), device=dev)

    return _cached_tensor(
        name="lut_2d7b_q_bs_i64",
        device=dev,
        cache=cache,
        build=_build,
    )


def lut_2d4b_sb_sq_i16(
    *, device: TorchDeviceLike = None, cache: TorchCacheMode = "device"
) -> torch.Tensor:
    """Torch tensor view of [`lut_2d4b_sb_sq_u16`][hilbertsfc._luts.lut_2d4b_sb_sq_u16] (as int16)."""

    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u16_as_i16(_luts.lut_2d4b_sb_sq_u16(), device=dev)

    return _cached_tensor(
        name="lut_2d4b_sb_sq_i16",
        device=dev,
        cache=cache,
        build=_build,
    )


def lut_2d4b_sq_sb_i16(
    *, device: TorchDeviceLike = None, cache: TorchCacheMode = "device"
) -> torch.Tensor:
    """Torch tensor view of [`lut_2d4b_sq_sb_u16`][hilbertsfc._luts.lut_2d4b_sq_sb_u16] (as int16)."""

    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u16_as_i16(_luts.lut_2d4b_sq_sb_u16(), device=dev)

    return _cached_tensor(
        name="lut_2d4b_sq_sb_i16",
        device=dev,
        cache=cache,
        build=_build,
    )


# --- 3D 2-bit flat (uint16 on disk, exposed as int16) ---


def lut_3d2b_sb_so_i16(
    *,
    device: TorchDeviceLike = None,
    cache: TorchCacheMode = "device",
) -> torch.Tensor:
    """3D 2-bit LUT (state, bb) -> packed (next_state, oo)."""
    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u16_as_i16(_luts.lut_3d2b_sb_so(), device=dev)

    return _cached_tensor(
        name="lut_3d2b_sb_so_i16", device=dev, cache=cache, build=_build
    )


def lut_3d2b_so_sb_i16(
    *,
    device: TorchDeviceLike = None,
    cache: TorchCacheMode = "device",
) -> torch.Tensor:
    """3D 2-bit LUT (state, oo) -> packed (next_state, bb)."""
    dev = _resolve_device(device)

    def _build() -> torch.Tensor:
        return _lut_u16_as_i16(_luts.lut_3d2b_so_sb(), device=dev)

    return _cached_tensor(
        name="lut_3d2b_so_sb_i16", device=dev, cache=cache, build=_build
    )
