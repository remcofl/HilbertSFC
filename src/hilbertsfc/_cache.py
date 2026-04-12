"""Cache helpers for hilbertsfc.

Centralizes kernel builder caching so:
- all builders share a consistent maxsize
- caches can be cleared in one call (useful for benchmarks/tests)
"""

from collections.abc import Callable
from functools import cache, lru_cache
from threading import RLock
from typing import cast

_KERNEL_CACHE_MAXSIZE = 64

_KERNEL_CACHES: list[Callable[[], None]] = []
_LUT_CACHES: list[Callable[[], None]] = []

_CACHE_REGISTRY_LOCK = RLock()


def register_cache_clear(clear: Callable[[], None], *, kind: str) -> None:
    """Register a cache-clear function.

    Parameters
    ----------
    clear:
        Callable that clears the cache.
    kind:
        One of ``"kernel"`` or ``"lut"``.
    """

    with _CACHE_REGISTRY_LOCK:
        if kind == "kernel":
            _KERNEL_CACHES.append(clear)
        elif kind == "lut":
            _LUT_CACHES.append(clear)
        else:
            raise ValueError(f"Unknown cache kind: {kind!r}")


def kernel_cache[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    """Decorate a builder function with an LRU cache and register it."""

    wrapped = lru_cache(maxsize=_KERNEL_CACHE_MAXSIZE)(func)
    register_cache_clear(wrapped.cache_clear, kind="kernel")
    return cast(Callable[P, T], wrapped)


def lut_cache[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    """Decorate a LUT accessor with a process-wide cache and register it."""

    wrapped = cache(func)
    register_cache_clear(wrapped.cache_clear, kind="lut")
    return cast(Callable[P, T], wrapped)


def clear_kernel_caches() -> None:
    """Clear all registered kernel builder caches."""

    with _CACHE_REGISTRY_LOCK:
        clears = list(_KERNEL_CACHES)
    for clear in clears:
        clear()


def clear_lut_caches() -> None:
    """Clear all registered LUT caches.

    Notes
    -----
    This does *not* clear torch-side device LUT caches; for those, see
    [`hilbertsfc.torch.clear_torch_lut_caches`][hilbertsfc.torch.clear_torch_lut_caches].
    """

    with _CACHE_REGISTRY_LOCK:
        clears = list(_LUT_CACHES)
    for clear in clears:
        clear()


def clear_all_caches() -> None:
    """Clear LUT caches and kernel builder caches.

    Notes
    -----
    This does *not* clear torch-side device LUT caches; for those, see
    [`hilbertsfc.torch.clear_torch_lut_caches`][hilbertsfc.torch.clear_torch_lut_caches].
    """

    clear_lut_caches()
    clear_kernel_caches()
