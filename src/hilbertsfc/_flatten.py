"""Internal array helpers.

These helpers are performance-sensitive and shared between the 2D/3D public APIs.
"""

from __future__ import annotations

import numpy as np


def flatten_nocopy(arr: np.ndarray, name: str) -> np.ndarray:
    """Return a 1D view of `arr` without copying.

    Uses NumPy's `copy=False` (NumPy >= 2.0) when available. On older NumPy
    versions, falls back to `reshape(order="A")` and rejects if it produced a
    copy.

    This is primarily used to adapt arbitrary-shaped inputs/outputs to the
    underlying 1D Numba kernels.
    """

    message = (
        f"{name} must support a zero-copy 1D view (no implicit copies). "
        f"Got shape={arr.shape}, strides={arr.strides}, dtype={arr.dtype}. "
        "Hint: for guaranteed zero-copy flattening to 1D, pass an already-1D array "
        "or make it contiguous first (e.g. `np.ascontiguousarray(...)` or `np.asfortranarray(...)`)."
    )

    try:
        return arr.reshape(-1, order="A", copy=False)
    except TypeError:
        # NumPy < 2.0: no `copy=` argument.
        reshaped = arr.reshape(-1, order="A")
        if arr.size == 0:
            return reshaped
        if not np.shares_memory(reshaped, arr):
            raise ValueError(message) from None
        return reshaped
    except ValueError as e:
        raise ValueError(message) from e
