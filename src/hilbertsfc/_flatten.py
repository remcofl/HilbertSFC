"""Internal array helpers.

These helpers are performance-sensitive and shared between the 2D/3D public APIs.
"""

from typing import Literal

import numpy as np

OrderCFA = Literal["C", "F", "A"]


def flatten_nocopy(
    arr: np.ndarray,
    name: str,
    *,
    order: OrderCFA = "C",
    strict: bool = True,
) -> np.ndarray:
    """Return a 1D view of ``arr`` without copying.

    Parameters
    ----------
    order
        Reshape order to attempt. Default is ``"C"``.
    strict
        If ``True`` (default), raise when a zero-copy 1D view is not possible.
        If ``False``, return ``arr`` unchanged on failure.

    Notes
    -----
    - Uses NumPy's ``copy=False`` (NumPy >= 2.0) when available.
    - On older NumPy versions, falls back to ``reshape`` and verifies that the
      result shares memory (rejecting implicit copies).
    """

    message = (
        f"{name} must support a zero-copy 1D view (no implicit copies). "
        f"Got shape={arr.shape}, strides={arr.strides}, dtype={arr.dtype}. "
        "Hint: for guaranteed zero-copy flattening to 1D, pass an already-1D array "
        "or make it contiguous first (e.g. `np.ascontiguousarray(...)` or `np.asfortranarray(...)`)."
    )

    try:
        return arr.reshape(-1, order=order, copy=False)
    except TypeError:
        # NumPy < 2.0: no `copy=` argument.
        reshaped = arr.reshape(-1, order=order)
        if arr.size == 0:
            return reshaped
        if not np.shares_memory(reshaped, arr):
            if strict:
                raise ValueError(message) from None
            return arr
        return reshaped
    except ValueError as e:
        if strict:
            raise ValueError(message) from e
        return arr
