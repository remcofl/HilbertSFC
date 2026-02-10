"""Input shape/type checks used by the public API.

These helpers keep scalar-vs-array dispatch consistent across modules.
"""

from __future__ import annotations

import numpy as np


def is_scalar_int(x: object) -> bool:
    """True for Python/NumPy integer scalars, but not bool."""

    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def is_0d_int_array(x: object) -> bool:
    """True for NumPy 0-D integer arrays (e.g. np.array(1, dtype=np.uint32))."""

    return (
        isinstance(x, np.ndarray)
        and x.shape == ()
        and np.issubdtype(x.dtype, np.integer)
    )


def is_int_scalar_or_0d_array(x: object) -> bool:
    return is_scalar_int(x) or is_0d_int_array(x)
