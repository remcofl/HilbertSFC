"""Types for hilbertsfc.

Type aliases used throughout the public API to keep function signatures readable.

"""

import numpy as np
from numpy.typing import NDArray

type IntScalar = int | np.integer
"""Scalar integer input type.

Accepts a Python ``int`` or a NumPy integer scalar (e.g. ``np.uint32(1)``).
"""

type UIntArray = NDArray[np.unsignedinteger]
"""NumPy unsigned integer array."""

type IntArray = NDArray[np.integer]
"""NumPy integer array (signed or unsigned integer dtype)."""

type UInt8Array = NDArray[np.uint8]
"""NumPy ``uint8`` array."""

type UInt16Array = NDArray[np.uint16]
"""NumPy ``uint16`` array."""

type UInt32Array = NDArray[np.uint32]
"""NumPy ``uint32`` array."""

type UInt64Array = NDArray[np.uint64]
"""NumPy ``uint64`` array."""

# Supported LUT dtypes: keep this narrow so we can reject accidental signed types.
type LutUIntType = type[np.uint16] | type[np.uint32] | type[np.uint64]
"""LUT dtype as a *type* (``np.uint16``/``np.uint32``/``np.uint64``)."""

type LutUIntDType = np.dtype[np.uint16] | np.dtype[np.uint32] | np.dtype[np.uint64]
"""LUT dtype as a ``numpy.dtype`` instance."""

type LutUIntDTypeLike = LutUIntType | LutUIntDType
"""LUT dtype accepted by LUT/kernel builders.

Allows either the dtype *type* (e.g. ``np.uint32``) or a ``numpy.dtype`` instance
(e.g. ``np.dtype(np.uint32)``).
"""
