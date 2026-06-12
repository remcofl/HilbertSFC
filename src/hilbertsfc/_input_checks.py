"""Input shape/type checks used by the public API.

These helpers keep scalar-vs-array dispatch consistent across modules.
"""

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


def require_int_array(x: object, name: str) -> np.ndarray:
    """Return an integer NumPy array or raise a public-API-friendly error."""

    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a NumPy integer array")
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype; got {x.dtype!r}")
    return x


def reject_obvious_array_memory_overlap(
    output: np.ndarray,
    output_name: str,
    *others: tuple[np.ndarray, str],
) -> None:
    """Reject output arrays when memory overlap is cheaply and definitely detected.

    This is intentionally conservative: it first uses a cheap bounds check to
    rule out impossible overlap, then asks NumPy for a bounded exact overlap
    check. If NumPy cannot decide within the work limit, overlap is treated as
    unproven and not rejected.
    """

    for other, other_name in others:
        if not np.may_share_memory(output, other):
            continue

        try:
            overlaps = np.shares_memory(output, other, max_work=1000)  # type: ignore[reportArgumentType]
        except np.exceptions.TooHardError:
            overlaps = False

        if overlaps:
            raise ValueError(f"{output_name} must not overlap {other_name}")
