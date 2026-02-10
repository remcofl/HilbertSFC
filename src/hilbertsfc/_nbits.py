"""Low-overhead `nbits` validation helpers."""

from __future__ import annotations


def validate_nbits_2d(nbits: int) -> None:
    if not (1 <= nbits <= 32):
        raise ValueError("nbits must be in range [1..32]")


def validate_nbits_3d(nbits: int) -> None:
    if not (1 <= nbits <= 21):
        # 3D indices grow as 3*nbits; uint64 fits up to 21 bits per coord.
        raise ValueError("nbits must be in range [1..21]")
