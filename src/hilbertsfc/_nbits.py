"""Low-overhead `nbits` validation helpers."""

MAX_NBITS_2D = 32
MAX_NBITS_3D = 21


def validate_nbits_2d(nbits: int) -> None:
    if not (1 <= nbits <= MAX_NBITS_2D):
        raise ValueError("nbits must be in range [1..32]")


def validate_nbits_3d(nbits: int) -> None:
    if not (1 <= nbits <= MAX_NBITS_3D):
        # 3D indices grow as 3*nbits; uint64 fits up to 21 bits per coord.
        raise ValueError("nbits must be in range [1..21]")
