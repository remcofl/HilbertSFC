import numpy as np
import pytest

from hilbertsfc._dtype import (
    choose_lut_dtype_for_index_dtype,
    choose_uint_coord_dtype,
    choose_uint_index_dtype,
    dtype_effective_bits,
    max_nbits_for_index_dtype,
    unsigned_view,
)


def test_dtype_effective_bits_unsigned_and_signed() -> None:
    assert dtype_effective_bits(np.dtype(np.uint8)) == 8
    assert dtype_effective_bits(np.dtype(np.uint16)) == 16
    assert dtype_effective_bits(np.dtype(np.uint64)) == 64

    assert dtype_effective_bits(np.dtype(np.int8)) == 7
    assert dtype_effective_bits(np.dtype(np.int16)) == 15
    assert dtype_effective_bits(np.dtype(np.int64)) == 63


def test_dtype_effective_bits_rejects_non_integer() -> None:
    with pytest.raises(TypeError):
        dtype_effective_bits(np.dtype(np.float32))


def test_max_nbits_for_index_dtype() -> None:
    assert max_nbits_for_index_dtype(np.dtype(np.uint16), dims=2) == 8
    assert max_nbits_for_index_dtype(np.dtype(np.uint32), dims=2) == 16
    assert max_nbits_for_index_dtype(np.dtype(np.uint64), dims=3) == 21

    with pytest.raises(ValueError):
        max_nbits_for_index_dtype(np.dtype(np.uint64), dims=0)


def test_unsigned_view_signed_is_zero_copy_and_unsigned() -> None:
    arr = np.array([0, 1, 2, 3], dtype=np.int16)
    view = unsigned_view(arr)

    assert view.dtype == np.dtype(np.uint16)
    assert view.base is arr
    # Same bytes in memory
    assert view.__array_interface__["data"][0] == arr.__array_interface__["data"][0]


def test_unsigned_view_unsigned_returns_same_array() -> None:
    arr = np.array([0, 1, 2], dtype=np.uint16)
    view = unsigned_view(arr)
    assert view is arr


@pytest.mark.parametrize(
    ("nbits", "dims", "expected"),
    [
        (1, 2, np.uint8),
        (4, 2, np.uint8),
        (8, 2, np.uint16),
        (9, 2, np.uint32),
        (16, 2, np.uint32),
        (17, 2, np.uint64),
        (5, 3, np.uint16),
        (10, 3, np.uint32),
        (21, 3, np.uint64),
    ],
)
def test_choose_uint_index_dtype(
    nbits: int, dims: int, expected: type[np.unsignedinteger]
) -> None:
    assert choose_uint_index_dtype(nbits=nbits, dims=dims) == expected


def test_choose_uint_index_dtype_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        choose_uint_index_dtype(nbits=0, dims=2)
    with pytest.raises(ValueError):
        choose_uint_index_dtype(nbits=1, dims=0)
    with pytest.raises(ValueError):
        choose_uint_index_dtype(nbits=22, dims=3)  # 66 bits


@pytest.mark.parametrize(
    ("nbits", "expected"),
    [
        (1, np.uint8),
        (8, np.uint8),
        (9, np.uint16),
        (16, np.uint16),
        (17, np.uint32),
        (32, np.uint32),
    ],
)
def test_choose_uint_coord_dtype(
    nbits: int, expected: type[np.unsignedinteger]
) -> None:
    assert choose_uint_coord_dtype(nbits=nbits) == expected


def test_choose_uint_coord_dtype_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        choose_uint_coord_dtype(nbits=0)
    with pytest.raises(ValueError):
        choose_uint_coord_dtype(nbits=33)


def test_choose_lut_dtype_for_index_dtype() -> None:
    assert choose_lut_dtype_for_index_dtype(np.dtype(np.uint16)) is np.uint16
    assert choose_lut_dtype_for_index_dtype(np.dtype(np.uint32)) is np.uint32
    assert choose_lut_dtype_for_index_dtype(np.dtype(np.uint64)) is np.uint64

    # Signed dtype has one fewer effective bit
    assert choose_lut_dtype_for_index_dtype(np.dtype(np.int16)) is np.uint16
