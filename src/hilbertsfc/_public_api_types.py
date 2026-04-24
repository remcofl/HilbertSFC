"""Type aliases for shared public API wrappers."""

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from .types import IntScalar, UIntArray

type Encode2DScalarKernel = Callable[[IntScalar, IntScalar], int]
type Decode2DScalarKernel = Callable[[IntScalar], tuple[int, int]]
type Encode3DScalarKernel = Callable[[IntScalar, IntScalar, IntScalar], int]
type Decode3DScalarKernel = Callable[[IntScalar], tuple[int, int, int]]

type Encode2DBatchKernel = Callable[[UIntArray, UIntArray, UIntArray], None]
type Decode2DBatchKernel = Callable[[UIntArray, UIntArray, UIntArray], None]
type Encode3DBatchKernel = Callable[[UIntArray, UIntArray, UIntArray, UIntArray], None]
type Decode3DBatchKernel = Callable[
    [UIntArray, UIntArray, UIntArray, UIntArray],
    None,
]


class BuildEncode2DScalar(Protocol):
    def __call__(self, nbits: int) -> Encode2DScalarKernel: ...


class BuildDecode2DScalar(Protocol):
    def __call__(self, nbits: int) -> Decode2DScalarKernel: ...


class BuildEncode3DScalar(Protocol):
    def __call__(self, nbits: int) -> Encode3DScalarKernel: ...


class BuildDecode3DScalar(Protocol):
    def __call__(self, nbits: int) -> Decode3DScalarKernel: ...


class BuildEncode2DBatch(Protocol):
    def __call__(
        self, nbits: int, *, parallel: bool = False
    ) -> Encode2DBatchKernel: ...


class BuildDecode2DBatch(Protocol):
    def __call__(
        self, nbits: int, *, parallel: bool = False
    ) -> Decode2DBatchKernel: ...


class BuildEncode3DBatch(Protocol):
    def __call__(
        self,
        nbits: int,
        *,
        parallel: bool = False,
        index_dtype: np.dtype[Any],
    ) -> Encode3DBatchKernel: ...


class BuildDecode3DBatch(Protocol):
    def __call__(
        self,
        nbits: int,
        *,
        parallel: bool = False,
        index_dtype: np.dtype[Any],
    ) -> Decode3DBatchKernel: ...
