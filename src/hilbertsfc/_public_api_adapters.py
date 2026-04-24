"""Typed adapter objects for shared public API wrappers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

type BuildEncode2DScalar = Callable[[int], Encode2DScalarKernel]
type BuildDecode2DScalar = Callable[[int], Decode2DScalarKernel]
type BuildEncode3DScalar = Callable[[int], Encode3DScalarKernel]
type BuildDecode3DScalar = Callable[[int], Decode3DScalarKernel]

type BuildEncode2DBatch = Callable[[int, bool], Encode2DBatchKernel]
type BuildDecode2DBatch = Callable[[int, bool], Decode2DBatchKernel]
type BuildEncode3DBatch = Callable[[int, bool, np.dtype[Any]], Encode3DBatchKernel]
type BuildDecode3DBatch = Callable[[int, bool, np.dtype[Any]], Decode3DBatchKernel]


@dataclass(frozen=True)
class Encode2DAdapter:
    build_scalar: BuildEncode2DScalar
    build_batch: BuildEncode2DBatch


@dataclass(frozen=True)
class Decode2DAdapter:
    build_scalar: BuildDecode2DScalar
    build_batch: BuildDecode2DBatch


@dataclass(frozen=True)
class Encode3DAdapter:
    build_scalar: BuildEncode3DScalar
    build_batch: BuildEncode3DBatch


@dataclass(frozen=True)
class Decode3DAdapter:
    build_scalar: BuildDecode3DScalar
    build_batch: BuildDecode3DBatch
