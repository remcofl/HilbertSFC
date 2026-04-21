from collections.abc import Callable

from hilbertsfc.types import IntArray


def get_hilbert_encode_2d_triton():
    from ._kernels.triton.hilbert2d_encode import hilbert_encode_2d_triton

    return hilbert_encode_2d_triton


def get_hilbert_encode_2d_numba() -> Callable[..., int | IntArray]:
    import torch

    from ..hilbert2d import hilbert_encode_2d

    return torch.compiler.disable(hilbert_encode_2d)  # type: ignore[return-value]


def get_hilbert_decode_2d_triton():
    from ._kernels.triton.hilbert2d_decode import hilbert_decode_2d_triton

    return hilbert_decode_2d_triton


def get_hilbert_decode_2d_numba() -> Callable[..., tuple[IntArray, IntArray]]:
    import torch

    from ..hilbert2d import hilbert_decode_2d

    return torch.compiler.disable(hilbert_decode_2d)  # type: ignore[return-value]


def get_hilbert_encode_3d_triton():
    from ._kernels.triton.hilbert3d_encode import hilbert_encode_3d_triton

    return hilbert_encode_3d_triton


def get_hilbert_encode_3d_numba() -> Callable[..., int | IntArray]:
    import torch

    from ..hilbert3d import hilbert_encode_3d

    return torch.compiler.disable(hilbert_encode_3d)  # type: ignore[return-value]


def get_hilbert_decode_3d_triton():
    from ._kernels.triton.hilbert3d_decode import hilbert_decode_3d_triton

    return hilbert_decode_3d_triton


def get_hilbert_decode_3d_numba() -> Callable[..., tuple[IntArray, IntArray, IntArray]]:
    import torch

    from ..hilbert3d import hilbert_decode_3d

    return torch.compiler.disable(hilbert_decode_3d)  # type: ignore[return-value]
