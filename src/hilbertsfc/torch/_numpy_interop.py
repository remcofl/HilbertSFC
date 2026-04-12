"""Internal Torch/NumPy compatibility helpers.

This module is imported by the torch frontend. It assumes torch is installed.
"""

import numpy as np
import torch

from ._tensor_int import require_int_tensor


def int_tensor_to_numpy_view(x: torch.Tensor, name: str) -> np.ndarray:
    """Return a zero-copy NumPy view of a CPU integer tensor.

    The returned array shares memory with the tensor (writes are reflected both
    ways) and may be non-contiguous (strided).

    Raises
    ------
    TypeError
        If ``x`` is not an integer tensor.
    ValueError
        If ``x`` is not on CPU or if ``x.numpy()`` cannot provide a zero-copy view.
    """

    require_int_tensor(x, name)

    # This implies no grad.

    if x.device.type != "cpu":
        raise ValueError(
            f"{name} must be on CPU for NumPy fallback; got device={x.device}"
        )

    try:
        return x.numpy()
    except Exception as e:
        raise ValueError(
            f"{name} must support a zero-copy NumPy view via `.numpy()`; "
            f"got dtype={x.dtype}, device={x.device}, shape={tuple(x.shape)}, "
            f"strides={tuple(x.stride())}, layout={x.layout}, is_contiguous={x.is_contiguous()}. "
            "Hint: ensure it's a CPU strided tensor; try `.contiguous()` if needed. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
