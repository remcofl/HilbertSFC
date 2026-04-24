"""Torch 2D Hilbert API dispatch layer."""

import torch

from ._dispatch import (
    get_hilbert_decode_2d_numba,
    get_hilbert_decode_2d_triton,
    get_hilbert_encode_2d_numba,
    get_hilbert_encode_2d_triton,
)
from ._dispatch_common import CPUBackend, GPUBackend
from ._kernels.torch.hilbert2d_decode import hilbert_decode_2d_torch
from ._kernels.torch.hilbert2d_encode import hilbert_encode_2d_torch
from ._luts import TorchCacheMode, validate_torch_cache_mode
from ._public_api_shared_2d import decode_2d_api, encode_2d_api
from ._tuning_mode import TritonTuningMode


def hilbert_encode_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    nbits: int | None = None,
    out: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> torch.Tensor:
    """Encode 2D integer coordinates to Hilbert indices.

    This function provides a PyTorch equivalent of
    [`hilbert_encode_2d`][hilbertsfc.hilbert2d.hilbert_encode_2d]. It accepts
    integer ``torch.Tensor`` of arbitrary shape on any device, and dispatches
    to backend-specific implementations depending on device and backend settings.

    Parameters
    ----------
    x, y
        Integer coordinate tensors to encode.

        Must have identical ``shape`` and be on the same device.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis. For inputs outside that domain, only the low
        ``nbits`` bits of each coordinate are used.

        Must satisfy ``1 <= nbits <= 32``. If provided, it must also fit within
        the usable bits of the coordinate dtype.

        If ``None``:

        - Array mode: inferred from the coordinate dtype using its usable bit
        width, capped at 32. For example, ``uint16`` -> 16, ``int16`` -> 15
        (sign bit excluded), and ``uint64``/``int64`` -> 32.
        - Scalar mode: defaults to 32.

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input coordinate range.
    out
        Optional output tensor.

        Must have the same shape and device as ``x`` and ``y`` and an integer
        dtype wide enough to hold ``2 * nbits`` bits.
    lut_cache
        Cache mode for look-up tables (LUTs) used by the Torch/Triton kernels.

        - ``"device"`` (default): cache the converted LUT tensors per-device
            for reuse across calls.
        - ``"host_only"``: do not keep a torch-side LUT cache; materialize on
            demand from the (process-wide) NumPy LUT cache.

        This setting is ignored by the CPU Numba path.
    cpu_parallel
        Controls whether the CPU Numba kernel may execute in parallel.

        Only applies when dispatching to the CPU Numba backend and the input is
        not a scalar tensor. If ``None``, a heuristic is used.
    cpu_backend
        CPU backend selection.

        - ``"auto"`` (default): use the Numba kernel unless inside ``torch.compile``,
            in which case the torch backend is used.
        - ``"numba"``: always use the Numba kernel. This mode is not
            ``torch.compile``-friendly.
        - ``"torch"``: always use the torch implementation.
    gpu_backend
        GPU (accelerator) backend selection.

        - ``"auto"`` (default): on CUDA, use the Triton kernel when available and
            all tensors are contiguous; otherwise fall back to the Torch kernel.
            Fallbacks due to non-contiguity or Triton runtime failure emit a
            ``UserWarning``.
        - ``"triton"``: force the Triton kernel. Requires CUDA tensors,
            Triton availability, and contiguous inputs/outputs;
            raises on violation or kernel failure.
        - ``"torch"``: force the Torch implementation.
    triton_tuning
        Triton launch config selection policy.

        - ``"heuristic"`` (default): use static launch heuristics.
        - ``"autotune_bucketed"``: autotune from a fixed config set and cache by
        input size bucket.
        - ``"autotune_exact"``: autotune from the same config set and cache by
        exact input size.

        Only applies when the Triton backend is used.

    Returns
    -------
    torch.Tensor
        Hilbert indices.

        - Has the same shape/device as the inputs.
        - If ``out`` is provided, returns ``out``.
        - Otherwise, chooses a minimal integer dtype that can represent
            ``2 * nbits`` bits, preferring unsigned if all inputs are unsigned
            and a fitting unsigned dtype is available.

    Raises
    ------
    TypeError
        If a non-integer tensor is provided.
    ValueError
        If inputs are on different devices, have mismatched shapes, if ``nbits``
        is invalid or does not fit in the input/output dtypes, or if backend
        arguments are invalid.
    RuntimeError
        If ``gpu_backend='triton'`` is requested but Triton is unavailable or the
        Triton kernel fails at runtime.

    Notes
    -----
    When using this function with ``torch.compile``, call
    [`precache_compile_luts`][hilbertsfc.torch._luts.precache_compile_luts]
    before compilation. This avoids materialization of LUTs inside the
    compiled region, which causes graph breaks, extra overhead, and failure
    with ``fullgraph=True``.
    """

    lut_cache = validate_torch_cache_mode(lut_cache)

    def torch_kernel(*args, **kwargs):
        return hilbert_encode_2d_torch(*args, **kwargs, lut_cache=lut_cache)

    def get_triton():
        triton_kernel = get_hilbert_encode_2d_triton()

        def kernel(*args, **kwargs):
            return triton_kernel(*args, **kwargs, lut_cache=lut_cache)

        return kernel

    return encode_2d_api(
        x,
        y,
        nbits=nbits,
        out=out,
        cpu_parallel=cpu_parallel,
        cpu_backend=cpu_backend,
        gpu_backend=gpu_backend,
        triton_tuning=triton_tuning,
        torch_kernel=torch_kernel,
        get_numba=get_hilbert_encode_2d_numba,
        get_triton=get_triton,
    )


def hilbert_decode_2d(
    index: torch.Tensor,
    *,
    nbits: int | None = None,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode Hilbert indices to 2D integer coordinates.

    This function provides a PyTorch equivalent of
    [`hilbert_decode_2d`][hilbertsfc.hilbert2d.hilbert_decode_2d]. It accepts
    integer ``torch.Tensor`` of arbitrary shape on any device, and dispatches
    to backend-specific implementations depending on device and backend settings.

    Parameters
    ----------
    index
        Integer Hilbert index tensor to decode.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis and a Hilbert index range of
        ``[0, 2**(2 * nbits))``. For indices outside that range, only the low
        ``2 * nbits`` bits are used.

        Must satisfy ``1 <= nbits <= 32``. If provided, it must also fit within
        the usable bits of the index dtype.

        If ``None``:

        - Inferred from the index dtype as half its usable bit width, capped at 32.
        For example, ``uint16`` -> 8, ``uint64`` -> 32, and ``int64`` -> 31
        (sign bit excluded).

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input index range.
    out_x, out_y
        Optional output coordinate tensors. Either provide both or neither.

        Each must have the same shape and device as ``index`` and an integer
        dtype wide enough to hold ``nbits`` bits.
    lut_cache
        Cache mode for look-up tables (LUTs) used by the Torch/Triton kernels.

        - ``"device"`` (default): cache the converted LUT tensors per-device
        for reuse across calls.
        - ``"host_only"``: do not keep a torch-side LUT cache; materialize on
        demand from the (process-wide) NumPy LUT cache.

        This setting is ignored by the CPU Numba path.
    cpu_parallel
        Controls whether the CPU Numba kernel may execute in parallel.

        Only applies when dispatching to the CPU Numba backend and the input is
        not a scalar tensor. If ``None``, a heuristic is used.
    cpu_backend
        CPU backend selection.

        - ``"auto"`` (default): use the Numba kernel unless inside ``torch.compile``,
        in which case the torch backend is used.
        - ``"numba"``: always use the Numba kernel. This mode is not
        ``torch.compile``-friendly.
        - ``"torch"``: always use the torch implementation.
    gpu_backend
        GPU (accelerator) backend selection.

        - ``"auto"`` (default): on CUDA, use the Triton kernel when available and
        all tensors are contiguous; otherwise fall back to the Torch kernel.
        Fallbacks due to non-contiguity or Triton runtime failure emit a
        ``UserWarning``.
        - ``"triton"``: require CUDA tensors, Triton availability, and contiguous
        inputs/outputs; raises on violation or kernel failure.
        - ``"torch"``: force the Torch implementation.
    triton_tuning
        Triton launch config selection policy.

        - ``"heuristic"`` (default): use static launch heuristics.
        - ``"autotune_bucketed"``: autotune from a fixed config set and cache by
        input size bucket.
        - ``"autotune_exact"``: autotune from the same config set and cache by
        exact input size.

        Only applies when the Triton backend is used.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Decoded coordinates ``(x, y)``.

        - Each tensor has the same shape/device as ``index``.
        - If ``out_x`` and ``out_y`` are provided, returns ``(out_x, out_y)``.
        - Otherwise, each result uses a minimal integer dtype that can represent
        ``nbits`` bits, preferring unsigned if the input is unsigned
        and a fitting unsigned dtype is available.

    Raises
    ------
    TypeError
        If a non-integer tensor is provided.
    ValueError
        If ``nbits`` is invalid or does not fit in the input/output dtypes, if
        outputs are inconsistent or have incorrect shapes/devices, or if backend
        arguments are invalid.
    RuntimeError
        If ``gpu_backend='triton'`` is requested but Triton is unavailable or the
        Triton kernel fails at runtime.

    Notes
    -----
    When using this function with ``torch.compile``, call
    [`precache_compile_luts`][hilbertsfc.torch._luts.precache_compile_luts]
    before compilation. This avoids materialization of LUTs inside the
    compiled region, which causes graph breaks, extra overhead, and failure
    with ``fullgraph=True``.
    """

    lut_cache = validate_torch_cache_mode(lut_cache)

    def torch_kernel(*args, **kwargs):
        return hilbert_decode_2d_torch(*args, **kwargs, lut_cache=lut_cache)

    def get_triton():
        triton_kernel = get_hilbert_decode_2d_triton()

        def kernel(*args, **kwargs):
            return triton_kernel(*args, **kwargs, lut_cache=lut_cache)

        return kernel

    return decode_2d_api(
        index,
        nbits=nbits,
        out_x=out_x,
        out_y=out_y,
        cpu_parallel=cpu_parallel,
        cpu_backend=cpu_backend,
        gpu_backend=gpu_backend,
        triton_tuning=triton_tuning,
        torch_kernel=torch_kernel,
        get_numba=get_hilbert_decode_2d_numba,
        get_triton=get_triton,
    )
