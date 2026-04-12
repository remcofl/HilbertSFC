"""Torch 2D Hilbert API dispatch layer."""

import warnings
from typing import cast

import torch

from .._nbits import MAX_NBITS_2D, validate_nbits_2d
from ._dispatch import (
    get_hilbert_decode_2d_numba,
    get_hilbert_decode_2d_triton,
    get_hilbert_encode_2d_numba,
    get_hilbert_encode_2d_triton,
)
from ._dispatch_common import (
    CPUBackend,
    GPUBackend,
    attempt_run_triton,
    choose_coord_torch_dtype,
    choose_index_torch_dtype,
    effective_bits_torch_dtype,
    max_nbits_for_torch_index_dtype,
    resolve_cpu_parallel,
    validate_cpu_backend,
    validate_gpu_backend,
)
from ._kernels.torch.hilbert2d_decode import hilbert_decode_2d_torch
from ._kernels.torch.hilbert2d_encode import (
    hilbert_encode_2d_torch,
)
from ._luts import TorchCacheMode
from ._numpy_interop import int_tensor_to_numpy_view
from ._tensor_int import (
    int_tensor_to_signed_view,
    is_uint_torch_dtype,
    require_int_tensor,
)


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
    cpu_backend = validate_cpu_backend(cpu_backend)
    gpu_backend = validate_gpu_backend(gpu_backend)

    if x.device != y.device:
        raise ValueError(
            f"x and y must be on the same device; got {x.device=}, {y.device=}"
        )
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape=}, {y.shape=}"
        )

    require_int_tensor(x, "x")
    require_int_tensor(y, "y")

    max_coord_nbits = max(
        effective_bits_torch_dtype(x.dtype), effective_bits_torch_dtype(y.dtype)
    )
    if nbits is None:
        nbits = max_coord_nbits
        if nbits > MAX_NBITS_2D:
            warnings.warn(
                f"The maximum effective bits of the coordinate dtype is {nbits}, "
                f"which exceeds the algorithm maximum of {MAX_NBITS_2D}. "
                f"Using nbits={MAX_NBITS_2D} instead. This means that excess bits in the input coordinates "
                f"will be ignored. To silence this warning, explicitly set nbits<={MAX_NBITS_2D}.",
                UserWarning,
                stacklevel=2,
            )
            nbits = MAX_NBITS_2D
    else:
        validate_nbits_2d(nbits)
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in coordinate dtypes; "
                f"got {x.dtype=} with {effective_bits_torch_dtype(x.dtype)} effective bits, "
                f"{y.dtype=} with {effective_bits_torch_dtype(y.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}."
            )

    out_provided = out is not None

    if not out_provided:
        prefer_uint = is_uint_torch_dtype(x.dtype) and is_uint_torch_dtype(y.dtype)
        auto_out_dtype = choose_index_torch_dtype(
            nbits=nbits,
            dims=2,
            prefer_unsigned=prefer_uint,
        )
        out = torch.empty(x.shape, dtype=auto_out_dtype, device=x.device)
    else:
        if out.device != x.device:
            raise ValueError(f"out must be on {x.device}; got {out.device=}")
        if out.shape != x.shape:
            raise ValueError(f"out must have shape {x.shape}; got {out.shape=}")
        require_int_tensor(out, "out")

        max_index_nbits = max_nbits_for_torch_index_dtype(out.dtype, dims=2)
        if nbits > max_index_nbits:
            prefer_uint = is_uint_torch_dtype(x.dtype) and is_uint_torch_dtype(y.dtype)
            try:
                viable_dtype = choose_index_torch_dtype(
                    nbits=nbits, dims=2, prefer_unsigned=prefer_uint
                )
            except Exception:
                viable_dtype = None
            msg = (
                f"{nbits=} does not fit in out dtype; got {out.dtype=} "
                f"which supports up to nbits={max_index_nbits}."
            )
            if viable_dtype is not None:
                msg += f" Consider using {viable_dtype} or a wider dtype, or reduce nbits to fit the out dtype."
            else:
                msg += "Reduce nbits to fit the out dtype."
            raise ValueError(msg)

    is_cpu = x.device.type == "cpu"
    is_cuda = x.device.type == "cuda"
    is_accelerator = not is_cpu
    is_compiling = torch.compiler.is_compiling()

    if is_accelerator and (not is_cuda) and gpu_backend == "triton":
        raise RuntimeError(
            f"gpu_backend='triton' requires CUDA tensors; got {x.device.type=}"
        )

    if is_cpu and (
        cpu_backend == "numba" or (cpu_backend == "auto" and not is_compiling)
    ):
        x_np = int_tensor_to_numpy_view(x, "x")
        y_np = int_tensor_to_numpy_view(y, "y")
        out_np = int_tensor_to_numpy_view(out, "out")

        hilbert_encode_2d_numba = get_hilbert_encode_2d_numba()

        if x.ndim == 0:
            idx = hilbert_encode_2d_numba(x_np, y_np, nbits=nbits)
            out_np[...] = idx
            return out

        parallel = resolve_cpu_parallel(cpu_parallel, x.numel())

        hilbert_encode_2d_numba(x_np, y_np, nbits=nbits, out=out_np, parallel=parallel)
        return out

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = x.is_contiguous() and y.is_contiguous() and out.is_contiguous()
        contig_details = (
            f"{x.is_contiguous()=}, {y.is_contiguous()=}, {out.is_contiguous()=}"
            if out_provided
            else f"{x.is_contiguous()=}, {y.is_contiguous()=}"
        )

        def _call() -> None:
            triton_encode_2d = get_hilbert_encode_2d_triton()
            # Do not pass a dtype-view alias of `out` into Triton.
            # When in compiled region inductor can hit an internal assertion when a Triton
            # wrapper mutates via `out.view(int64)` and the graph returns the base `out`.
            triton_encode_2d(x, y, nbits=nbits, out=out, lut_cache=lut_cache)

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out

    # CPU - auto with compiling, or explicit torch backend
    # Accelerator - auto with non-CUDA or with CUDA not all contiguous,
    #               or explicit torch backend

    # Not strictly needed when is_compiling with inductor.
    x_i = int_tensor_to_signed_view(x, "x")
    y_i = int_tensor_to_signed_view(y, "y")
    out_i = int_tensor_to_signed_view(out, "out")

    hilbert_encode_2d_torch(x_i, y_i, nbits=nbits, out=out_i, lut_cache=lut_cache)
    return out


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

    cpu_backend = validate_cpu_backend(cpu_backend)
    gpu_backend = validate_gpu_backend(gpu_backend)

    require_int_tensor(index, "index")

    max_index_nbits = max_nbits_for_torch_index_dtype(index.dtype, dims=2)
    if nbits is None:
        nbits = max_index_nbits
    else:
        validate_nbits_2d(nbits)
        if nbits > max_index_nbits:
            raise ValueError(
                f"nbits={nbits} exceeds the effective bits of the index dtype; "
                f"got {index.dtype=} which supports up to {max_index_nbits} bits for 2D coordinates. "
                f"max nbits is {max_index_nbits}."
            )

    if (out_x is None) != (out_y is None):
        raise ValueError("out_x and out_y must be provided together")

    out_provided = out_x is not None and out_y is not None

    if not out_provided:
        prefer_uint = is_uint_torch_dtype(index.dtype)
        auto_coord_dtype = choose_coord_torch_dtype(
            nbits=nbits,
            prefer_unsigned=prefer_uint,
        )
        out_x = torch.empty(index.shape, dtype=auto_coord_dtype, device=index.device)
        out_y = torch.empty(index.shape, dtype=auto_coord_dtype, device=index.device)
    else:
        out_x = cast(torch.Tensor, out_x)
        out_y = cast(torch.Tensor, out_y)

        if out_x.device != index.device or out_y.device != index.device:
            raise ValueError(
                "out_x and out_y must be on the same device as index; "
                f"got {index.device=}, {out_x.device=}, {out_y.device=}"
            )
        if out_x.shape != index.shape or out_y.shape != index.shape:
            raise ValueError(
                "out_x and out_y must have the same shape as index; "
                f"got {index.shape=}, {out_x.shape=}, {out_y.shape=}"
            )

        require_int_tensor(out_x, "out_x")
        require_int_tensor(out_y, "out_y")

        max_coord_nbits = min(
            effective_bits_torch_dtype(out_x.dtype),
            effective_bits_torch_dtype(out_y.dtype),
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in out_x/out_y dtypes; "
                f"got {out_x.dtype=} with {effective_bits_torch_dtype(out_x.dtype)} effective bits, "
                f"{out_y.dtype=} with {effective_bits_torch_dtype(out_y.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}"
            )

    is_cpu = index.device.type == "cpu"
    is_cuda = index.device.type == "cuda"
    is_accelerator = not is_cpu
    is_compiling = torch.compiler.is_compiling()

    if is_accelerator and (not is_cuda) and gpu_backend == "triton":
        raise RuntimeError(
            f"gpu_backend='triton' requires CUDA tensors; got {index.device.type=}"
        )

    if is_cpu and (
        cpu_backend == "numba" or (cpu_backend == "auto" and not is_compiling)
    ):
        index_np = int_tensor_to_numpy_view(index, "index")
        out_x_np = int_tensor_to_numpy_view(out_x, "out_x")
        out_y_np = int_tensor_to_numpy_view(out_y, "out_y")

        hilbert_decode_2d_numba = get_hilbert_decode_2d_numba()

        if index.ndim == 0:
            x_i, y_i = hilbert_decode_2d_numba(index_np, nbits=nbits)
            out_x_np[...] = x_i
            out_y_np[...] = y_i
            return out_x, out_y

        parallel = resolve_cpu_parallel(cpu_parallel, index.numel())

        hilbert_decode_2d_numba(
            index_np, nbits=nbits, out_x=out_x_np, out_y=out_y_np, parallel=parallel
        )
        return out_x, out_y

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = (
            index.is_contiguous() and out_x.is_contiguous() and out_y.is_contiguous()
        )
        contig_details = (
            f"{index.is_contiguous()=}, {out_x.is_contiguous()=}, {out_y.is_contiguous()=}"
            if out_provided
            else f"{index.is_contiguous()=}"
        )

        def _call() -> None:
            triton_decode_2d = get_hilbert_decode_2d_triton()
            triton_decode_2d(
                index,
                nbits=nbits,
                out_x=out_x,
                out_y=out_y,
                lut_cache=lut_cache,
            )

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out_x, out_y

    # CPU - auto with compiling, or explicit torch backend
    # Accelerator - auto with non-CUDA or with CUDA not all contiguous,
    #               or explicit torch backend

    # Not strictly needed when is_compiling with inductor.
    index_i = int_tensor_to_signed_view(index, "index")
    out_x_i = int_tensor_to_signed_view(out_x, "out_x")
    out_y_i = int_tensor_to_signed_view(out_y, "out_y")

    hilbert_decode_2d_torch(
        index_i,
        nbits=nbits,
        out_x=out_x_i,
        out_y=out_y_i,
        lut_cache=lut_cache,
    )
    return out_x, out_y
