"""Torch 3D Hilbert API dispatch layer."""

import warnings
from typing import cast

import torch

from .._nbits import MAX_NBITS_3D, validate_nbits_3d
from ._dispatch import (
    get_hilbert_decode_3d_numba,
    get_hilbert_decode_3d_triton,
    get_hilbert_encode_3d_numba,
    get_hilbert_encode_3d_triton,
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
from ._kernels.torch.hilbert3d_decode import hilbert_decode_3d_torch
from ._kernels.torch.hilbert3d_encode import hilbert_encode_3d_torch
from ._luts import TorchCacheMode
from ._numpy_interop import int_tensor_to_numpy_view
from ._tensor_int import (
    int_tensor_to_signed_view,
    is_uint_torch_dtype,
    require_int_tensor,
)
from ._tuning_mode import TritonTuningMode, validate_triton_tuning_mode


def hilbert_encode_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    nbits: int | None = None,
    out: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> torch.Tensor:
    """Encode 3D integer coordinates to Hilbert indices.

    This function provides a PyTorch equivalent of
    [`hilbert_encode_3d`][hilbertsfc.hilbert3d.hilbert_encode_3d]. It accepts
    integer ``torch.Tensor`` of arbitrary shape on any device, and dispatches
    to backend-specific implementations depending on device and backend settings.

    Parameters
    ----------
    x, y, z
        Integer coordinate tensors to encode.

        Must have identical ``shape`` and be on the same device.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis. For inputs outside that domain, only the low
        ``nbits`` bits of each coordinate are used.

        Must satisfy ``1 <= nbits <= 21``. If provided, it must also fit within
        the usable bits of the coordinate dtype.

        If ``None``:

        - Array mode: inferred from the coordinate dtype using its usable bit
          width, capped at 21. For example, ``uint16`` -> 16, ``int16`` -> 15
          (sign bit excluded), and ``uint64``/``int64`` -> 21.
        - Scalar mode: defaults to 21.

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input coordinate range.
    out
        Optional output tensor.

        Must have the same shape and device as ``x``, ``y``, and ``z`` and an integer
        dtype wide enough to hold ``3 * nbits`` bits.
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
    torch.Tensor
        Hilbert indices.

        - Has the same shape/device as the inputs.
        - If ``out`` is provided, returns ``out``.
        - Otherwise, chooses a minimal integer dtype that can represent
          ``3 * nbits`` bits, preferring unsigned if all inputs are unsigned
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
    triton_tuning = validate_triton_tuning_mode(triton_tuning)

    if x.device != y.device or x.device != z.device:
        raise ValueError(
            "x, y, z must be on the same device; "
            f"got {x.device=}, {y.device=}, {z.device=}"
        )
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"x, y, z must have the same shape; got {x.shape=}, {y.shape=}, {z.shape=}"
        )

    require_int_tensor(x, "x")
    require_int_tensor(y, "y")
    require_int_tensor(z, "z")

    max_coord_nbits = max(
        effective_bits_torch_dtype(x.dtype),
        effective_bits_torch_dtype(y.dtype),
        effective_bits_torch_dtype(z.dtype),
    )
    if nbits is None:
        nbits = max_coord_nbits
        if nbits > MAX_NBITS_3D:
            warnings.warn(
                f"The maximum effective bits of the coordinate dtype is {nbits}, "
                f"which exceeds the algorithm maximum of {MAX_NBITS_3D}. "
                f"Using nbits={MAX_NBITS_3D} instead. This means that excess bits in the input coordinates "
                f"will be ignored. To silence this warning, explicitly set nbits<={MAX_NBITS_3D}.",
                UserWarning,
                stacklevel=2,
            )
            nbits = MAX_NBITS_3D
    else:
        validate_nbits_3d(nbits)
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in coordinate dtypes; "
                f"got {x.dtype=} with {effective_bits_torch_dtype(x.dtype)} effective bits, "
                f"{y.dtype=} with {effective_bits_torch_dtype(y.dtype)} effective bits, "
                f"{z.dtype=} with {effective_bits_torch_dtype(z.dtype)} effective bits; "
                f"max nbits is {max_coord_nbits}."
            )

    out_provided = out is not None

    if not out_provided:
        prefer_uint = (
            is_uint_torch_dtype(x.dtype)
            and is_uint_torch_dtype(y.dtype)
            and is_uint_torch_dtype(z.dtype)
        )
        auto_out_dtype = choose_index_torch_dtype(
            nbits=nbits,
            dims=3,
            prefer_unsigned=prefer_uint,
        )
        out = torch.empty(x.shape, dtype=auto_out_dtype, device=x.device)
    else:
        if out.device != x.device:
            raise ValueError(f"out must be on {x.device}; got {out.device=}")
        if out.shape != x.shape:
            raise ValueError(f"out must have shape {x.shape}; got {out.shape=}")
        require_int_tensor(out, "out")

        max_index_nbits = max_nbits_for_torch_index_dtype(out.dtype, dims=3)
        if nbits > max_index_nbits:
            prefer_uint = (
                is_uint_torch_dtype(x.dtype)
                and is_uint_torch_dtype(y.dtype)
                and is_uint_torch_dtype(z.dtype)
            )
            try:
                viable_dtype = choose_index_torch_dtype(
                    nbits=nbits, dims=3, prefer_unsigned=prefer_uint
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
                msg += " Reduce nbits to fit the out dtype."
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
        z_np = int_tensor_to_numpy_view(z, "z")
        out_np = int_tensor_to_numpy_view(out, "out")

        hilbert_encode_3d_numba = get_hilbert_encode_3d_numba()

        if x.ndim == 0:
            idx = hilbert_encode_3d_numba(x_np, y_np, z_np, nbits=nbits)
            out_np[...] = idx
            return out

        parallel = resolve_cpu_parallel(cpu_parallel, x.numel())
        hilbert_encode_3d_numba(
            x_np,
            y_np,
            z_np,
            nbits=nbits,
            out=out_np,
            parallel=parallel,
        )
        return out

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = (
            x.is_contiguous()
            and y.is_contiguous()
            and z.is_contiguous()
            and out.is_contiguous()
        )
        contig_details = (
            f"{x.is_contiguous()=}, {y.is_contiguous()=}, {z.is_contiguous()=}, {out.is_contiguous()=}"
            if out_provided
            else f"{x.is_contiguous()=}, {y.is_contiguous()=}, {z.is_contiguous()=}"
        )

        def _call() -> None:
            triton_encode_3d = get_hilbert_encode_3d_triton()
            triton_encode_3d(
                x,
                y,
                z,
                nbits=nbits,
                out=out,
                lut_cache=lut_cache,
                triton_tuning=triton_tuning,
            )

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out

    x_i = int_tensor_to_signed_view(x, "x")
    y_i = int_tensor_to_signed_view(y, "y")
    z_i = int_tensor_to_signed_view(z, "z")
    out_i = int_tensor_to_signed_view(out, "out")

    hilbert_encode_3d_torch(x_i, y_i, z_i, nbits=nbits, out=out_i, lut_cache=lut_cache)
    return out


def hilbert_decode_3d(
    index: torch.Tensor,
    *,
    nbits: int | None = None,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_z: torch.Tensor | None = None,
    lut_cache: TorchCacheMode = "device",
    cpu_parallel: bool | None = None,
    cpu_backend: CPUBackend = "auto",
    gpu_backend: GPUBackend = "auto",
    triton_tuning: TritonTuningMode = "heuristic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode Hilbert indices to 3D integer coordinates.

    This function provides a PyTorch equivalent of
    [`hilbert_decode_3d`][hilbertsfc.hilbert3d.hilbert_decode_3d]. It accepts
    integer ``torch.Tensor`` of arbitrary shape on any device, and dispatches
    to backend-specific implementations depending on device and backend settings.

    Parameters
    ----------
    index
        Integer Hilbert index tensor to decode.
    nbits
        Number of bits per coordinate axis. This defines a coordinate domain of
        ``[0, 2**nbits)`` on each axis and a Hilbert index range of
        ``[0, 2**(3 * nbits))``. For indices outside that range, only the low
        ``3 * nbits`` bits are used.

        Must satisfy ``1 <= nbits <= 21``. If provided, it must also fit within
        the usable bits of the index dtype.

        If ``None``:

        - Inferred from the index dtype as one third of its usable bit width, capped
        at 21. For example, ``uint16`` -> 5, ``uint64`` -> 21, and ``int64`` -> 21
        (sign bit excluded).

        For best performance and tighter output dtypes, pass the smallest value
        that covers the input index range.
    out_x, out_y, out_z
        Optional output coordinate tensors. Either provide all three or none.

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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Decoded coordinates ``(x, y, z)``.

        - Each tensor has the same shape/device as ``index``.
        - If ``out_x``, ``out_y``, and ``out_z`` are provided, returns
        ``(out_x, out_y, out_z)``.
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
    triton_tuning = validate_triton_tuning_mode(triton_tuning)

    require_int_tensor(index, "index")

    max_index_nbits = max_nbits_for_torch_index_dtype(index.dtype, dims=3)
    if nbits is None:
        nbits = max_index_nbits
    else:
        validate_nbits_3d(nbits)
        if nbits > max_index_nbits:
            raise ValueError(
                f"nbits={nbits} exceeds the effective bits of the index dtype; "
                f"got {index.dtype=} which supports up to {max_index_nbits} bits for 3D coordinates. "
                f"max nbits is {max_index_nbits}."
            )

    n_outs = (out_x is not None) + (out_y is not None) + (out_z is not None)
    if n_outs not in (0, 3):
        raise ValueError("out_x, out_y, out_z must be provided together")

    out_provided = n_outs == 3

    if not out_provided:
        prefer_uint = is_uint_torch_dtype(index.dtype)
        auto_coord_dtype = choose_coord_torch_dtype(
            nbits=nbits,
            prefer_unsigned=prefer_uint,
        )
        out_x = torch.empty(index.shape, dtype=auto_coord_dtype, device=index.device)
        out_y = torch.empty(index.shape, dtype=auto_coord_dtype, device=index.device)
        out_z = torch.empty(index.shape, dtype=auto_coord_dtype, device=index.device)
    else:
        out_x = cast(torch.Tensor, out_x)
        out_y = cast(torch.Tensor, out_y)
        out_z = cast(torch.Tensor, out_z)

        if (
            out_x.device != index.device
            or out_y.device != index.device
            or out_z.device != index.device
        ):
            raise ValueError(
                "out_x, out_y, out_z must be on the same device as index; "
                f"got {index.device=}, {out_x.device=}, {out_y.device=}, {out_z.device=}"
            )
        if (
            out_x.shape != index.shape
            or out_y.shape != index.shape
            or out_z.shape != index.shape
        ):
            raise ValueError(
                "out_x, out_y, out_z must have the same shape as index; "
                f"got {index.shape=}, {out_x.shape=}, {out_y.shape=}, {out_z.shape=}"
            )

        require_int_tensor(out_x, "out_x")
        require_int_tensor(out_y, "out_y")
        require_int_tensor(out_z, "out_z")

        max_coord_nbits = min(
            effective_bits_torch_dtype(out_x.dtype),
            effective_bits_torch_dtype(out_y.dtype),
            effective_bits_torch_dtype(out_z.dtype),
        )
        if nbits > max_coord_nbits:
            raise ValueError(
                f"{nbits=} does not fit in out_x/out_y/out_z dtypes; "
                f"got {out_x.dtype=} with {effective_bits_torch_dtype(out_x.dtype)} effective bits, "
                f"{out_y.dtype=} with {effective_bits_torch_dtype(out_y.dtype)} effective bits, "
                f"{out_z.dtype=} with {effective_bits_torch_dtype(out_z.dtype)} effective bits; "
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
        out_z_np = int_tensor_to_numpy_view(out_z, "out_z")

        hilbert_decode_3d_numba = get_hilbert_decode_3d_numba()

        if index.ndim == 0:
            x_i, y_i, z_i = hilbert_decode_3d_numba(index_np, nbits=nbits)
            out_x_np[...] = x_i
            out_y_np[...] = y_i
            out_z_np[...] = z_i
            return out_x, out_y, out_z

        parallel = resolve_cpu_parallel(cpu_parallel, index.numel())
        hilbert_decode_3d_numba(
            index_np,
            nbits=nbits,
            out_x=out_x_np,
            out_y=out_y_np,
            out_z=out_z_np,
            parallel=parallel,
        )
        return out_x, out_y, out_z

    if is_cuda and (gpu_backend == "auto" or gpu_backend == "triton"):
        all_contiguous = (
            index.is_contiguous()
            and out_x.is_contiguous()
            and out_y.is_contiguous()
            and out_z.is_contiguous()
        )
        contig_details = (
            f"{index.is_contiguous()=}, {out_x.is_contiguous()=}, {out_y.is_contiguous()=}, {out_z.is_contiguous()=}"
            if out_provided
            else f"{index.is_contiguous()=}"
        )

        def _call() -> None:
            triton_decode_3d = get_hilbert_decode_3d_triton()
            triton_decode_3d(
                index,
                nbits=nbits,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                lut_cache=lut_cache,
                triton_tuning=triton_tuning,
            )

        if attempt_run_triton(
            gpu_backend=gpu_backend,
            all_contiguous=all_contiguous,
            contiguity_details=contig_details,
            call_triton=_call,
        ):
            return out_x, out_y, out_z

    index_i = int_tensor_to_signed_view(index, "index")
    out_x_i = int_tensor_to_signed_view(out_x, "out_x")
    out_y_i = int_tensor_to_signed_view(out_y, "out_y")
    out_z_i = int_tensor_to_signed_view(out_z, "out_z")

    hilbert_decode_3d_torch(
        index_i,
        nbits=nbits,
        out_x=out_x_i,
        out_y=out_y_i,
        out_z=out_z_i,
        lut_cache=lut_cache,
    )
    return out_x, out_y, out_z
