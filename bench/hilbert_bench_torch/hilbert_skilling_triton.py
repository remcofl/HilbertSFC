from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "encode_2d_triton",
    "encode_3d_triton",
    "decode_2d_triton",
    "decode_3d_triton",
]


def _require_int_dtype(dtype: torch.dtype, *, name: str) -> None:
    if (
        not torch.is_floating_point(torch.empty((), dtype=dtype))
        and dtype != torch.bool
    ):
        return
    raise TypeError(f"{name} must be an integer dtype, got {dtype}")


def _resolve_out_dtype(
    *,
    out: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    fallback: torch.dtype,
    name: str,
) -> torch.dtype:
    if out is not None:
        _require_int_dtype(out.dtype, name=name)
        return out.dtype
    if out_dtype is None:
        return fallback
    _require_int_dtype(out_dtype, name=name)
    return out_dtype


def _require_cuda_tensor(t: torch.Tensor, *, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor for Triton kernels")


@triton.jit
def _exchange_invert_step(
    x,
    other,
    lower_mask,
    high_mask,
    other_bit,
):
    x_low = x & lower_mask
    other_low = other & lower_mask
    x_swapped = (x & high_mask) | other_low
    other_swapped = (other & high_mask) | x_low
    x_inverted = x ^ lower_mask
    x = tl.where(other_bit == 0, x_swapped, x_inverted)
    other = tl.where(other_bit == 0, other_swapped, other)
    return x, other


@triton.jit
def hilbert_encode_skilling_2d_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,  # noqa: N803
    NBITS: tl.constexpr,  # noqa: N803
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.uint64)
    y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.uint64)

    if NBITS < 64:
        in_mask = (1 << NBITS) - 1  # type: ignore[reportOperatorIssue]
        x = x & in_mask
        y = y & in_mask

    for i in tl.static_range(0, NBITS):  # type: ignore[reportGeneralTypeIssues]
        b = (NBITS - 1) - i
        lower_mask = (1 << b) - 1
        high_mask = ~lower_mask

        x_bit = (x >> b) & 1
        x = tl.where(x_bit != 0, x ^ lower_mask, x)

        y_bit = (y >> b) & 1
        x, y = _exchange_invert_step(x, y, lower_mask, high_mask, y_bit)  # type: ignore[reportGeneralTypeIssues]

    h_gray = tl.zeros([BLOCK], dtype=tl.uint64)
    for i in tl.static_range(0, NBITS):
        b = (NBITS - 1) - i
        h_gray = (h_gray << 1) | ((x >> b) & 1)
        h_gray = (h_gray << 1) | ((y >> b) & 1)

    h = h_gray
    h ^= h >> 1
    h ^= h >> 2
    h ^= h >> 4
    h ^= h >> 8
    h ^= h >> 16
    h ^= h >> 32

    tl.store(out_ptr + offsets, h, mask=mask)


@triton.jit
def hilbert_encode_skilling_3d_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,  # noqa: N803
    NBITS: tl.constexpr,  # noqa: N803
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.uint64)
    y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.uint64)
    z = tl.load(z_ptr + offsets, mask=mask, other=0).to(tl.uint64)

    if NBITS < 64:
        in_mask = (1 << NBITS) - 1  # type: ignore[reportOperatorIssue]
        x = x & in_mask
        y = y & in_mask
        z = z & in_mask

    for i in tl.static_range(0, NBITS):  # type: ignore[reportGeneralTypeIssues]
        b = (NBITS - 1) - i
        lower_mask = (1 << b) - 1
        high_mask = ~lower_mask

        x_bit = (x >> b) & 1
        x = tl.where(x_bit != 0, x ^ lower_mask, x)

        y_bit = (y >> b) & 1
        x, y = _exchange_invert_step(x, y, lower_mask, high_mask, y_bit)  # type: ignore[reportGeneralTypeIssues]

        z_bit = (z >> b) & 1
        x, z = _exchange_invert_step(x, z, lower_mask, high_mask, z_bit)

    h_gray = tl.zeros([BLOCK], dtype=tl.uint64)
    for i in tl.static_range(0, NBITS):
        b = (NBITS - 1) - i
        h_gray = (h_gray << 1) | ((x >> b) & 1)
        h_gray = (h_gray << 1) | ((y >> b) & 1)
        h_gray = (h_gray << 1) | ((z >> b) & 1)

    h = h_gray
    h ^= h >> 1
    h ^= h >> 2
    h ^= h >> 4
    h ^= h >> 8
    h ^= h >> 16
    h ^= h >> 32

    tl.store(out_ptr + offsets, h, mask=mask)


@triton.jit
def hilbert_decode_skilling_2d_kernel(
    idx_ptr,
    out_x_ptr,
    out_y_ptr,
    n_elements,
    BLOCK: tl.constexpr,  # noqa: N803
    NBITS: tl.constexpr,  # noqa: N803
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    idx = tl.load(idx_ptr + offsets, mask=mask, other=0).to(tl.uint64)
    if NBITS < 32:
        idx = idx & ((1 << (2 * NBITS)) - 1)  # type: ignore[reportOperatorIssue]

    gray = idx ^ (idx >> 1)

    x = tl.zeros([BLOCK], dtype=tl.uint64)
    y = tl.zeros([BLOCK], dtype=tl.uint64)
    for i in tl.static_range(0, NBITS):
        b = (NBITS - 1) - i
        x |= ((gray >> (2 * b + 1)) & 1) << b
        y |= ((gray >> (2 * b)) & 1) << b

    for s in tl.static_range(0, NBITS):
        lower_mask = (1 << s) - 1
        high_mask = ~lower_mask

        y_bit = (y >> s) & 1
        x, y = _exchange_invert_step(x, y, lower_mask, high_mask, y_bit)

        x_bit = (x >> s) & 1
        x = tl.where(x_bit != 0, x ^ lower_mask, x)

    tl.store(out_x_ptr + offsets, x.to(out_x_ptr.dtype.element_ty), mask=mask)
    tl.store(out_y_ptr + offsets, y.to(out_y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def hilbert_decode_skilling_3d_kernel(
    idx_ptr,
    out_x_ptr,
    out_y_ptr,
    out_z_ptr,
    n_elements,
    BLOCK: tl.constexpr,  # noqa: N803
    NBITS: tl.constexpr,  # noqa: N803
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    idx = tl.load(idx_ptr + offsets, mask=mask, other=0).to(tl.uint64)
    if NBITS < 22:
        idx = idx & ((1 << (3 * NBITS)) - 1)  # type: ignore[reportOperatorIssue]

    gray = idx ^ (idx >> 1)

    x = tl.zeros([BLOCK], dtype=tl.uint64)
    y = tl.zeros([BLOCK], dtype=tl.uint64)
    z = tl.zeros([BLOCK], dtype=tl.uint64)
    for i in tl.static_range(0, NBITS):
        b = (NBITS - 1) - i
        x |= ((gray >> (3 * b + 2)) & 1) << b
        y |= ((gray >> (3 * b + 1)) & 1) << b
        z |= ((gray >> (3 * b)) & 1) << b

    for s in tl.static_range(0, NBITS):
        lower_mask = (1 << s) - 1
        high_mask = ~lower_mask

        z_bit = (z >> s) & 1
        x, z = _exchange_invert_step(x, z, lower_mask, high_mask, z_bit)

        y_bit = (y >> s) & 1
        x, y = _exchange_invert_step(x, y, lower_mask, high_mask, y_bit)

        x_bit = (x >> s) & 1
        x = tl.where(x_bit != 0, x ^ lower_mask, x)

    tl.store(out_x_ptr + offsets, x.to(out_x_ptr.dtype.element_ty), mask=mask)
    tl.store(out_y_ptr + offsets, y.to(out_y_ptr.dtype.element_ty), mask=mask)
    tl.store(out_z_ptr + offsets, z.to(out_z_ptr.dtype.element_ty), mask=mask)


def encode_2d_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_bits: int,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    block: int = 1024,
) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
    _require_int_dtype(x.dtype, name="x")
    _require_int_dtype(y.dtype, name="y")
    _require_cuda_tensor(x, name="x")
    _require_cuda_tensor(y, name="y")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if not (1 <= num_bits <= 32):
        raise ValueError("2D uses 2*num_bits bits and supports 1 <= num_bits <= 32")

    resolved_out_dtype = _resolve_out_dtype(
        out=out,
        out_dtype=out_dtype,
        fallback=torch.uint64,
        name="out",
    )
    if out is None:
        out = torch.empty_like(x, dtype=resolved_out_dtype)
    elif out.shape != x.shape:
        raise ValueError(f"out must have shape {x.shape}, got {out.shape}")
    _require_cuda_tensor(out, name="out")
    if out.device != x.device:
        raise ValueError("out must be on the same device as inputs")

    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block),)
    hilbert_encode_skilling_2d_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK=block,
        NBITS=num_bits,
        num_warps=8,  # type: ignore[reportCallIssue]
    )
    return out


def encode_3d_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    num_bits: int,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    block: int = 1024,
) -> torch.Tensor:
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"x, y, z must have same shape, got {x.shape}, {y.shape}, {z.shape}"
        )
    _require_int_dtype(x.dtype, name="x")
    _require_int_dtype(y.dtype, name="y")
    _require_int_dtype(z.dtype, name="z")
    _require_cuda_tensor(x, name="x")
    _require_cuda_tensor(y, name="y")
    _require_cuda_tensor(z, name="z")
    if x.device != y.device or x.device != z.device:
        raise ValueError("x, y, z must be on the same device")
    if not (1 <= num_bits <= 21):
        raise ValueError("3D uses 3*num_bits bits and supports 1 <= num_bits <= 21")

    resolved_out_dtype = _resolve_out_dtype(
        out=out,
        out_dtype=out_dtype,
        fallback=torch.uint64,
        name="out",
    )
    if out is None:
        out = torch.empty_like(x, dtype=resolved_out_dtype)
    elif out.shape != x.shape:
        raise ValueError(f"out must have shape {x.shape}, got {out.shape}")
    _require_cuda_tensor(out, name="out")
    if out.device != x.device:
        raise ValueError("out must be on the same device as inputs")

    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block),)
    hilbert_encode_skilling_3d_kernel[grid](
        x,
        y,
        z,
        out,
        n_elements,
        BLOCK=block,
        NBITS=num_bits,
        num_warps=8,  # type: ignore[reportCallIssue]
    )
    return out


def decode_2d_triton(
    h: torch.Tensor,
    *,
    num_bits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_int_dtype(h.dtype, name="h")
    _require_cuda_tensor(h, name="h")
    if not (1 <= num_bits <= 32):
        raise ValueError("2D uses 2*num_bits bits and supports 1 <= num_bits <= 32")

    resolved_out_dtype = _resolve_out_dtype(
        out=out_x,
        out_dtype=out_dtype,
        fallback=torch.uint64,
        name="out_x",
    )

    if out_x is None:
        out_x = torch.empty_like(h, dtype=resolved_out_dtype)
    elif out_x.shape != h.shape:
        raise ValueError(f"out_x must have shape {h.shape}, got {out_x.shape}")

    if out_y is None:
        out_y = torch.empty_like(h, dtype=resolved_out_dtype)
    elif out_y.shape != h.shape:
        raise ValueError(f"out_y must have shape {h.shape}, got {out_y.shape}")

    _require_int_dtype(out_x.dtype, name="out_x")
    _require_int_dtype(out_y.dtype, name="out_y")
    _require_cuda_tensor(out_x, name="out_x")
    _require_cuda_tensor(out_y, name="out_y")
    if out_x.device != h.device or out_y.device != h.device:
        raise ValueError("out_x and out_y must be on the same device as h")

    n_elements = h.numel()
    grid = (triton.cdiv(n_elements, block),)
    hilbert_decode_skilling_2d_kernel[grid](
        h,
        out_x,
        out_y,
        n_elements,
        BLOCK=block,
        NBITS=num_bits,
        num_warps=8,  # type: ignore[reportCallIssue]
    )
    return out_x, out_y


def decode_3d_triton(
    h: torch.Tensor,
    *,
    num_bits: int,
    out_x: torch.Tensor | None = None,
    out_y: torch.Tensor | None = None,
    out_z: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_int_dtype(h.dtype, name="h")
    _require_cuda_tensor(h, name="h")
    if not (1 <= num_bits <= 21):
        raise ValueError("3D uses 3*num_bits bits and supports 1 <= num_bits <= 21")

    resolved_out_dtype = _resolve_out_dtype(
        out=out_x,
        out_dtype=out_dtype,
        fallback=torch.uint64,
        name="out_x",
    )

    if out_x is None:
        out_x = torch.empty_like(h, dtype=resolved_out_dtype)
    elif out_x.shape != h.shape:
        raise ValueError(f"out_x must have shape {h.shape}, got {out_x.shape}")

    if out_y is None:
        out_y = torch.empty_like(h, dtype=resolved_out_dtype)
    elif out_y.shape != h.shape:
        raise ValueError(f"out_y must have shape {h.shape}, got {out_y.shape}")

    if out_z is None:
        out_z = torch.empty_like(h, dtype=resolved_out_dtype)
    elif out_z.shape != h.shape:
        raise ValueError(f"out_z must have shape {h.shape}, got {out_z.shape}")

    _require_int_dtype(out_x.dtype, name="out_x")
    _require_int_dtype(out_y.dtype, name="out_y")
    _require_int_dtype(out_z.dtype, name="out_z")
    _require_cuda_tensor(out_x, name="out_x")
    _require_cuda_tensor(out_y, name="out_y")
    _require_cuda_tensor(out_z, name="out_z")
    if out_x.device != h.device or out_y.device != h.device or out_z.device != h.device:
        raise ValueError("out_x, out_y, out_z must be on the same device as h")

    n_elements = h.numel()
    grid = (triton.cdiv(n_elements, block),)
    hilbert_decode_skilling_3d_kernel[grid](
        h,
        out_x,
        out_y,
        out_z,
        n_elements,
        BLOCK=block,
        NBITS=num_bits,
        num_warps=8,  # type: ignore[reportCallIssue]
    )
    return out_x, out_y, out_z
