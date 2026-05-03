# Quick start

HilbertSFC provides simple Hilbert and Morton encode/decode APIs for Python scalars, NumPy arrays, and PyTorch tensors.

## Installation

Install the base package with either `pip` or `uv`:

/// tab | pip

```bash
pip install hilbertsfc
```

///

/// tab | uv

```bash
uv add hilbertsfc
```

///

### PyTorch support

To enable the optional PyTorch extension, install with the `torch` extra:

/// tab | pip

```bash
pip install "hilbertsfc[torch]"
```

///

/// tab | uv

```bash
uv add "hilbertsfc[torch]"
```

///

/// admonition | Note
By default, installing `hilbertsfc[torch]` pulls in a platform-default PyTorch build:

- **Windows:** CPU-only
- **Linux:** CUDA-enabled

If you need a specific PyTorch, CUDA, or ROCm version, follow the official
[PyTorch installation instructions](https://pytorch.org/get-started/locally/). Then install `hilbertsfc[torch]` as shown above.

Alternatively, you can install it in one step by specifying the appropriate PyTorch wheel index, e.g., for CUDA 13.0:
/// tab | pip

```bash
pip install "hilbertsfc[torch]" --extra-index-url https://download.pytorch.org/whl/cu130
```

///

/// tab | uv

```bash
uv add "hilbertsfc[torch]" --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

///
///

## First steps

### Python scalars

Use [`hilbert_encode_2d`][hilbertsfc.hilbert_encode_2d] and [`hilbert_decode_2d`][hilbertsfc.hilbert_decode_2d] directly on Python integers:

```python
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

index = hilbert_encode_2d(17, 23, nbits=10)   # index = 534
x, y = hilbert_decode_2d(index, nbits=10)     # x = 17, y = 23
```

`nbits` controls the coordinate domain `[0, 2**nbits)` on each axis. For best performance, pass the smallest value that covers your input range.

The 3D API follows the same pattern via [`hilbert_encode_3d`][hilbertsfc.hilbert_encode_3d] and [`hilbert_decode_3d`][hilbertsfc.hilbert_decode_3d]. Morton/z-order functions mirror these names with [`morton_encode_2d`][hilbertsfc.morton_encode_2d], [`morton_decode_2d`][hilbertsfc.morton_decode_2d], [`morton_encode_3d`][hilbertsfc.morton_encode_3d], and [`morton_decode_3d`][hilbertsfc.morton_decode_3d].

/// admonition | `nbits` compatibility
    type: info

Hilbert indices obtained with a certain `nbits` are compatible with those from another `nbits`, given that the coordinates are within the valid range. This is because the kernels resolve the starting state parity to ensure compatibility.
///

### NumPy arrays

The same functions also accept NumPy integer arrays:

```python
import numpy as np
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

nbits = 10
shape = (256, 256)
rng = np.random.default_rng(0)

xs = rng.integers(0, 2**nbits, size=shape, dtype=np.uint32)
ys = rng.integers(0, 2**nbits, size=shape, dtype=np.uint32)

indices = hilbert_encode_2d(xs, ys, nbits=nbits)     # indices.shape = (256, 256)
xs2, ys2 = hilbert_decode_2d(indices, nbits=nbits)   # xs2 = xs, ys2 = ys
```

Arbitrary shapes are supported with *zero-copy* access. Strided views also work but they can reduce performance since the kernels are close to memory-bandwidth bound. You can optionally provide `out=...` buffers for encode and `out_xs/out_ys/...` buffers for decode to reuse memory or write into memory-mapped arrays.

/// admonition | Parallel execution
    type: tip

Use `parallel=True` to dispatch the parallel kernel. The number of threads can be controlled with the environment variable `NUMBA_NUM_THREADS` or at runtime via `numba.set_num_threads()`.
///

### PyTorch tensors

Use the torch frontend API [`hilbertsfc.torch`][hilbertsfc.torch] for PyTorch tensors on CPU and accelerator devices. By default on CUDA, contiguous tensors take the Triton path when available; otherwise the implementation falls back to the Torch backend.

```python
import torch
from hilbertsfc.torch import hilbert_decode_2d, hilbert_encode_2d

device = "cuda" if torch.cuda.is_available() else "cpu"
nbits = 10
n = 4096

xs = torch.randint(0, 2**nbits, (n,), dtype=torch.int32, device=device)
ys = torch.randint(0, 2**nbits, (n,), dtype=torch.int32, device=device)

indices = hilbert_encode_2d(xs, ys, nbits=nbits)
xs2, ys2 = hilbert_decode_2d(indices, nbits=nbits)
```

#### Use with `torch.compile`

If you plan to use [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html), call [`precache_compile_luts`][hilbertsfc.torch.precache_compile_luts] first so LUT materialization happens outside the compiled region. This avoids graph breaks and extra overhead, and is required for `fullgraph=True`.

```python
import torch
from hilbertsfc.torch import hilbert_encode_2d, precache_compile_luts

device = "cuda" if torch.cuda.is_available() else "cpu"
precache_compile_luts(device=device, op="hilbert_encode_2d")

def encode_then_scale(x: torch.Tensor, y: torch.Tensor, nbits: int) -> torch.Tensor:
    return hilbert_encode_2d(x, y, nbits=nbits) * 2

compiled_encode_then_scale = torch.compile(encode_then_scale, fullgraph=True)
```

/// admonition | Free LUT cache
    type: info

Torch-side LUT tensors are cached per device for reuse. The LUT cache is only a couple KiB, so clearing it is rarely necessary, but you can free the associated device memory with [`clear_torch_lut_caches`][hilbertsfc.torch.clear_torch_lut_caches].
///

## Next steps

For advanced usage, including embedding scalar kernels into your own Numba code, see [Advanced usage](advanced-usage.md).
