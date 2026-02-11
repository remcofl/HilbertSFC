# Quick start

If you're here, you probably have a pile of 2D/3D coordinates and you want a 1D ordering that preserves locality.
That's exactly what the Hilbert space-filling curve is good at: points that are close to each other in Euclidean space tend to stay close after mapping them to a 1D Hilbert index.

HilbertSFC gives you that mapping with a performance-first implementation (NumPy + Numba) with straightforward APIs for both scalar and array operations. It's designed to be fast and easy to use for a wide range of applications, from spatial indexing to data compression and more.

## Install

With pip:

```bash
pip install hilbertsfc
```

Or with uv:

```bash
uv add hilbertsfc
```

## First steps

### Understanding `nbits`

`nbits` specifies the number of bits per coordinate and defines the grid domain as `[0, 2**nbits)`. For example, `nbits=10` means coordinates are in the range `[0, 1024)`.

A tighter `nbits` improves performance and reduces output dtypes.

When omitted, `nbits` is inferred as follows:

- **For arrays**: inferred from the input dtype:
    - For encoding: uses the effective bits of coordinate dtypes (e.g., `uint8` → **8** bits, `int16` → **15** bits), capped at the algorithm maximum (**32** for 2D, **21** for 3D)
    - For decoding: uses `index_bits / dims` (e.g., `uint32` index → **16** bits for 2D)

- **For Python scalars**: defaults to the maximum (**32** for 2D, **21** for 3D)

### Scalar 2D

Use scalar encode/decode when you're working with individual points:

```python
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

# With explicit nbits
index = hilbert_encode_2d(17, 23, nbits=10)  # index = 534
x, y = hilbert_decode_2d(index, nbits=10)    # x, y = (17, 23)

# Using default nbits=32
index = hilbert_encode_2d(17, 23)
x, y = hilbert_decode_2d(index)
```

### Batch 2D

Batch mode is where HilbertSFC shines: pass NumPy arrays and use vectorized encode/decode kernels with minimal overhead:

```python
import numpy as np
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

xs = np.arange(1024, dtype=np.uint16)
ys = xs[::-1]

# With explicit nbits for tight bounds
indices = hilbert_encode_2d(xs, ys, nbits=10)     # shape (1024,), dtype uint32
xs2, ys2 = hilbert_decode_2d(indices, nbits=10)   # xs2 = xs, ys2 = ys

# Or use default nbits=32
indices = hilbert_encode_2d(xs, ys)
xs2, ys2 = hilbert_decode_2d(indices)
```

### Batch 3D

The 3D API works the same way, but with `(x, y, z)` coordinates:

```python
import numpy as np
from hilbertsfc import hilbert_decode_3d, hilbert_encode_3d

nbits = 10
n = 10_000
rng = np.random.default_rng(0)

xs = rng.integers(0, 2**nbits, size=n, dtype=np.uint32)
ys = rng.integers(0, 2**nbits, size=n, dtype=np.uint32)
zs = rng.integers(0, 2**nbits, size=n, dtype=np.uint32)

indices = hilbert_encode_3d(xs, ys, zs, nbits=nbits)
xs2, ys2, zs2 = hilbert_decode_3d(indices, nbits=nbits)
```

## Tips

- Extra bits in the input outside the range `[0, 2**nbits)` are ignored, so you can pass wider integer dtypes if that's convenient
- Hilbert indices obtained with a certain `nbits` are compatible with those from another `nbits`, given that the coordinates are within the valid range. This is because the kernels resolve the starting state parity to ensure compatibility.
- The batched API accepts arbitrary shapes and preserves the input shape. The requirement is that inputs/outputs support a *zero-copy* 1D view. Most strided views are supported but they can reduce performance since the kernels are close to memory-bandwidth bound.
- You can pass `out=...` buffers for batch encode, and `out_xs/out_ys/out_zs` for batch decode. This can for example be useful to write into memory-mapped arrays or to reuse buffers across multiple calls.
- `parallel=True` dispatches the parallel version of the kernel (when available). The number of threads can be controlled with the environment variable `NUMBA_NUM_THREADS` or during runtime with `numba.set_num_threads()`.
