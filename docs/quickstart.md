# Quick start

If you’re here, you probably have a pile of coordinates and you want a 1D ordering that preserves locality.
That’s exactly what the Hilbert space-filling curve is good at: points that are "close" in 2D/3D tend to stay close after mapping to a single integer index.

HilbertSFC gives you that mapping with a performance-first implementation (NumPy + Numba), so you can use it both interactively and in tight loops.

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

### Pick `nbits`

`nbits` is the number of bits per coordinate, so each coordinate must be in:

- `[0, 2**nbits)`

For example, `nbits=10` means you have a `1024×1024` (2D) or `1024×1024×1024` (3D) grid.

### Scalar 2D

Use scalar encode/decode when you're working with individual points:

```python
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

nbits = 10

index = hilbert_encode_2d(17, 23, nbits)
x, y = hilbert_decode_2d(index, nbits)
```

### Batch 2D

Batch mode is where HilbertSFC shines: pass NumPy arrays and use vectorized encode/decode kernels with minimal overhead:

```python
import numpy as np
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

nbits = 10
xs = np.arrange(1024, dtype=np.uint32)
ys = xs[::-1]

indices = hilbert_encode_2d(xs, ys, nbits)
xs2, ys2 = hilbert_decode_2d(indices, nbits)
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

indices = hilbert_encode_3d(xs, ys, zs, nbits)
xs2, ys2, zs2 = hilbert_decode_3d(indices, nbits)
```

## Tips

- Extra bits in the inputs are ignored, so you can pass wider integer dtypes if that’s convenient.
- Batch functions accept arbitrary shapes and preserve the input shape.
- For high-throughput batch workloads, try `parallel=True` (and control threads via `NUMBA_NUM_THREADS` or `numba.set_num_threads()`).
