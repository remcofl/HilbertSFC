<!-- markdownlint-disable MD033 MD041 -->
---
<h1 align="center">
HilbertSFC
</h1>

<p align="center">
    <strong>Ultra-fast 2D &amp; 3D Hilbert space-filling curve encode/decode kernels for Python.</strong>
</p>

<p align="center">
    <img src="docs/img/hilbert2d_grid.png" width="420" align="middle" alt="2D Hilbert curves for nbits 1..5" />
    <img src="docs/img/hilbert3d_grid.webp" width="340" align="middle" hspace="5" alt="3D Hilbert curves animation grid for nbits 1..4" />
</p>

<p align="center">
    <sub>2D Hilbert curves (nbits 1..5) and 3D Hilbert curves (nbits 1..4, animated).</sub>
</p>

---

This project is **performance-first** and **implemented entirely in Python**. The hot kernels are JIT-compiled with Numba and tuned for:

- **Branchless, fully unrolled inner loops**
- **SIMD via LLVM vector intrinsics**
- **Small, L1-cache-friendly lookup tables (LUTs)**
- **Reduced dependency chains for better ILP, e.g., state-independent gather**
- **Optional multi-threading for batch operations**

It provides both convenient Python APIs and *kernel accessors* designed to be embedded into other Numba kernels.

## Performance

**HilbertSFC** is orders of magnitude faster than existing Python implementations. It also outperforms the **Fast Hilbert** implementation in Rust by a factor of ~7x. In fact, **HilbertSFC** takes only ~8 CPU cycles per point for 2D encode/decode of 32-bit coordinates.

#### 2D Points - Random, `nbits=32`, `n=5,000,000`

| Implementation | ns/pt (enc) | ns/pt (dec) | Mpts/s (enc) | Mpts/s (dec) |
| --- | ---: | ---: | ---: | ---: |
| 🔥**hilbertsfc (multi-threaded)** | 0.53 | 0.57 | 1883.52 | 1742.08 |
| 🔥**hilbertsfc (Python)** | 1.84 | 1.88 | 543.60 | 532.77 |
| [fast_hilbert (Rust)](https://crates.io/crates/fast_hilbert) | 13.71 | 13.47 | 72.92 | 74.23 |
| [hilbert_2d (Rust)](https://crates.io/crates/hilbert_2d) | 121.23 | 101.34 | 8.25 | 9.87 |
| [hilbert-bytes (Python)](https://pypi.org/project/hilbert-bytes/) | 2997.51 | 2642.86 | 0.334 | 0.378 |
| [numpy-hilbert-curve (Python)](https://pypi.org/project/numpy-hilbert-curve/) | 7606.88 | 5075.58 | 0.131 | 0.197 |
| [hilbertcurve (Python)](https://pypi.org/project/hilbertcurve/) | 14355.76 | 10411.20 | 0.0697 | 0.0961 |

> **System info:** Intel Core Ultra 7 258v, Ubuntu 24.04.4, Python 3.12.12, Numba 0.63.1

Additional benchmarks and details are available in the [benchmark.md](benchmark.md).

## Quickstart

### Installation

With pip:

```bash
pip install hilbertsfc
```

Or with uv:

```bash
uv add hilbertsfc
```

### Usage

Hilbert curves map multi-dimensional integer coordinates onto a single scalar index while preserving spatial locality. `hilbertsfc` provides an encode and decode API for 2D and 3D coordinates that support both scalar values and vectorized array inputs.

The `nbits` parameter specifies the number of bits per coordinate, defining the grid domain as `[0, 2**nbits)`. If omitted, it's inferred from the input array dtype (for arrays) or defaults to the maximum (32 for 2D, 21 for 3D).

#### Scalar 2D

Encode a single `(x, y)` coordinate into a Hilbert index, and decode it back:

```python
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

index = hilbert_encode_2d(17, 23, nbits=10)  # index = 534
x, y = hilbert_decode_2d(index, nbits=10)    # x, y = (17, 23)
```

#### Batch 2D

The same functions operate elementwise on NumPy arrays, preserving shape and avoiding Python loops:

```python
import numpy as np
from hilbertsfc import hilbert_decode_2d, hilbert_encode_2d

xs = np.arange(1024, dtype=np.uint16)
ys = xs[::-1]

indices = hilbert_encode_2d(xs, ys, nbits=10)    # shape (1024,), dtype uint32
xs2, ys2 = hilbert_decode_2d(indices, nbits=10)  # xs2 = xs, ys2 = ys
```

This is the preferred use for high-throughput workloads. It can be further accelerated with `parallel=True`.

#### Batch 3D

3D works identically, mapping `(x, y, z)` coordinates to a single Hilbert index:

```python
import numpy as np
from hilbertsfc import hilbert_decode_3d, hilbert_encode_3d

nbits = 10
n = 10_000
rng = np.random.default_rng(0)

xs = rng.integers(0, 2**nbits, size=n, dtype=np.uint32)
ys = rng.integers(0, 2**nbits, size=n, dtype=np.uint32)
zs = rng.integers(0, 2**nbits, size=n, dtype=np.uint32)

indices = hilbert_encode_3d(xs, ys, zs, nbits=nbits)      # shape (10000,), dtype uint32
xs2, ys2, zs2 = hilbert_decode_3d(indices, nbits=nbits)   # xs2 = xs, ys2 = ys, zs2 = zs
```

This is can be useful for applications like 3D spatial indexing, volumetric data processing, compression, and more.

### Demo Notebook

For more examples, see the [demo notebook](notebooks/hilbertsfc_demo.ipynb) which includes visualizations of the curves and embedding the kernels into custom Numba code.

## API notes

- `nbits` specifies the number of bits per coordinate. Coordinates must be in `[0, 2**nbits)`. A tighter `nbits` improves performance and reduces output dtypes. Excess bits are ignored.
- Hilbert indices obtained with a certain `nbits` are compatible with those from another `nbits`, given that the coordinates are within the valid range. This is because the kernels resolve the starting state parity to ensure compatibility.
- The batched API accepts arbitrary shapes and preserves the input shape. The requirement is that inputs/outputs support a *zero-copy* 1D view. Most strided views are supported but they can reduce performance since the kernels are close to memory-bandwidth bound.
- You can pass `out=...` buffers for batch encode, and `out_xs/out_ys/out_zs` for batch decode. This can for example be useful to write into memory-mapped arrays or to reuse buffers across multiple calls.
- `parallel=True` dispatches the parallel version of the kernel (when available). The number of threads can be controlled with the environment variable `NUMBA_NUM_THREADS` or during runtime with `numba.set_num_threads()`.

## Documentation

[Documentation](https://remcofl.github.io/HilbertSFC/) is hosted online. It includes a quick start guide, and API reference.

To serve the docs locally:

```bash
uv run --no-dev --group docs mkdocs serve
```

Build a static site into `site/`:

```bash
uv run --no-dev --group docs mkdocs build
```

## Embedding kernels in your own Numba code

While the main API is designed for ease of use, the package also provides *kernel accessors* that expose the scalar encode/decode kernels. This allows you to embed the Hilbert curve logic directly into your own Numba kernels, enabling further optimizations like loop fusion and reduced Python call overhead.

### Example embedding the 2D encode kernel

```python
import numpy as np
import numba as nb

from hilbertsfc import get_hilbert_encode_2d_kernel

encode_2d_10 = get_hilbert_encode_2d_kernel(nbits=10)

@nb.njit
def encode_many(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.empty(xs.shape, dtype=np.uint32)
    for i in range(xs.size):
        out[i] = encode_2d_10(xs[i], ys[i])
    return out
```

The same pattern works for decode and for 3D kernels.

## Cache control

If you want to clear cached kernels and lookup tables (e.g., for benchmarking or testing), you can use the `clear_all_caches()` function:

```python
from hilbertsfc import clear_all_caches

clear_all_caches()
```
