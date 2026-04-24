# Advanced usage

This page covers lower-level integration patterns and performance-oriented usage beyond the basic API usage shown in the [Quick start](quickstart.md).

## Embedding kernels in custom Numba code

While the main API is designed for ease of use, the package also provides *kernel accessors* that expose the scalar encode/decode kernels. This allows you to embed Hilbert/Morton curve logic directly into your own `@numba.njit` code.

This is useful for fusing surrounding numerical operations such as quantization, tiling, or other pre- and post-processing steps into a single compiled loop. This can improve performance by keeping intermediate data in registers, reducing memory movement, avoiding temporary arrays, and reducing Python call overhead.


### Example: tiling / quantization inside a fused Numba loop

A common use case is to quantize (floating-point) coordinates into a fixed tile grid. With kernel accessors, you can perform the quantization and Hilbert encoding inside one compiled loop.

This example compares:

- **sequential**: quantize `x/y` in NumPy (three intermediate arrays), then call `hilbert_encode_2d`.
- **fused**: quantize and encode inside one `@numba.njit` loop.

```python
import time

import numba as nb
import numpy as np

from hilbertsfc import get_hilbert_encode_2d_kernel, hilbert_encode_2d

NBITS = 5
TILE_SIZE = 16
N = 2_000_000

encode_2d = get_hilbert_encode_2d_kernel(nbits=NBITS)
rng = np.random.default_rng(0)

points_xy = rng.integers(0, (2**NBITS) * TILE_SIZE, size=(N, 2), dtype=np.int32)


@nb.njit
def encode_points_to_tiles(points_xy: np.ndarray) -> np.ndarray:
    out = np.empty(points_xy.shape[0], dtype=np.uint32)
    for i in range(points_xy.shape[0]):
        x = points_xy[i, 0] // TILE_SIZE
        y = points_xy[i, 1] // TILE_SIZE
        out[i] = encode_2d(x, y)
    return out


def encode_sequential(points_xy: np.ndarray) -> np.ndarray:
    tx = (points_xy[:, 0] // TILE_SIZE)
    ty = (points_xy[:, 1] // TILE_SIZE)
    return hilbert_encode_2d(tx, ty, nbits=NBITS)


def bench(fn, *args, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


# Warm up compilation
encode_points_to_tiles(points_xy[:8])

# Verify both paths produce the same result
assert np.array_equal(encode_sequential(points_xy), encode_points_to_tiles(points_xy))

t_seq = bench(encode_sequential, points_xy)
t_fused = bench(encode_points_to_tiles, points_xy)

print(f"sequential: {t_seq * 1e3:.1f} ms")
print(f"fused:      {t_fused * 1e3:.1f} ms")
print(f"speedup:    {t_seq / t_fused:.2f}x")

```

Output:

```text
sequential: 9.3 ms
fused:      2.7 ms
speedup:    3.42x
```

/// admonition | Performance
    type: info

The performance benefit depends on how well the surrounding code can be optimized by the compiler, and may vary with the workload, data size, and hardware. In the example above it is crucial to insert a constant `TILE_SIZE` enabling the compiler to strength reduce the division. Benchmarking/profiling is recommended to verify the speedup for your specific use case.
///

## PyTorch and `torch.compile`

HilbertSFC's PyTorch API is designed to integrate naturally with PyTorch and its [compiler stack](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). Unlike the Numba backend, which exposes scalar kernels via *kernel accessors* for manual embedding into `@numba.njit` code, the PyTorch API lets you compose tensor operations normally and wrap the code with a [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) to optimize the surrounding tensor computation graph.

HilbertSFC is designed to work cleanly with this model. The main requirement for a graph-break-free capture is that [`precache_compile_luts`][hilbertsfc.torch.precache_compile_luts] must be called before entering the compiled region. This ensures the kernel lookup tables are loaded and cached outside the compiled region. This avoids graph breaks and extra overhead, and is required for `fullgraph=True`. See the [`torch.compile` example in the Quick start](quickstart.md/#use-with-torchcompile).

### Automatic backend selection

By default the dispatcher chooses the most suitable implementation for the given tensors and execution context. On CPU, the default backend is Numba, but inside `torch.compile` it switches to the Torch backend, since the Numba path is not `torch.compile`-friendly. This allows HilbertSFC and surrounding ATen operations to be optimized together by Torch Inductor (or other compiler backends).

On CUDA and ROCm, both the Torch backend and the Triton backend can participate in compiled graphs and be fused with surrounding tensor operations. By default, the Triton backend is used when available and when the tensors are contiguous; otherwise execution falls back to the Torch backend.

The backends can also be manually selected with the `cpu_backend` or `gpu_backend` options, if you want to force a specific implementation. See the [API reference](api/torch.md) for more details.

## Cache behavior

HilbertSFC uses caches to avoid repeated setup overhead, and in normal use you usually do not need to manage them manually. Cache clearing is mainly useful in benchmarks, tests, or when you want to reset state explicitly.

For the core process-wide caches, use [`clear_all_caches`][hilbertsfc.clear_all_caches]. This clears the registered Numba kernel-builder caches and the process-wide LUT caches.

Torch-side LUT tensor caches are managed separately on a per-device basis and are not affected by [`clear_all_caches`][hilbertsfc.clear_all_caches]. You can free the associated device memory with [`clear_torch_lut_caches`][hilbertsfc.torch.clear_torch_lut_caches].
