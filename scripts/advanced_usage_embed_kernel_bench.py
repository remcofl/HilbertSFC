# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hilbertsfc>=0.2.0",
#     "numba>=0.64.0",
#     "numpy>=2.4.3",
# ]
# ///

"""Script to generate output for docs/advanced-usage.md."""

import time

import numba as nb
import numpy as np

from hilbertsfc import get_hilbert_encode_2d_kernel, hilbert_encode_2d

NBITS = 5
TILE_SIZE = 17
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
    tx = points_xy[:, 0] // TILE_SIZE
    ty = points_xy[:, 1] // TILE_SIZE
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
