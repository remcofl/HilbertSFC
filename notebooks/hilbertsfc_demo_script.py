# ruff: noqa: E402

# %% [markdown]
# # HilbertSFC demo: visualize the curve and use the kernels
#
# This notebook demonstrates:
#
# - 2D/3D Hilbert curve visualization (gradient path)
# - Array encode/decode usage
# - How to embed the scalar kernels inside your own Numba kernels

# %%
import sys
from importlib.metadata import version

print("python", sys.version)
print("hilbertsfc", version("hilbertsfc"))
print("numba", version("numba"))
print("numpy", version("numpy"))

# %% [markdown]
# ## 2D curve visualization (path)
#
# We generate indices `0..(2^(2*nbits)-1)` and decode to `(x, y)`.

# %%
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from hilbertsfc import hilbert_decode_2d

# 2D visuals
NBITS_2D = 5  # 32x32 => 1024 points

nbits = NBITS_2D  # 2D grid: side = 2**nbits
n = 2 ** (2 * nbits)
indices = np.arange(n, dtype=np.uint32)
xs, ys = hilbert_decode_2d(indices, nbits)

xs = xs.astype(np.int32)
ys = ys.astype(np.int32)

fig, ax = plt.subplots(figsize=(2, 2))

# Transparent background (blends into markdown light/dark)
fig.patch.set_facecolor("none")
fig.patch.set_alpha(0.0)
ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

pts2d = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
segs2d = np.stack([pts2d[:-1], pts2d[1:]], axis=1)
seg_t = np.linspace(0.0, 1.0, segs2d.shape[0], dtype=np.float32)

lc2d = LineCollection(
    cast(Any, segs2d),
    cmap="turbo",
    linewidths=2.0,
    alpha=1,
    antialiased=True,
    capstyle="round",
    joinstyle="round",
)
lc2d.set_array(seg_t)
ax.add_collection(lc2d)

# Avoid clipping at the borders (line width needs a bit of breathing room)
side = float(2**nbits - 1)
pad = 0.75
ax.set_xlim(-pad, side + pad)
ax.set_ylim(-pad, side + pad)

# plt.title(f"2D Hilbert curve path (nbits={nbits}, points={n})")
ax.set_aspect("equal", adjustable="box")
ax.margins(0)
ax.set_axis_off()
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plt.show()

# %% [markdown]
# ## 3D curve visualization (animated)
#
# We plot the full 3D traversal as a single curve, color it from start -> end, and rotate the view to show the structure.

# %%
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib import animation
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from hilbertsfc import hilbert_decode_3d

# 3D visuals
NBITS_3D = 3  # 8x8x8 => 512 points

# Animation tuning
# - Full 360° wrap-around
# - Increase Matplotlib's embed limit to avoid dropped frames
ANIM_TOTAL_DEG = 360.0
ANIM_FPS = 30
ANIM_DURATION_S = 12.0
ANIM_FRAMES = int(ANIM_FPS * ANIM_DURATION_S)
ANIM_INTERVAL_MS = int(round(1000 / ANIM_FPS))
ANIM_DPI = 90

# Default is ~20MB. Raise it so JSHTML can embed more frames at higher quality.
# If the notebook gets too heavy, set this back to 20.
ANIM_EMBED_LIMIT_MB = 80
mpl.rcParams["animation.embed_limit"] = float(ANIM_EMBED_LIMIT_MB)

print(
    f"3D anim settings: frames={ANIM_FRAMES}, interval_ms={ANIM_INTERVAL_MS}, dpi={ANIM_DPI}, total_deg={ANIM_TOTAL_DEG}, embed_limit_mb={ANIM_EMBED_LIMIT_MB}"
)

nbits3d = NBITS_3D
n3d = 2 ** (3 * nbits3d)
indices3d = np.arange(n3d, dtype=np.uint32)
xs3d, ys3d, zs3d = hilbert_decode_3d(indices3d, nbits3d)
xs3d, ys3d, zs3d = hilbert_decode_3d(indices3d, nbits3d)

pts = np.column_stack(
    (xs3d.astype(np.float32), ys3d.astype(np.float32), zs3d.astype(np.float32))
)
segs = np.stack([pts[:-1], pts[1:]], axis=1)
seg_t = np.linspace(0.0, 1.0, segs.shape[0], dtype=np.float32)
seg_mid = (segs[:, 0, :] + segs[:, 1, :]) * 0.5

fig = plt.figure(figsize=(4, 4), dpi=ANIM_DPI)
ax = fig.add_subplot(111, projection="3d")

# Transparent background (blends into markdown light/dark)
fig.patch.set_facecolor("none")
fig.patch.set_alpha(0.0)
ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
rgba_clear = to_rgba("white", 0.0)

cast(Any, ax.xaxis).set_pane_color(rgba_clear)
cast(Any, ax.yaxis).set_pane_color(rgba_clear)
cast(Any, ax.zaxis).set_pane_color(rgba_clear)

lc = Line3DCollection(
    cast(Any, segs),
    cmap="turbo",
    linewidths=4,
    alpha=1,
    antialiased=True,
    capstyle="round",
    joinstyle="round",
)
lc.set_array(seg_t)
ax.add_collection(lc)

side = float(2**nbits3d - 1)
ax.set_xlim(0.0, side)
ax.set_ylim(0.0, side)
ax.set_zlim(0.0, side)

cast(Any, ax).set_box_aspect((1, 1, 1))

ax.set_axis_off()
# Remove the default whitespace around a 3D axis
ax.set_position((0.0, 0.0, 1.0, 1.0))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)


def _update(frame: int):
    azim = ANIM_TOTAL_DEG * frame / ANIM_FRAMES
    cast(Any, ax).view_init(elev=15, azim=azim)

    # Matplotlib 3D uses a painter's algorithm (no z-buffer). With alpha < 1,
    # we need to draw back-to-front to avoid "far" segments appearing on top.
    # Re-sorting per-frame keeps the blending/occlusion consistent as we rotate.
    if seg_mid.size:
        _, _, tz = proj3d.proj_transform(
            seg_mid[:, 0],
            seg_mid[:, 1],
            seg_mid[:, 2],
            cast(Any, ax).get_proj(),
        )
        # In this projection, "near" tends to have smaller (more negative) tz,
        # so we sort descending to draw far -> near.
        order = np.argsort(tz)[::-1]
        lc.set_segments(cast(Any, segs[order]))
        lc.set_array(seg_t[order])

    return (lc,)


anim = animation.FuncAnimation(
    fig,
    _update,
    frames=ANIM_FRAMES,
    interval=ANIM_INTERVAL_MS,
    blit=False,
    cache_frame_data=False,
)

raw_html = anim.to_jshtml()
plt.close(fig)

# CSS to remove extra padding/margins around the animation output
css = """
<style>
div.output, div.output_area, div.output_subarea { padding: 0 !important; }
.animation { margin: 0 !important; }
.animation canvas { background: transparent !important; display: block; }
</style>
"""
display(
    HTML(
        css
        + "<div style='line-height:0; background: transparent; padding:0; margin:0'>"
        + raw_html
        + "</div>"
    )
)

# %% [markdown]
# ## Embedding the scalar kernel (tiling / quantization)
#
# The accessors like `get_hilbert_encode_2d_kernel(nbits)` return a Numba-compiled *scalar* kernel
# that you can call inside your own `@numba.njit` loop.
#
# A common application is to quantize coordinates into a fixed grid of tiles ("binning") and
# then order points by their Hilbert index:
#
# - `tile_x = (x - min_x) // tile_size`
# - `tile_y = (y - min_y) // tile_size`
# - `h = hilbert_encode_2d(tile_x, tile_y, nbits)`
#
# With `NBITS_2D = 5`, the tile grid is `32 x 32` (tile indices `0..31`).

# %%
import numba as nb
import numpy as np

from hilbertsfc import get_hilbert_encode_2d_kernel, hilbert_encode_2d

NBITS_2D = 5

TILES_SIZE = 16
nbits = NBITS_2D  # tile coords will be in [0, 2**nbits)
encode_2d = get_hilbert_encode_2d_kernel(nbits)


@nb.njit
def encode_points_to_tiles(
    points_xy: np.ndarray, min_x: int, min_y: int, tile_size: int
) -> np.ndarray:
    out = np.empty(points_xy.shape[0], dtype=np.uint32)
    for i in range(points_xy.shape[0]):
        x = (points_xy[i, 0] - min_x) // tile_size
        y = (points_xy[i, 1] - min_y) // tile_size
        out[i] = encode_2d(x, y)
    return out


# Example: points in [0, 512) with tile_size=16 => tile coords in [0, 32)
points_xy = np.array(
    [
        [12, 13],
        [300, 40],
        [511, 511],
        [128, 256],
        [9, 9],
        [400, 120],
    ],
    dtype=np.int32,
)

min_x = 0
min_y = 0
h_tiles = encode_points_to_tiles(points_xy, min_x, min_y, TILES_SIZE)

# Cross-check: compute tile coords explicitly and use the array form
tile_xs = ((points_xy[:, 0] - min_x) // TILES_SIZE).astype(np.uint32)
tile_ys = ((points_xy[:, 1] - min_y) // TILES_SIZE).astype(np.uint32)
h_tiles_batch = hilbert_encode_2d(tile_xs, tile_ys, nbits)

print("tile_xs:", tile_xs)
print("tile_ys:", tile_ys)
print("hilbert (tiled):", h_tiles)
print("matches batch:", np.array_equal(h_tiles, h_tiles_batch))

order = np.argsort(h_tiles)
points_xy_ordered = points_xy[order]
print("order:", order)
print("points ordered by tiled Hilbert:\n", points_xy_ordered)


# %% [markdown]
# ### Timing: sequential NumPy ops vs fused Numba loop
#
# Below we compare two ways to compute tiled Hilbert indices for many points:
#
# 1. **Sequential NumPy ops**: compute `tile_xs/tile_ys` as separate arrays, then call `hilbert_encode_2d`.
# 2. **Fused Numba loop**: quantize + encode in one `@njit` loop (`encode_points_to_tiles`).
#
# The fused loop can be a win when you already have a Numba pipeline and want to avoid
# extra intermediate arrays and reduce memory traffic.


# %%
import time

from hilbertsfc import hilbert_encode_2d

rng = np.random.default_rng(0)

N_POINTS = 2_000_000
TILES_SIZE = 16

world_side = (2**nbits) * TILES_SIZE  # ensures tile coords are 0..(2**nbits-1)
points_big = rng.integers(0, world_side, size=(N_POINTS, 2), dtype=np.int32)

# Warm up Numba compilation (exclude compile time from timing)
_ = encode_points_to_tiles(points_big[:8], min_x, min_y, TILES_SIZE)


def _timeit(fn, repeats: int = 20) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _sequential_numpy() -> np.ndarray:
    tx = ((points_big[:, 0] - min_x) // TILES_SIZE).astype(np.uint32)
    ty = ((points_big[:, 1] - min_y) // TILES_SIZE).astype(np.uint32)
    return hilbert_encode_2d(tx, ty, nbits)


def _fused_numba() -> np.ndarray:
    return encode_points_to_tiles(points_big, min_x, min_y, TILES_SIZE)


h1 = _sequential_numpy()
h2 = _fused_numba()
print("equal:", np.array_equal(h1, h2))

t_seq = _timeit(_sequential_numpy)
t_fused = _timeit(_fused_numba)

print(f"sequential numpy: {t_seq * 1e3:.1f} ms")
print(f"fused numba:      {t_fused * 1e3:.1f} ms")
print(f"speedup:          {t_seq / t_fused:.2f}x")

# %%
