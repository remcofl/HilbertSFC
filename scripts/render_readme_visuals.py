# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hilbertsfc",
#     "matplotlib>=3.10.8",
#     "numpy>=2.3.5",
#     "pillow>=12.1.0",
#     "rich>=14.3.2",
# ]
#
# [tool.uv.sources]
# hilbertsfc = { git = "https://github.com/RFLeijenaar/HilbertSFC.git" }
# ///
"""Render README visuals.

Creates:
- docs/img/hilbert2d_grid.png  (nbits 1..5, turbo gradient)
- docs/img/hilbert3d.png       (nbits=3 example, turbo gradient; optional)
- docs/img/hilbert3d_grid.webp (animated grid, default nbits 1..4)

Run (from repo root):
    # As standalone script (within the repo)
    uv run scripts/render_readme_visuals.py

    # or within the project virtual environment (`uv sync --group scripts`)
    uv run python scripts/render_readme_visuals.py

    # or (if you've installed dependencies some other way)
    python scripts/render_readme_visuals.py

Optional:
    # Disable per-nbits linewidth scaling for the 2D grid
    uv run scripts/render_readme_visuals.py --no-2d-linewidth-scale

    # Render the static 3D still (off by default)
    uv run scripts/render_readme_visuals.py --render-3d-still

    # Use APNG instead of WebP (typically faster to encode)
    uv run scripts/render_readme_visuals.py --3d-grid-format apng

Notes:
    - GIF output is intentionally not supported.
    - Dependencies are declared in the inline script header above.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.axes3d import Axes3D

from hilbertsfc import hilbert_decode_2d, hilbert_decode_3d


@dataclass(frozen=True)
class RenderConfig:
    cmap: str = "turbo"
    line_width_2d: float = 2.0
    line_width_3d: float = 24
    alpha_2d: float = 0.95
    alpha_3d: float = 0.95
    scale_2d_linewidth_by_nbits: bool = True
    linewidth_scale_factor_2d: float = 2.0
    scale_3d_linewidth_by_nbits: bool = True
    linewidth_scale_factor_3d: float = 3.0
    # Micro-segment subdivision to keep a smooth gradient even for low nbits.
    subdiv_per_segment_2d: int = 40
    subdiv_per_segment_3d: int = 12
    subdiv_per_segment_3d_anim: int = 8

    anim_fps: int = 24
    # Increase frames for smoother motion while keeping speed manageable.
    # With anim_total_deg fixed at 360, doubling frames halves angular speed.
    anim_frames: int = 180
    anim_total_deg: float = 360.0
    anim_elev: float = 18.0
    anim_azim0: float = 35.0

    # 3D grid layout (tune for README aesthetics)
    grid_panel_width_3d: float = 2.6
    grid_fig_height_3d: float = 2.4
    grid_dpi_3d: int = 96
    grid_gap_3d: float = 0.003

    # Post-processing: crop each 3D panel to its content and recompose.
    # This removes mplot3d's large internal left/right margins.
    grid_recompose_panels_3d: bool = True
    grid_compose_gap_in_3d: float = 0.1
    grid_crop_margin_x_3d: int = 6
    grid_crop_margin_y_3d: int = 4

    # Animated WebP encoding (smaller than lossless APNG for most cases)
    webp_quality: int = 80
    webp_method: int = 6
    webp_lossless: bool = False


def _compose_gap_px_for_fig(cfg: RenderConfig, *, dpi: float) -> int:
    """Compute stitched-panel gap in pixels.

    Recomposition happens in pixel space (after cropping), but the user-facing
    setting is inches so it is stable across DPI/size changes.
    """

    return max(0, int(round(float(cfg.grid_compose_gap_in_3d) * float(dpi))))


def _segments_with_gradient(
    points: np.ndarray, subdiv_per_segment: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (segments, t) for a polyline with micro-segment subdivision.

    points: (N, D)
    segments: (M, 2, D)
    t: (M,) normalized 0..1 for coloring
    """
    if points.shape[0] < 2:
        raise ValueError("need at least 2 points")

    p0 = points[:-1]
    p1 = points[1:]
    base_segments = np.stack([p0, p1], axis=1)  # (S, 2, D)

    if subdiv_per_segment <= 1:
        segments = base_segments
        t = np.linspace(0.0, 1.0, segments.shape[0], dtype=np.float32)
        return segments, t

    # Subdivide each original segment into K micro-segments.
    s = base_segments.shape[0]
    k = int(subdiv_per_segment)

    # Interpolation parameter for micro-segments: [0..1)
    u0 = np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)  # (k,)
    u1 = np.linspace(1.0 / k, 1.0, k, endpoint=True, dtype=np.float32)  # (k,)

    # Expand to (s, k, D)
    start = base_segments[:, 0, :][:, None, :]
    end = base_segments[:, 1, :][:, None, :]

    a = start * (1.0 - u0[None, :, None]) + end * (u0[None, :, None])
    b = start * (1.0 - u1[None, :, None]) + end * (u1[None, :, None])

    segments = np.stack([a, b], axis=2).reshape(s * k, 2, points.shape[1])
    t = np.linspace(0.0, 1.0, segments.shape[0], dtype=np.float32)
    return segments, t


def _alpha_bbox(
    rgba: np.ndarray, *, alpha_threshold: int = 1
) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) bbox of pixels with alpha > threshold.

    Coordinates are inclusive-exclusive (like slicing). Returns None if empty.
    """

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected RGBA array")

    a = rgba[:, :, 3]
    mask = a > np.uint8(alpha_threshold)
    if not np.any(mask):
        return None

    ys, xs = np.where(mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return x0, y0, x1, y1


def render_2d_grid(out_path: Path, cfg: RenderConfig) -> None:
    nbits_list = [1, 2, 3, 4, 5]
    max_nbits = max(nbits_list)
    # Slightly shorter figure so the README section takes less vertical space.
    fig, axes = plt.subplots(1, len(nbits_list), figsize=(10, 2.4), dpi=200)

    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("render 2D grid", style="blue"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    )
    progress.start()
    task_id = progress.add_task("render", total=len(nbits_list))
    try:
        for ax, nbits in zip(axes, nbits_list, strict=True):
            n = 2 ** (2 * nbits)
            indices = np.arange(n, dtype=np.uint32)
            xs, ys = hilbert_decode_2d(indices, nbits=nbits)

            pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
            segs, t = _segments_with_gradient(pts, cfg.subdiv_per_segment_2d)

            if cfg.scale_2d_linewidth_by_nbits:
                # Keep nbits=max_nbits at the base linewidth, and scale each step down.
                lw = cfg.line_width_2d * (
                    cfg.linewidth_scale_factor_2d ** (max_nbits - nbits)
                )
            else:
                lw = cfg.line_width_2d

            lc = LineCollection(
                cast(Any, segs),
                cmap=cfg.cmap,
                linewidths=lw,
                alpha=cfg.alpha_2d,
                antialiased=True,
                capstyle="round",
                joinstyle="round",
            )
            lc.set_array(t)
            ax.add_collection(lc)

            side = float(2**nbits - 1)
            pad = 0.75
            ax.set_xlim(-pad, side + pad)
            ax.set_ylim(-pad, side + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            ax.set_facecolor((0, 0, 0, 0))
            progress.advance(task_id, 1)
    finally:
        progress.stop()

    # No subplot titles; keep side/bottom padding but reduce top margin.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def render_3d(out_path: Path, cfg: RenderConfig) -> None:
    nbits = 3
    n = 2 ** (3 * nbits)
    indices = np.arange(n, dtype=np.uint32)
    xs, ys, zs = hilbert_decode_3d(indices, nbits=nbits)

    pts = np.column_stack(
        (xs.astype(np.float32), ys.astype(np.float32), zs.astype(np.float32))
    )
    segs, t = _segments_with_gradient(pts, cfg.subdiv_per_segment_3d)

    lw = cfg.line_width_3d
    if cfg.scale_3d_linewidth_by_nbits:
        # Baseline is the configured linewidth at nbits=1.
        # Higher nbits => thinner lines (divide per step).
        lw = cfg.line_width_3d / (cfg.linewidth_scale_factor_3d ** (nbits - 1))

    fig = plt.figure(figsize=(4.2, 4.2), dpi=220)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    ax: Axes3D = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.set_facecolor((0, 0, 0, 0))

    lc = Line3DCollection(
        cast(Any, segs),
        cmap=cfg.cmap,
        linewidths=lw,
        alpha=cfg.alpha_3d,
        antialiased=True,
        capstyle="round",
        joinstyle="round",
    )
    lc.set_array(t)
    ax.add_collection(lc)

    side = float(2**nbits - 1)
    ax.set_xlim(0.0, side)
    ax.set_ylim(0.0, side)
    ax.set_zlim(0.0, side)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # Transparent panes
    rgba_clear = (1.0, 1.0, 1.0, 0.0)
    try:
        cast(Any, ax.xaxis).set_pane_color(rgba_clear)
        cast(Any, ax.yaxis).set_pane_color(rgba_clear)
        cast(Any, ax.zaxis).set_pane_color(rgba_clear)
    except Exception:
        pass

    ax.set_axis_off()
    ax.set_position((0.0, 0.0, 1.0, 1.0))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    # Nice static view
    ax.view_init(elev=18, azim=35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


def render_3d_grid_anim(
    out_path: Path,
    cfg: RenderConfig,
    nbits_list: list[int] | tuple[int, ...],
    *,
    scene_nbits: int,
) -> None:
    """Render a looping 3D animation grid.

    Supports APNG (animated PNG) and animated WebP.

    APNG is typically way faster to encode and tends to be robust with alpha.
    WebP is often smaller.
    """

    nbits_list = list(nbits_list)
    if not nbits_list:
        raise ValueError("nbits_list must not be empty")

    # Preserve order but drop duplicates.
    seen: set[int] = set()
    nbits_list = [n for n in nbits_list if not (n in seen or seen.add(n))]

    if any(n <= 0 for n in nbits_list):
        raise ValueError("nbits values must be positive")
    max_nbits = max(nbits_list)
    if scene_nbits < max_nbits:
        raise ValueError("scene_nbits must be >= max(nbits_list)")

    common_side = float(2**scene_nbits - 1)
    pad = 0.75

    # Keep the grid compact (less whitespace between panels) while maintaining
    # sufficient resolution for README display.
    fig_width_in = float(cfg.grid_panel_width_3d) * float(len(nbits_list))
    fig = plt.figure(
        figsize=(fig_width_in, cfg.grid_fig_height_3d),
        dpi=int(cfg.grid_dpi_3d),
    )
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    n_panels = len(nbits_list)
    gap = float(cfg.grid_gap_3d)
    panel_w = (1.0 - gap * (n_panels - 1)) / n_panels

    rgba_clear = to_rgba("white", 0.0)

    per_ax: list[tuple[np.ndarray, np.ndarray, np.ndarray, Line3DCollection, Any]] = []

    for i, nbits in enumerate(nbits_list):
        x0 = i * (panel_w + gap)
        ax: Axes3D = cast(
            "Axes3D", fig.add_axes((x0, 0.0, panel_w, 1.0), projection="3d")
        )
        ax.set_facecolor((0, 0, 0, 0))

        n = 2 ** (3 * nbits)
        indices = np.arange(n, dtype=np.uint32)
        xs, ys, zs = hilbert_decode_3d(indices, nbits=nbits)

        # Map integer grid coordinates to *cell centers* in the max-nbits cube.
        # For a unit cube this corresponds to centers at:
        # - nbits=1: 0.25, 0.75
        # - nbits=2: 0.125, 0.375, 0.625, 0.875
        # Here we scale to 0..common_side.
        cell_scale = np.float32(common_side / (2**nbits))
        pts = (
            np.column_stack(
                (xs.astype(np.float32), ys.astype(np.float32), zs.astype(np.float32))
            )
            + np.float32(0.5)
        ) * cell_scale
        segs, t = _segments_with_gradient(pts, cfg.subdiv_per_segment_3d_anim)
        seg_mid = (segs[:, 0, :] + segs[:, 1, :]) * 0.5

        lw = cfg.line_width_3d
        if cfg.scale_3d_linewidth_by_nbits:
            lw = cfg.line_width_3d / (cfg.linewidth_scale_factor_3d ** (nbits - 1))

        lc = Line3DCollection(
            cast(Any, segs),
            cmap=cfg.cmap,
            linewidths=lw,
            alpha=cfg.alpha_3d,
            antialiased=True,
            capstyle="round",
            joinstyle="round",
        )
        lc.set_array(t)
        ax.add_collection(lc)

        ax.set_xlim(-pad, common_side + pad)
        ax.set_ylim(-pad, common_side + pad)
        ax.set_zlim(-pad, common_side + pad)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        try:
            cast(Any, ax.xaxis).set_pane_color(rgba_clear)
            cast(Any, ax.yaxis).set_pane_color(rgba_clear)
            cast(Any, ax.zaxis).set_pane_color(rgba_clear)
        except Exception:
            pass

        ax.set_axis_off()
        per_ax.append((segs, t, seg_mid, lc, ax))

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def _update(frame: int):
        azim = cfg.anim_azim0 + cfg.anim_total_deg * frame / max(cfg.anim_frames, 1)

        artists = []
        for segs, t, seg_mid, lc, ax in per_ax:
            ax.view_init(elev=cfg.anim_elev, azim=azim)

            # Matplotlib 3D uses a painter's algorithm for collections.
            # Re-sorting per-frame improves occlusion ordering as we rotate.
            if seg_mid.size:
                _, _, tz = proj3d.proj_transform(
                    seg_mid[:, 0],
                    seg_mid[:, 1],
                    seg_mid[:, 2],
                    ax.get_proj(),
                )
                order = np.argsort(tz)[::-1]
                lc.set_segments(cast(Any, segs[order]))
                lc.set_array(t[order])

            artists.append(lc)

        return tuple(artists)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix not in {".png", ".webp"}:
        raise ValueError("3D grid animation output must be .png (APNG) or .webp")

    duration_ms = int(round(1000 / max(cfg.anim_fps, 1)))
    frames: list[Image.Image] = []

    # Ensure we have a canvas and it is ready.
    fig.canvas.draw()

    # Compute fixed per-panel crop boxes across a few sample frames.
    # Recompose panels to avoid large internal left/right margins from mplot3d.
    sample_frames = sorted(
        {
            0,
            int(cfg.anim_frames) // 4,
            int(cfg.anim_frames) // 2,
            (3 * int(cfg.anim_frames)) // 4,
            max(int(cfg.anim_frames) - 1, 0),
        }
    )

    rgba0 = np.asarray(cast(Any, fig.canvas).buffer_rgba()).copy()
    h0, w0 = rgba0.shape[:2]

    # Panel pixel extents based on our normalized add_axes placement.
    panel_px_boxes: list[tuple[int, int]] = []  # [(x0, x1), ...]
    for i in range(n_panels):
        x0_rel = i * (panel_w + gap)
        x1_rel = x0_rel + panel_w
        x0_px = int(round(x0_rel * w0))
        x1_px = int(round(x1_rel * w0))
        # Ensure monotonic increasing and within bounds.
        x0_px = max(0, min(w0, x0_px))
        x1_px = max(0, min(w0, x1_px))
        if x1_px < x0_px:
            x0_px, x1_px = x1_px, x0_px
        panel_px_boxes.append((x0_px, x1_px))

    panel_crop_boxes: list[tuple[int, int, int, int] | None] = [None] * n_panels
    for fi in sample_frames:
        _update(fi)
        fig.canvas.draw()
        rgba0 = np.asarray(cast(Any, fig.canvas).buffer_rgba()).copy()
        for pi, (px0, px1) in enumerate(panel_px_boxes):
            panel = rgba0[:, px0:px1, :]
            bb = _alpha_bbox(panel)
            if bb is None:
                continue
            # Convert panel-local bbox to panel-local crop box; we union per panel.
            prev = panel_crop_boxes[pi]
            if prev is None:
                panel_crop_boxes[pi] = bb
            else:
                x0, y0, x1, y1 = prev
                bx0, by0, bx1, by1 = bb
                panel_crop_boxes[pi] = (
                    min(x0, bx0),
                    min(y0, by0),
                    max(x1, bx1),
                    max(y1, by1),
                )

    if cfg.grid_recompose_panels_3d:
        # Expand each crop box with a safety margin.
        mx = int(cfg.grid_crop_margin_x_3d)
        my = int(cfg.grid_crop_margin_y_3d)
        for i, bb in enumerate(panel_crop_boxes):
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            ph, pw = h0, panel_px_boxes[i][1] - panel_px_boxes[i][0]
            panel_crop_boxes[i] = (
                max(0, x0 - mx),
                max(0, y0 - my),
                min(pw, x1 + mx),
                min(ph, y1 + my),
            )
    else:
        # Fallback: whole-frame vertical crop only.
        whole_bb = _alpha_bbox(rgba0)
        if whole_bb is not None:
            _, y0, _, y1 = whole_bb
            my = int(cfg.grid_crop_margin_y_3d)
            panel_crop_boxes = [None] * n_panels
            panel_px_boxes = [(0, w0)]
            panel_crop_boxes[0] = (0, max(0, y0 - my), w0, min(h0, y1 + my))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("render 3D grid", style="blue"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("<", style="cyan"),
        TimeRemainingColumn(),
        transient=True,
    )
    progress.start()
    task_id = progress.add_task("render", total=int(cfg.anim_frames))
    try:
        for frame_idx in range(int(cfg.anim_frames)):
            _update(frame_idx)
            fig.canvas.draw()
            rgba = np.asarray(cast(Any, fig.canvas).buffer_rgba()).copy()

            if cfg.grid_recompose_panels_3d:
                # Crop each panel slice and stitch them horizontally.
                cropped_panels: list[np.ndarray] = []
                for pi, (px0, px1) in enumerate(panel_px_boxes):
                    panel = rgba[:, px0:px1, :]
                    bb = panel_crop_boxes[pi]
                    if bb is not None:
                        x0, y0, x1, y1 = bb
                        panel = panel[y0:y1, x0:x1, :]
                    cropped_panels.append(panel)

                if cropped_panels:
                    gap_px = _compose_gap_px_for_fig(cfg, dpi=float(fig.dpi))
                    out_h = max(p.shape[0] for p in cropped_panels)
                    out_w = sum(p.shape[1] for p in cropped_panels) + gap_px * (
                        len(cropped_panels) - 1
                    )
                    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
                    x = 0
                    for p in cropped_panels:
                        y = (out_h - p.shape[0]) // 2
                        out[y : y + p.shape[0], x : x + p.shape[1], :] = p
                        x += p.shape[1] + gap_px
                    rgba = out
            else:
                # Single crop box (panel_crop_boxes[0])
                bb = panel_crop_boxes[0]
                if bb is not None:
                    x0, y0, x1, y1 = bb
                    rgba = rgba[y0:y1, x0:x1, :]

            # Copy into a standalone Image (Matplotlib reuses the canvas buffer).
            frames.append(Image.fromarray(rgba, mode="RGBA"))
            progress.advance(task_id, 1)
    finally:
        progress.stop()

    plt.close(fig)

    if not frames:
        raise RuntimeError("no frames rendered")

    # Saving settings:
    # - loop=0 => infinite looping
    # - disposal=2 => clear to background between frames
    # - blend=0 (APNG) => source replace (no compositing)
    if suffix == ".png":
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_ms,
            disposal=2,
            blend=0,
            optimize=False,
        )
    elif suffix == ".webp":
        # WebP supports alpha and (often) yields much smaller animated assets.
        # Note: some renderers may still show halos with heavy transparency.
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_ms,
            lossless=bool(cfg.webp_lossless),
            quality=int(cfg.webp_quality),
            method=int(cfg.webp_method),
        )
    else:
        raise AssertionError(f"unexpected suffix: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README visualization images.")

    def _parse_nbits_list(raw: str) -> list[int]:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            raise argparse.ArgumentTypeError(
                "expected a comma-separated list like '1,2,3'"
            )
        try:
            return [int(p) for p in parts]
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                "nbits list must contain only integers"
            ) from e

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: docs/img under repo root).",
    )
    parser.add_argument(
        "--2d-linewidth",
        type=float,
        default=RenderConfig.line_width_2d,
        help="Base 2D linewidth (used for nbits=5 when scaling enabled).",
    )
    parser.add_argument(
        "--2d-linewidth-scale",
        dest="scale_2d_linewidth_by_nbits",
        action=argparse.BooleanOptionalAction,
        default=RenderConfig.scale_2d_linewidth_by_nbits,
        help="Enable/disable per-nbits linewidth scaling for the 2D grid.",
    )
    parser.add_argument(
        "--2d-linewidth-scale-factor",
        type=float,
        default=RenderConfig.linewidth_scale_factor_2d,
        help="Linewidth multiplier per step down in nbits when scaling is enabled.",
    )

    parser.add_argument(
        "--render-2d-grid",
        dest="render_2d_grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable rendering the 2D nbits=1..5 grid image.",
    )
    parser.add_argument(
        "--render-3d-still",
        dest="render_3d_still",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable rendering the static 3D example image.",
    )
    parser.add_argument(
        "--render-3d-grid",
        "--3d-grid-gif",
        dest="render_3d_grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable rendering the 3D grid animation (APNG/WebP).",
    )
    parser.add_argument(
        "--3d-grid-format",
        choices=("apng", "webp"),
        default="webp",
        help="Animation format for the 3D grid (default: webp; apng is faster to encode).",
    )
    parser.add_argument(
        "--3d-grid-nbits",
        type=_parse_nbits_list,
        default=[1, 2, 3, 4],
        help="Comma-separated nbits panels for the 3D grid (e.g. '1,2' for quick renders).",
    )
    parser.add_argument(
        "--3d-grid-scene-nbits",
        type=int,
        default=4,
        help="The cube resolution used for the 3D grid scene scaling (keeps size consistent when rendering subsets).",
    )
    parser.add_argument(
        "--3d-grid-panel-width",
        type=float,
        default=RenderConfig.grid_panel_width_3d,
        help="Panel width (inches) for each 3D grid cell (default: 2.6).",
    )
    parser.add_argument(
        "--3d-grid-fig-height",
        type=float,
        default=RenderConfig.grid_fig_height_3d,
        help="Figure height (inches) for the 3D grid (default: 2.2).",
    )
    parser.add_argument(
        "--3d-grid-dpi",
        type=int,
        default=RenderConfig.grid_dpi_3d,
        help="DPI for the 3D grid render (higher = sharper but slower; default: 150).",
    )
    parser.add_argument(
        "--3d-grid-gap",
        type=float,
        default=RenderConfig.grid_gap_3d,
        help="Horizontal gap between 3D grid panels as a fraction of figure width (default: 0.003).",
    )
    parser.add_argument(
        "--3d-grid-recompose",
        dest="grid_recompose_panels_3d",
        action=argparse.BooleanOptionalAction,
        default=RenderConfig.grid_recompose_panels_3d,
        help="Crop each 3D panel to its content and stitch panels (removes huge left/right margins).",
    )
    parser.add_argument(
        "--3d-grid-compose-gap-in",
        type=float,
        default=RenderConfig.grid_compose_gap_in_3d,
        help="Gap between panels after recomposition, in inches (default: 0.1).",
    )
    parser.add_argument(
        "--3d-grid-crop-margin-x",
        type=int,
        default=RenderConfig.grid_crop_margin_x_3d,
        help="Extra pixels to keep on left/right when cropping each panel (default: 6).",
    )
    parser.add_argument(
        "--3d-grid-crop-margin-y",
        type=int,
        default=RenderConfig.grid_crop_margin_y_3d,
        help="Extra pixels to keep on top/bottom when cropping each panel (default: 4).",
    )

    parser.add_argument(
        "--webp-quality",
        type=int,
        default=RenderConfig.webp_quality,
        help="Animated WebP quality (0..100, higher = larger; default: 80).",
    )
    parser.add_argument(
        "--webp-method",
        type=int,
        default=RenderConfig.webp_method,
        help="Animated WebP method (0..6, higher = slower/better; default: 6).",
    )
    parser.add_argument(
        "--webp-lossless",
        dest="webp_lossless",
        action=argparse.BooleanOptionalAction,
        default=RenderConfig.webp_lossless,
        help="Enable/disable lossless WebP (default: disabled).",
    )
    parser.add_argument(
        "--anim-fps",
        type=int,
        default=RenderConfig.anim_fps,
        help="FPS for the 3D grid animation (lower = slower).",
    )
    parser.add_argument(
        "--anim-frames",
        type=int,
        default=RenderConfig.anim_frames,
        help="Number of frames for the 3D grid animation.",
    )

    parser.add_argument(
        "--3d-linewidth",
        type=float,
        default=RenderConfig.line_width_3d,
        help="Base 3D linewidth at nbits=1 (used when scaling enabled).",
    )
    parser.add_argument(
        "--3d-linewidth-scale",
        dest="scale_3d_linewidth_by_nbits",
        action=argparse.BooleanOptionalAction,
        default=RenderConfig.scale_3d_linewidth_by_nbits,
        help="Enable/disable per-nbits linewidth scaling for 3D renders.",
    )
    parser.add_argument(
        "--3d-linewidth-scale-factor",
        type=float,
        default=RenderConfig.linewidth_scale_factor_3d,
        help="Linewidth multiplier per step down in nbits when 3D scaling is enabled.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir if args.out_dir is not None else (root / "docs" / "img")

    cfg = RenderConfig(
        line_width_2d=float(args.__dict__["2d_linewidth"]),
        scale_2d_linewidth_by_nbits=bool(args.scale_2d_linewidth_by_nbits),
        linewidth_scale_factor_2d=float(args.__dict__["2d_linewidth_scale_factor"]),
        line_width_3d=float(args.__dict__["3d_linewidth"]),
        scale_3d_linewidth_by_nbits=bool(args.scale_3d_linewidth_by_nbits),
        linewidth_scale_factor_3d=float(args.__dict__["3d_linewidth_scale_factor"]),
        anim_fps=int(args.anim_fps),
        anim_frames=int(args.anim_frames),
        grid_panel_width_3d=float(args.__dict__["3d_grid_panel_width"]),
        grid_fig_height_3d=float(args.__dict__["3d_grid_fig_height"]),
        grid_dpi_3d=int(args.__dict__["3d_grid_dpi"]),
        grid_gap_3d=float(args.__dict__["3d_grid_gap"]),
        grid_recompose_panels_3d=bool(args.__dict__["grid_recompose_panels_3d"]),
        grid_compose_gap_in_3d=float(args.__dict__["3d_grid_compose_gap_in"]),
        grid_crop_margin_x_3d=int(args.__dict__["3d_grid_crop_margin_x"]),
        grid_crop_margin_y_3d=int(args.__dict__["3d_grid_crop_margin_y"]),
        webp_quality=int(args.__dict__["webp_quality"]),
        webp_method=int(args.__dict__["webp_method"]),
        webp_lossless=bool(args.__dict__["webp_lossless"]),
    )

    wrote: list[Path] = []

    if bool(args.render_2d_grid):
        render_2d_grid(out_dir / "hilbert2d_grid.png", cfg)
        wrote.append(out_dir / "hilbert2d_grid.png")

    if bool(args.render_3d_still):
        render_3d(out_dir / "hilbert3d.png", cfg)
        wrote.append(out_dir / "hilbert3d.png")

    if bool(args.render_3d_grid):
        fmt = args.__dict__["3d_grid_format"]
        if fmt == "apng":
            anim_name = "hilbert3d_grid.png"
        else:
            anim_name = "hilbert3d_grid.webp"

        anim_path = out_dir / anim_name
        render_3d_grid_anim(
            anim_path,
            cfg,
            args.__dict__["3d_grid_nbits"],
            scene_nbits=int(args.__dict__["3d_grid_scene_nbits"]),
        )

        wrote.append(anim_path)

    print("wrote:")
    for p in wrote:
        print("-", p)


if __name__ == "__main__":
    main()
