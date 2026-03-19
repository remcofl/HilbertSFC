# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hilbertsfc>=0.2.0",
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.3",
#     "pillow>=12.1.1",
# ]
# ///
"""Render MkDocs logo + favicon assets.

Creates:
- docs/img/logo.png    (square, transparent)
- docs/img/favicon.png (32x32, transparent)

Design: a static 2D Hilbert curve.

Run:
    uv run scripts/render_docs_branding.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from PIL import Image

from hilbertsfc import hilbert_decode_2d


@dataclass(frozen=True)
class BrandingConfig:
    nbits: int = 2

    # Output sizes
    logo_px: int = 512
    favicon_px: int = 32
    favicon_margin_px: int = 2

    # Styling
    cmap: str = "turbo"
    # If set, render as a single solid stroke color (e.g. "white", "#fff").
    # When None, the curve uses a colormap gradient.
    stroke_color: str | None = "white"
    alpha: float = 1.0
    line_width_pt: float = 18.0

    # Rendering quality
    dpi: int = 256
    subdiv_per_segment: int = 24

    # Post-processing
    trim_pad_px: int = 2


def _segments_with_gradient(
    points: np.ndarray, *, subdiv_per_segment: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (segments, t) for a polyline, optionally subdivided.

    For `nbits=2` the polyline is short, so this stays fast and keeps the
    colormap gradient smooth even with thick strokes.
    """

    if points.shape[0] < 2:
        raise ValueError("need at least 2 points")

    base_segments = np.stack([points[:-1], points[1:]], axis=1)  # (S, 2, 2)
    if subdiv_per_segment <= 1:
        t = np.linspace(0.0, 1.0, base_segments.shape[0], dtype=np.float32)
        return base_segments, t

    s = base_segments.shape[0]
    k = int(subdiv_per_segment)
    u0 = np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)
    u1 = np.linspace(1.0 / k, 1.0, k, endpoint=True, dtype=np.float32)

    start = base_segments[:, 0, :][:, None, :]
    end = base_segments[:, 1, :][:, None, :]
    a = start * (1.0 - u0[None, :, None]) + end * (u0[None, :, None])
    b = start * (1.0 - u1[None, :, None]) + end * (u1[None, :, None])

    segments = np.stack([a, b], axis=2).reshape(s * k, 2, 2)
    t = np.linspace(0.0, 1.0, segments.shape[0], dtype=np.float32)
    return segments, t


def _trim_by_alpha(img: Image.Image, *, pad_px: int) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.getchannel("A").getbbox()
    if bbox is None:
        return img

    x0, y0, x1, y1 = bbox
    pad = max(0, int(pad_px))
    return img.crop(
        (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(img.width, x1 + pad),
            min(img.height, y1 + pad),
        )
    )


def _fit_to_square(img: Image.Image, *, out_px: int) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    side = max(img.width, img.height)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.alpha_composite(
        img, dest=((side - img.width) // 2, (side - img.height) // 2)
    )
    return canvas.resize((out_px, out_px), resample=Image.Resampling.LANCZOS)


def _downscale_with_margin(
    img: Image.Image, *, out_px: int, margin_px: int
) -> Image.Image:
    """Downscale to a square output with a transparent margin."""

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    m = max(0, int(margin_px))
    inner = max(1, int(out_px) - 2 * m)
    scaled = img.resize((inner, inner), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (out_px, out_px), (0, 0, 0, 0))
    canvas.alpha_composite(scaled, dest=(m, m))
    return canvas


def _compute_safe_data_pad(*, cfg: BrandingConfig, out_px: int) -> float:
    """Compute data padding so thick strokes don't get clipped.

    Matplotlib autoscaling doesn't account for linewidth, so we expand limits.
    Padding is derived from linewidth in pixels and the data span.
    """

    side = float(2**cfg.nbits - 1)  # nbits=2 -> side=3
    lw_px = float(cfg.line_width_pt) * float(cfg.dpi) / 72.0
    # First estimate (no pad), then one refinement that accounts for pad.
    pad = 0.5 * lw_px * (side / float(out_px))
    pad = 0.5 * lw_px * ((side + 2.0 * pad) / float(out_px))
    # Small safety factor for antialiasing + round caps.
    return float(pad * 1.25 + 0.05)


def _render_curve_rgba(cfg: BrandingConfig, *, out_px: int) -> Image.Image:
    n = 2 ** (2 * cfg.nbits)
    indices = np.arange(n, dtype=np.uint32)
    xs, ys = hilbert_decode_2d(indices, nbits=cfg.nbits)

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    segs, t = _segments_with_gradient(pts, subdiv_per_segment=cfg.subdiv_per_segment)

    fig_inches = out_px / float(cfg.dpi)
    fig = plt.figure(figsize=(fig_inches, fig_inches), dpi=cfg.dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    lc_kwargs: dict[str, Any] = {
        "linewidths": cfg.line_width_pt,
        "alpha": cfg.alpha,
        "antialiased": True,
        "capstyle": "round",
        "joinstyle": "round",
    }

    if cfg.stroke_color is None:
        lc_kwargs["cmap"] = cfg.cmap
        lc = LineCollection(cast(Any, segs), **lc_kwargs)
        lc.set_array(t)
    else:
        # Solid stroke. Include alpha in the color itself to keep behavior
        # consistent when Matplotlib handles blending.
        r, g, b, _ = to_rgba(cfg.stroke_color)
        lc = LineCollection(
            cast(Any, segs),
            colors=[(r, g, b, cfg.alpha)],
            **lc_kwargs,
        )
    ax.add_collection(lc)

    side = float(2**cfg.nbits - 1)
    pad = _compute_safe_data_pad(cfg=cfg, out_px=out_px)
    ax.set_xlim(-pad, side + pad)
    ax.set_ylim(-pad, side + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    canvas = cast(Any, fig.canvas)
    canvas.draw()
    w, h = canvas.get_width_height()
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return Image.fromarray(rgba, mode="RGBA")


def main() -> None:
    cfg = BrandingConfig()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render big once for best downscaling quality.
    logo_raw = _render_curve_rgba(cfg, out_px=cfg.logo_px)
    logo = _fit_to_square(
        _trim_by_alpha(logo_raw, pad_px=cfg.trim_pad_px), out_px=cfg.logo_px
    )
    logo_path = out_dir / "logo.png"
    logo.save(logo_path, format="PNG", optimize=True)

    favicon = _downscale_with_margin(
        logo,
        out_px=cfg.favicon_px,
        margin_px=cfg.favicon_margin_px,
    )
    favicon_path = out_dir / "favicon.png"
    favicon.save(favicon_path, format="PNG", optimize=True)

    print(f"Wrote {logo_path.relative_to(repo_root)}")
    print(f"Wrote {favicon_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
