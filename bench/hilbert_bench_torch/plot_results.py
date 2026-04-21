# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
# ]
# ///

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager

DISPLAY_NAMES = {
    "skilling_eager": "Skilling: eager",
    "skilling_triton_2d": "Skilling (ours): triton",
    "skilling_triton_3d": "Skilling (ours): triton",
    "hilbertsfc_torch_eager": "HilbertSFC: eager",
    "hilbertsfc_torch_compile": "HilbertSFC: compile",
    "hilbertsfc_triton": "HilbertSFC: triton",
}

TRITON_TUNING_LABELS = {
    "heuristic": "heuristic",
    "autotune_bucketed": "autotune bucketed",
    "autotune_exact": "autotune exact",
}


STYLE_MAP = {
    # Skilling family (cool)
    "skilling_eager": {"color": "#4C78A8", "linestyle": "-", "marker": "o"},
    "skilling_triton_2d": {"color": "#2F5D8A", "linestyle": "-", "marker": "o"},
    "skilling_triton_3d": {"color": "#1F4E79", "linestyle": "-", "marker": "o"},
    # HilbertSFC family (warm)
    "hilbertsfc_torch_eager": {"color": "#F28E2B", "linestyle": "-", "marker": "o"},
    "hilbertsfc_torch_compile": {"color": "#A16340", "linestyle": "--", "marker": "o"},
    "hilbertsfc_triton": {"color": "#E15759", "linestyle": "-", "marker": "o"},
}

BAR_IMPL_COLORS = [
    "#FF6F61",
    "#4C72B0",
    "#F0C05A",
    "#009473",
    "#D94F70",
    "#7BC4C4",
]

BAR_IMPL_COLOR_BY_PROVIDER = {
    "skilling_eager": BAR_IMPL_COLORS[0],
    "hilbertsfc_torch_eager": BAR_IMPL_COLORS[1],
    "skilling_triton_2d": BAR_IMPL_COLORS[2],
    "skilling_triton_3d": BAR_IMPL_COLORS[2],
    "hilbertsfc_torch_compile": BAR_IMPL_COLORS[3],
    "hilbertsfc_triton": BAR_IMPL_COLORS[4],
}

LAYOUT_SPECS: dict[str, list[tuple[str, int]]] = {
    "2x2": [("encode", 2), ("decode", 2), ("encode", 3), ("decode", 3)],
    "2x1-2d": [("encode", 2), ("decode", 2)],
    "2x1-3d": [("encode", 3), ("decode", 3)],
}


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _size_label(size: int) -> str:
    if size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)}Mi"
    if size % 1024 == 0:
        return f"{size // 1024}Ki"
    return str(size)


def _parse_size_token(token: str) -> int:
    t = token.strip().lower().replace("_", "")
    if not t:
        raise ValueError("Empty size token")

    multipliers = {
        "ki": 1024,
        "mi": 1024 * 1024,
        "gi": 1024 * 1024 * 1024,
        "k": 1000,
        "m": 1000 * 1000,
        "g": 1000 * 1000 * 1000,
    }
    for suffix, mult in multipliers.items():
        if t.endswith(suffix):
            value = t[: -len(suffix)]
            if not value.isdigit():
                raise ValueError(f"Invalid size token: {token}")
            return int(value) * mult

    if not t.isdigit():
        raise ValueError(f"Invalid size token: {token}")
    return int(t)


def _parse_bar_sizes(value: str) -> list[int]:
    sizes = [_parse_size_token(tok) for tok in value.split(",") if tok.strip()]
    if not sizes:
        raise ValueError("No sizes provided")
    return sizes


def _display_name(provider: str) -> str:
    return DISPLAY_NAMES.get(provider, provider)


def _series_key(row: dict[str, str]) -> str:
    provider = row.get("provider", "")
    tuning = row.get("triton_tuning", "").strip()
    if provider == "hilbertsfc_triton" and tuning:
        return f"{provider}:{tuning}"
    return provider


def _provider_from_series(series_key: str) -> str:
    return series_key.split(":", 1)[0]


def _tuning_from_series(series_key: str) -> str:
    parts = series_key.split(":", 1)
    return parts[1] if len(parts) == 2 else ""


def _series_display_name(series_key: str) -> str:
    provider = _provider_from_series(series_key)
    tuning = _tuning_from_series(series_key)
    base = _display_name(provider)
    if provider == "hilbertsfc_triton" and tuning:
        tuning_label = TRITON_TUNING_LABELS.get(tuning, tuning)
        return f"{base} ({tuning_label})"
    return base


def _style(provider: str) -> dict[str, str]:
    return STYLE_MAP.get(
        provider, {"color": "#666666", "linestyle": "-", "marker": "o"}
    )


def _style_for_series(series_key: str) -> dict[str, str]:
    provider = _provider_from_series(series_key)
    tuning = _tuning_from_series(series_key)
    sty = dict(_style(provider))
    if provider == "hilbertsfc_triton":
        if tuning == "heuristic":
            sty["linestyle"] = "-"
        elif tuning == "autotune_bucketed":
            sty["linestyle"] = "--"
        elif tuning == "autotune_exact":
            sty["linestyle"] = ":"
    return sty


def _bar_color(provider: str, fallback_index: int) -> str:
    return BAR_IMPL_COLOR_BY_PROVIDER.get(
        provider, BAR_IMPL_COLORS[fallback_index % len(BAR_IMPL_COLORS)]
    )


def _is_eager_provider(provider: str) -> bool:
    return provider.endswith("_eager")


def _panel_title(
    op: str,
    dim: int,
    *,
    title_prefix: str = "",
    title_postfix: str = "",
) -> str:
    base = f"{op.title()} {dim}D"
    prefix = title_prefix.strip()
    postfix = title_postfix.strip()

    if prefix and postfix:
        return f"{prefix} - {base} - {postfix}"
    if prefix:
        return f"{prefix} - {base}"
    if postfix:
        return f"{base} - {postfix}"
    return base


def _set_plot_theme() -> None:
    preferred_fonts = [
        "Selawik",
        "IBM Plex Sans",
        "Segoe UI",
        "DejaVu Sans",
        "Arial",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = next(
        (name for name in preferred_fonts if name in installed), "DejaVu Sans"
    )

    plt.rcParams.update(
        {
            "font.family": selected_font,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "axes.labelweight": "bold",
            "axes.axisbelow": True,
            "axes.labelcolor": "#4D4D4D",
            "axes.edgecolor": "#4D4D4D",
            "xtick.color": "#4D4D4D",
            "ytick.color": "#4D4D4D",
            "text.color": "#4D4D4D",
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfc",
            "lines.linewidth": 1.6,
            "grid.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _select_rate_unit(max_mpts: float) -> tuple[str, float]:
    # Input is in Mpts/s; select a readable display scale for the left axis.
    if max_mpts >= 1_000_000.0:
        return "Tpts/s", 1_000_000.0
    if max_mpts >= 1_000.0:
        return "Gpts/s", 1_000.0
    return "Mpts/s", 1.0


def _ordered_providers(subset: list[dict[str, str]]) -> list[str]:
    series_keys: list[str] = []
    for r in subset:
        key = _series_key(r)
        if key not in series_keys:
            series_keys.append(key)

    preferred = [
        "skilling_eager",
        "skilling_triton_2d",
        "skilling_triton_3d",
        "hilbertsfc_torch_eager",
        "hilbertsfc_torch_compile",
        "hilbertsfc_triton",
    ]

    tuning_order = ["heuristic", "autotune_bucketed", "autotune_exact"]

    ordered: list[str] = []
    for provider in preferred:
        matching = [k for k in series_keys if _provider_from_series(k) == provider]
        for tuning in tuning_order:
            tuned = f"{provider}:{tuning}"
            if tuned in matching:
                ordered.append(tuned)
        if provider in matching:
            ordered.append(provider)
        for key in matching:
            if key not in ordered:
                ordered.append(key)

    extras = [k for k in series_keys if k not in ordered]
    return ordered + extras


def _ordered_bar_providers(subset: list[dict[str, str]], *, dim: int) -> list[str]:
    series_keys = _ordered_providers(subset)
    skilling_triton = "skilling_triton_2d" if dim == 2 else "skilling_triton_3d"

    preferred = [
        "skilling_eager",
        "hilbertsfc_torch_eager",
        skilling_triton,
        "hilbertsfc_torch_compile",
        "hilbertsfc_triton",
    ]

    ordered: list[str] = []
    for provider in preferred:
        ordered.extend([k for k in series_keys if _provider_from_series(k) == provider])
    extras = [k for k in series_keys if k not in ordered]
    return ordered + extras


def _draw_line_panel(
    ax_left,
    *,
    rows: list[dict[str, str]],
    op: str,
    dim: int,
    title_prefix: str,
    title_postfix: str,
    show_legend: bool,
) -> bool:
    subset = [
        r
        for r in rows
        if r.get("available", "").lower() == "true"
        and r.get("op") == op
        and int(r.get("dim", 0)) == dim
    ]
    if not subset:
        return False

    provider_order = _ordered_providers(subset)
    sizes_all = sorted({int(r["size"]) for r in subset})

    by_provider_size: dict[tuple[str, int], dict[str, str]] = {}
    for r in subset:
        by_provider_size[(_series_key(r), int(r["size"]))] = r

    valid_mpts = [
        float(r["mpts_p50"])
        for r in subset
        if float(r["mpts_p50"]) > 0.0 and float(r["gbps_p50"]) >= 0.0
    ]
    max_mpts = max(valid_mpts) if valid_mpts else 0.0
    rate_unit, rate_scale = _select_rate_unit(max_mpts)

    ratios = [
        float(r["gbps_p50"]) / float(r["mpts_p50"])
        for r in subset
        if float(r["mpts_p50"]) > 0.0 and float(r["gbps_p50"]) >= 0.0
    ]
    gbps_per_mpts = (sum(ratios) / len(ratios)) if ratios else 0.0

    x_positions = list(range(len(sizes_all)))

    for i, provider in enumerate(provider_order):
        sty = _style_for_series(provider)
        yvals: list[float] = []
        for size in sizes_all:
            row = by_provider_size.get((provider, int(size)))
            if row is None:
                yvals.append(float("nan"))
            else:
                yvals.append(float(row["mpts_p50"]) / rate_scale)
        ax_left.plot(
            x_positions,
            yvals,
            marker=sty["marker"],
            linewidth=2.0,
            markersize=4.0,
            linestyle=sty["linestyle"],
            color=sty["color"],
            label=_series_display_name(provider),
            zorder=3,
        )

    if gbps_per_mpts > 0.0:
        right_scale = gbps_per_mpts * rate_scale

        def _to_gbps(y_left: float) -> float:
            return y_left * right_scale

        def _to_left(y_right: float) -> float:
            return y_right / right_scale

        ax_right = ax_left.secondary_yaxis("right", functions=(_to_gbps, _to_left))
        ax_right.set_ylabel("GB/s")

    ax_left.set_xticks(x_positions)

    # Keep points equally spaced while avoiding unreadable tick label overlap.
    max_labels = 14
    step = max(1, len(sizes_all) // max_labels)
    labels = [
        (_size_label(s) if (idx % step == 0 or idx == len(sizes_all) - 1) else "")
        for idx, s in enumerate(sizes_all)
    ]
    ax_left.set_xticklabels(labels, rotation=25, ha="right")

    ax_left.set_xlabel("Number of elements")
    ax_left.set_ylabel(rate_unit)
    ax_left.set_title(
        _panel_title(
            op,
            dim,
            title_prefix=title_prefix,
            title_postfix=title_postfix,
        )
    )
    ax_left.grid(True, axis="both", linestyle=":", alpha=0.35)

    if show_legend:
        h1, l1 = ax_left.get_legend_handles_labels()
        ax_left.legend(
            h1,
            l1,
            ncol=1,
            fontsize=8,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
        )

    return True


def _draw_bar_panel(
    ax_left,
    *,
    rows: list[dict[str, str]],
    op: str,
    dim: int,
    title_prefix: str,
    title_postfix: str,
    sizes: list[int],
    show_legend: bool,
) -> bool:
    subset = [
        r
        for r in rows
        if r.get("available", "").lower() == "true"
        and r.get("op") == op
        and int(r.get("dim", 0)) == dim
    ]
    if not subset:
        return False

    provider_order = _ordered_bar_providers(subset, dim=dim)
    sizes_available = {int(r["size"]) for r in subset}
    selected_sizes = [s for s in sizes if s in sizes_available]
    if not selected_sizes:
        selected_sizes = sorted(sizes_available)[: min(4, len(sizes_available))]

    by_provider_size: dict[tuple[str, int], dict[str, str]] = {}
    for r in subset:
        by_provider_size[(_series_key(r), int(r["size"]))] = r

    ratios = [
        float(r["gbps_p50"]) / (float(r["mpts_p50"]) / 1000.0)
        for r in subset
        if float(r["mpts_p50"]) > 0.0 and float(r["gbps_p50"]) >= 0.0
    ]
    gbps_per_gpts = (sum(ratios) / len(ratios)) if ratios else 0.0

    x_positions = list(range(len(selected_sizes)))
    n_providers = len(provider_order)
    bar_width = min(0.8 / max(1, n_providers), 0.22)

    max_y = 0.0
    for i_provider, provider in enumerate(provider_order):
        offset = (i_provider - (n_providers - 1) / 2.0) * bar_width
        yvals: list[float] = []
        for size in selected_sizes:
            row = by_provider_size.get((provider, size))
            if row is None:
                yvals.append(float("nan"))
            else:
                yvals.append(float(row["mpts_p50"]) / 1000.0)

        bars = ax_left.bar(
            [x + offset for x in x_positions],
            yvals,
            width=bar_width * 0.95,
            color=_bar_color(_provider_from_series(provider), i_provider),
            label=_series_display_name(provider),
            zorder=3,
        )

        for bar, y in zip(bars, yvals):
            if y == y and y > 0.0:  # NaN check via self-equality
                max_y = max(max_y, y)

    text_pad = max(0.02 * max_y, 0.01)
    for i_provider, provider in enumerate(provider_order):
        if not _is_eager_provider(provider):
            continue

        offset = (i_provider - (n_providers - 1) / 2.0) * bar_width
        for x, size in zip(x_positions, selected_sizes):
            row = by_provider_size.get((provider, size))
            if row is None:
                ax_left.text(
                    x + offset,
                    text_pad,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#444444",
                    rotation=90,
                )
                continue

            y = float(row["mpts_p50"]) / 1000.0
            if y > 0.0:
                ax_left.text(
                    x + offset,
                    y + text_pad,
                    f"{y:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                    rotation=90,
                )

    if gbps_per_gpts > 0.0:

        def _to_gbps(y_left: float) -> float:
            return y_left * gbps_per_gpts

        def _to_left(y_right: float) -> float:
            return y_right / gbps_per_gpts

        ax_right = ax_left.secondary_yaxis("right", functions=(_to_gbps, _to_left))
        ax_right.set_ylabel("GB/s")

    ax_left.set_xticks(x_positions)
    ax_left.set_xticklabels(
        [_size_label(s) for s in selected_sizes],
        rotation=20,
        ha="right",
    )
    ax_left.set_xlabel("Number of elements")
    ax_left.set_ylabel("Gpts/s")
    ax_left.set_title(
        _panel_title(
            op,
            dim,
            title_prefix=title_prefix,
            title_postfix=title_postfix,
        )
    )
    ax_left.grid(True, axis="y", linestyle=":", alpha=0.35)

    if show_legend:
        h1, l1 = ax_left.get_legend_handles_labels()
        ax_left.legend(
            h1,
            l1,
            ncol=1,
            fontsize=8,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            title="Implementation",
            title_fontsize=8,
        )

    return True


def _layout_shape(layout: str) -> tuple[int, int, tuple[float, float]]:
    if layout == "2x2":
        return 2, 2, (11, 7)
    return 1, 2, (10, 3.8)


def _single_panel_figsize() -> tuple[float, float]:
    # Match the visual scale of one panel in the side-by-side layout.
    _, _, pair_size = _layout_shape("2x1-3d")
    return pair_size[0] / 2.0, pair_size[1]


def _plot_line_layout(
    rows: list[dict[str, str]],
    *,
    layout: str,
    title_prefix: str,
    title_postfix: str,
    out_path: Path,
) -> None:
    specs = LAYOUT_SPECS[layout]
    nrows, ncols, figsize = _layout_shape(layout)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_flat: list[Any] = list(axes.flat) if hasattr(axes, "flat") else [axes]

    first_legend_axis = None
    for ax, (op, dim) in zip(axes_flat, specs):
        ok = _draw_line_panel(
            ax,
            rows=rows,
            op=op,
            dim=dim,
            title_prefix=title_prefix,
            title_postfix=title_postfix,
            show_legend=False,
        )
        if not ok:
            ax.set_title(
                _panel_title(
                    op,
                    dim,
                    title_prefix=title_prefix,
                    title_postfix=title_postfix,
                )
                + " (no data)"
            )
        elif first_legend_axis is None:
            first_legend_axis = ax

    legend_axis: Any = (
        first_legend_axis if first_legend_axis is not None else axes_flat[0]
    )
    handles, labels = legend_axis.get_legend_handles_labels()
    if handles and labels:
        legend_axis.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            frameon=True,
            borderaxespad=0.0,
        )

    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def _plot_bar_layout(
    rows: list[dict[str, str]],
    *,
    layout: str,
    title_prefix: str,
    title_postfix: str,
    sizes: list[int],
    out_path: Path,
) -> None:
    specs = LAYOUT_SPECS[layout]
    nrows, ncols, figsize = _layout_shape(layout)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_flat: list[Any] = list(axes.flat) if hasattr(axes, "flat") else [axes]

    first_legend_axis = None
    for ax, (op, dim) in zip(axes_flat, specs):
        ok = _draw_bar_panel(
            ax,
            rows=rows,
            op=op,
            dim=dim,
            title_prefix=title_prefix,
            title_postfix=title_postfix,
            sizes=sizes,
            show_legend=False,
        )
        if not ok:
            ax.set_title(
                _panel_title(
                    op,
                    dim,
                    title_prefix=title_prefix,
                    title_postfix=title_postfix,
                )
                + " (no data)"
            )
        elif first_legend_axis is None:
            first_legend_axis = ax

    legend_axis: Any = (
        first_legend_axis if first_legend_axis is not None else axes_flat[0]
    )
    handles, labels = legend_axis.get_legend_handles_labels()
    if handles and labels:
        legend_axis.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            frameon=True,
            borderaxespad=0.0,
            title="Implementation",
            title_fontsize=8,
        )

    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def _plot_line_one(
    rows: list[dict[str, str]],
    *,
    op: str,
    dim: int,
    title_prefix: str,
    title_postfix: str,
    out_path: Path,
) -> None:

    fig, ax_left = plt.subplots(figsize=_single_panel_figsize())
    ok = _draw_line_panel(
        ax_left,
        rows=rows,
        op=op,
        dim=dim,
        title_prefix=title_prefix,
        title_postfix=title_postfix,
        show_legend=True,
    )
    if not ok:
        plt.close(fig)
        print(f"[warn] no rows for op={op} dim={dim}; skipping {out_path.name}")
        return

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def _plot_line_combined(
    rows: list[dict[str, str]], *, title_prefix: str, title_postfix: str, out_path: Path
) -> None:
    _plot_line_layout(
        rows,
        layout="2x2",
        title_prefix=title_prefix,
        title_postfix=title_postfix,
        out_path=out_path,
    )


def main() -> int:
    _set_plot_theme()

    p = argparse.ArgumentParser(
        description=(
            "Create line/bar benchmark charts from unified bench_results.csv. "
            "Supports 2x2 and 2x1 layouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="bench/hilbert_bench_torch/results/unified_run",
        help="Directory containing bench_results.csv",
    )
    p.add_argument(
        "--plot-mode",
        type=str,
        choices=["line", "bar", "both"],
        default="line",
        help="Select line plots, bar plots, or both.",
    )
    p.add_argument(
        "--layout",
        type=str,
        choices=sorted(LAYOUT_SPECS.keys()),
        default="2x2",
        help="Panel layout for combined line/bar figures.",
    )
    p.add_argument(
        "--bar-sizes",
        type=str,
        default="128Ki,1Mi,8Mi,32Mi",
        help="Comma-separated sizes for bar mode (e.g. 128Ki,1Mi,8Mi,32Mi).",
    )
    p.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help=(
            "Optional prefix prepended to each panel title, e.g. 'MI300X - Encode 3D'."
        ),
    )
    p.add_argument(
        "--title-postfix",
        type=str,
        default="",
        help=(
            "Optional postfix appended to each panel title, e.g. "
            "'Encode 3D - nbits=32'."
        ),
    )
    # Backward-compatible aliases for older scripts.
    p.add_argument(
        "--subplot-title-prefix",
        dest="title_prefix",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--subplot-title-postfix",
        dest="title_postfix",
        help=argparse.SUPPRESS,
    )

    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "bench_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    rows = _load_rows(csv_path)
    selected_bar_sizes = _parse_bar_sizes(args.bar_sizes)
    title_prefix = args.title_prefix
    title_postfix = args.title_postfix

    if args.plot_mode in {"line", "both"}:
        plots = [
            ("encode", 2),
            ("encode", 3),
            ("decode", 2),
            ("decode", 3),
        ]
        for op, dim in plots:
            out_path = results_dir / f"lines_{op}_{dim}d.png"
            _plot_line_one(
                rows,
                op=op,
                dim=dim,
                title_prefix=title_prefix,
                title_postfix=title_postfix,
                out_path=out_path,
            )

        _plot_line_layout(
            rows,
            layout=args.layout,
            title_prefix=title_prefix,
            title_postfix=title_postfix,
            out_path=results_dir / f"lines_{args.layout}.png",
        )
        if args.layout == "2x2":
            _plot_line_combined(
                rows,
                title_prefix=title_prefix,
                title_postfix=title_postfix,
                out_path=results_dir / "lines_2x2_all.png",
            )

    if args.plot_mode in {"bar", "both"}:
        _plot_bar_layout(
            rows,
            layout=args.layout,
            title_prefix=title_prefix,
            title_postfix=title_postfix,
            sizes=selected_bar_sizes,
            out_path=results_dir / f"bars_{args.layout}.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
