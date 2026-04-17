# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
# ]
# ///

import argparse
import csv
from pathlib import Path

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

MAX_Y_MPTS = 88_000.0


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _size_label(size: int) -> str:
    if size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)}Mi"
    if size % 1024 == 0:
        return f"{size // 1024}Ki"
    return str(size)


def _display_name(provider: str) -> str:
    return DISPLAY_NAMES.get(provider, provider)


def _style(provider: str) -> dict[str, str]:
    return STYLE_MAP.get(
        provider, {"color": "#666666", "linestyle": "-", "marker": "o"}
    )


def _panel_title(op: str, dim: int) -> str:
    return f"{op.title()} {dim}D"


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
    providers: list[str] = []
    for r in subset:
        p = r["provider"]
        if p not in providers:
            providers.append(p)

    preferred = [
        "skilling_eager",
        "skilling_triton_2d",
        "skilling_triton_3d",
        "hilbertsfc_torch_eager",
        "hilbertsfc_torch_compile",
        "hilbertsfc_triton",
    ]

    ordered = [p for p in preferred if p in providers]
    extras = [p for p in providers if p not in ordered]
    return ordered + extras


def _draw_line_panel(
    ax_left,
    *,
    rows: list[dict[str, str]],
    op: str,
    dim: int,
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
        by_provider_size[(r["provider"], int(r["size"]))] = r

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
        sty = _style(provider)
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
            label=_display_name(provider),
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
    ax_left.set_title(_panel_title(op, dim))
    ax_left.set_ylim(-2000 / rate_scale, MAX_Y_MPTS / rate_scale)
    ax_left.grid(True, axis="both", linestyle=":", alpha=0.35)

    if show_legend:
        h1, l1 = ax_left.get_legend_handles_labels()
        ax_left.legend(
            h1,
            l1,
            ncol=1,
            fontsize=8,
            frameon=True,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
        )

    return True


def _plot_line_one(
    rows: list[dict[str, str]],
    *,
    op: str,
    dim: int,
    out_path: Path,
) -> None:

    fig, ax_left = plt.subplots(figsize=(13, 6.8))
    ok = _draw_line_panel(
        ax_left,
        rows=rows,
        op=op,
        dim=dim,
        show_legend=True,
    )
    if not ok:
        plt.close(fig)
        print(f"[warn] no rows for op={op} dim={dim}; skipping {out_path.name}")
        return

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def _plot_line_combined(rows: list[dict[str, str]], *, out_path: Path) -> None:

    specs = [
        ("encode", 2),
        ("decode", 2),
        ("encode", 3),
        ("decode", 3),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    for ax, (op, dim) in zip(axes.flat, specs):
        ok = _draw_line_panel(
            ax,
            rows=rows,
            op=op,
            dim=dim,
            show_legend=False,
        )
        if not ok:
            ax.set_title(f"{op.title()} {dim}D (no data)")

    # fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.94, wspace=0.22, hspace=0.30)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles and labels:
        axes.flat[0].legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            frameon=True,
            borderaxespad=0.0,
        )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def _plot_line_3d_pair(rows: list[dict[str, str]], *, out_path: Path) -> None:
    specs = [("encode", 3), ("decode", 3)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    for ax, (op, dim) in zip(axes.flat, specs):
        ok = _draw_line_panel(
            ax,
            rows=rows,
            op=op,
            dim=dim,
            show_legend=False,
        )
        if not ok:
            ax.set_title(f"{op.title()} {dim}D (no data)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles and labels:
        axes.flat[0].legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            frameon=True,
            borderaxespad=0.0,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def main() -> int:
    _set_plot_theme()

    p = argparse.ArgumentParser(
        description=(
            "Create provider line charts from unified bench_results.csv. "
            "Generates 4 line plots and one combined 2x2 line figure."
        )
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="bench/hilbert_bench_torch/results/unified_run",
        help="Directory containing bench_results.csv",
    )

    args = p.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "bench_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    rows = _load_rows(csv_path)

    plots = [
        ("encode", 2),
        ("encode", 3),
        ("decode", 2),
        ("decode", 3),
    ]
    for op, dim in plots:
        out_path = results_dir / f"lines_{op}_{dim}d.png"
        _plot_line_one(rows, op=op, dim=dim, out_path=out_path)

    _plot_line_combined(rows, out_path=results_dir / "lines_2x2_all.png")

    repo_root = Path(__file__).resolve().parents[2]
    _plot_line_3d_pair(
        rows, out_path=repo_root / "docs" / "img" / "torch_cuda_3d_encode_decode.png"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
