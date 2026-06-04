from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOP_LABELS = ["Top1", "Top2", "Top3"]
MODELS = ["AnchorRec", "AlignRec"]
COLORS = {
    # "AnchorRec": "#2E4358",
    # "AlignRec": "#D95F02",
    # "BM3": "#5E8C61",
    "AnchorRec": "#2E4358",
    "AlignRec": "#6C99C7",
    "BM3": "#36699F",
}

ONE_HOP_COUNTS = {
    "AnchorRec": [12, 12, 18],
    "AlignRec": [16, 25, 23],
}
TWO_HOP_COUNTS = {
    "AnchorRec": [269, 285, 361],
    "AlignRec": [284, 376, 423],
}
AVG_TWO_HOP = {
    "AlignRec": 282.030827,
    "AnchorRec": 208.799953,
    "BM3": 407.979243,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 5 1-hop/2-hop bar charts.")
    parser.add_argument("--out-dir", type=Path, default=Path("RQ3"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def style_axis(
    ax,
    y_min: float,
    y_max: float,
    ylabel_fontsize: int = 15,
    tick_fontsize: int | None = None,
) -> None:
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("", fontsize=ylabel_fontsize, labelpad=8)
    ax.tick_params(axis="both", width=1.2, length=6)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.4)


def add_value_labels(ax, bars, fmt: str = "{:.0f}", fontsize: int = 9) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="black",
        )


def tight_count_limits(values: list[float]) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    if value_range <= 0:
        padding = max(1.0, max_value * 0.08)
        return max(0.0, min_value - padding), max_value + padding

    step = 5 if max_value <= 50 else 25
    y_min = max(0.0, np.floor((min_value - value_range * 0.12) / step) * step)
    y_max = np.ceil((max_value + value_range * 0.18) / step) * step
    return y_min, y_max


def plot_top123_grouped(
    data: dict[str, list[int]],
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float] = (5.8, 3.2),
    xtick_fontsize: int = 12,
    tick_fontsize: int | None = None,
    value_fontsize: int = 9,
    show_values: bool = True,
    show_legend: bool = True,
) -> None:
    x = np.arange(len(TOP_LABELS))
    width = 0.34
    offsets = [-width / 2, width / 2]

    all_values = [value for model in MODELS for value in data[model]]
    y_min, y_max = tight_count_limits(all_values)

    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    for idx, model in enumerate(MODELS):
        bars = ax.bar(
            x + offsets[idx],
            data[model],
            width=width,
            label=model,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        if show_values:
            add_value_labels(ax, bars, fontsize=value_fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(TOP_LABELS, fontsize=xtick_fontsize, fontweight="bold")
    style_axis(ax, y_min, y_max, tick_fontsize=tick_fontsize)
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#D0D0D0",
            fontsize=max(6, xtick_fontsize - 3),
            handlelength=1.6,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def save_model_legend(out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(1.35, 0.82), dpi=160)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[model], edgecolor="black", linewidth=0.8)
        for model in MODELS
    ]
    ax.legend(
        handles,
        MODELS,
        loc="center",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=7,
        handlelength=1.4,
        labelspacing=0.45,
        prop={"weight": "bold", "size": 7},
    )
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def plot_avg_two_hop(
    out_path: Path,
    dpi: int,
    models: list[str],
    height: float = 3.2,
) -> None:
    models = sorted(models, key=lambda model: AVG_TWO_HOP[model], reverse=True)
    values = [AVG_TWO_HOP[model] for model in models]
    x = np.arange(len(models))
    width = 0.56
    y_min = 150
    _, y_max = tight_count_limits(values)

    fig_width = 2.1 if len(models) <= 2 else 2.9
    fig, ax = plt.subplots(figsize=(fig_width, height), dpi=160)
    bars = ax.bar(
        x,
        values,
        width=width,
        color=[COLORS[model] for model in models],
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, fontweight="bold")
    for label in ax.get_xticklabels():
        if label.get_text() == "AnchorRec":
            label.set_bbox(
                {
                    "facecolor": "#FFF176",
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.16",
                    "alpha": 0.9,
                }
            )
    style_axis(ax, y_min, y_max, ylabel_fontsize=9, tick_fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_top123_grouped(
        ONE_HOP_COUNTS,
        args.out_dir / "figure_5_1_hop_bar.png",
        args.dpi,
    )
    plot_top123_grouped(
        TWO_HOP_COUNTS,
        args.out_dir / "figure_5_2_hop_bar.png",
        args.dpi,
    )
    plot_top123_grouped(
        TWO_HOP_COUNTS,
        args.out_dir / "figure_5_2_hop_bar_compact.png",
        args.dpi,
        figsize=(2.1, 3.2),
        xtick_fontsize=8,
        tick_fontsize=8,
        value_fontsize=7,
        show_values=False,
        show_legend=False,
    )
    save_model_legend(args.out_dir / "figure_5_2_hop_bar_compact_legend.png", args.dpi)
    plot_avg_two_hop(
        args.out_dir / "figure_5_avg_2_hop_bar.png",
        args.dpi,
        models=["AlignRec", "AnchorRec"],
    )
    plot_avg_two_hop(
        args.out_dir / "figure_5_avg_2_hop_bar_half_height.png",
        args.dpi,
        models=["AlignRec", "AnchorRec"],
        height=1.6,
    )
    plot_avg_two_hop(
        args.out_dir / "figure_5_avg_2_hop_bar_with_bm3.png",
        args.dpi,
        models=["AlignRec", "AnchorRec", "BM3"],
    )
    plot_avg_two_hop(
        args.out_dir / "figure_5_avg_2_hop_bar_with_bm3_half_height.png",
        args.dpi,
        models=["AlignRec", "AnchorRec", "BM3"],
        height=1.6,
    )


if __name__ == "__main__":
    main()
