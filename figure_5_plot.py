from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


TEXT_COLS = ["text top1", "text top2", "text top3"]
VISION_COLS = ["vision top1", "vision top2", "vision top3"]
METRIC_LABELS = ["top1", "top2", "top3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one grouped bar chart per figure_5_avg_sim_AnchorRec_*.txt file."
    )
    parser.add_argument("--pattern", default="figure_5_avg_sim_AnchorRec_*.txt")
    parser.add_argument("--out-dir", default="figure_5_plots")
    parser.add_argument("--show-tick-labels", action="store_true")
    parser.add_argument("--show-y-ticks", action="store_true")
    return parser.parse_args()


def read_pipe_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s*\|\s*", engine="python")
    df.columns = [col.strip() for col in df.columns]

    required = ["model"] + TEXT_COLS + VISION_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    for col in TEXT_COLS + VISION_COLS:
        df[col] = pd.to_numeric(df[col])
    return df


def comparison_name_from_path(path: Path) -> str:
    match = re.match(r"figure_5_avg_sim_AnchorRec_(.+)\.txt$", path.name)
    if match:
        return match.group(1)
    return path.stem


def extract_rows(path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = read_pipe_table(path)
    anchor_rows = df[df["model"].astype(str).str.lower() == "anchorrec"]
    if anchor_rows.empty:
        raise ValueError(f"{path} has no AnchorRec row.")

    other_rows = df[df["model"].astype(str).str.lower() != "anchorrec"]
    if other_rows.empty:
        raise ValueError(f"{path} has no comparison row.")

    other = other_rows.iloc[0]
    anchor = anchor_rows.iloc[0]
    other_label = comparison_name_from_path(path)

    other_text = other[TEXT_COLS].to_numpy(dtype=float)
    anchor_text = anchor[TEXT_COLS].to_numpy(dtype=float)
    other_vision = other[VISION_COLS].to_numpy(dtype=float)
    anchor_vision = anchor[VISION_COLS].to_numpy(dtype=float)
    return other_label, other_text, anchor_text, other_vision, anchor_vision


def hide_ticks(ax: plt.Axes, show_tick_labels: bool, show_y_ticks: bool) -> None:
    if not show_y_ticks:
        ax.tick_params(axis="y", which="both", labelleft=False)
    if not show_tick_labels:
        ax.tick_params(axis="x", which="both", labelbottom=False)


def draw_metric_panel(
    ax: plt.Axes,
    labels: list[str],
    other_values: np.ndarray,
    anchor_values: np.ndarray,
    colors: list[str],
    show_tick_labels: bool,
    show_y_ticks: bool,
) -> None:
    x = np.arange(len(METRIC_LABELS))
    width = 0.35

    ax.bar(x - width / 2, other_values, width, label=labels[0], color=colors[0])
    ax.bar(x + width / 2, anchor_values, width, label=labels[1], color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS if show_tick_labels else [])
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    y_top = max(float(np.max(other_values)), float(np.max(anchor_values))) * 1.18
    if y_top <= 0:
        y_top = 1.0
    ax.set_ylim(0, y_top)
    hide_ticks(ax, show_tick_labels, show_y_ticks)


def save_legend(handles, labels: list[str], out_path: Path) -> None:
    fig = plt.figure(figsize=(2.8, 1.0), facecolor="white")
    fig.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=16,
        prop={"weight": "bold"},
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def plot_one_file(path: Path, out_dir: Path, show_tick_labels: bool, show_y_ticks: bool) -> None:
    other_label, other_text, anchor_text, other_vision, anchor_vision = extract_rows(path)
    labels = [other_label, "AnchorRec"]
    colors = ["#4C78A8", "#F28E2B"]

    stem = path.stem.replace("figure_5_avg_sim_", "figure_5_")
    combined_out = out_dir / f"{stem}.png"
    text_out = out_dir / f"{stem}_text.png"
    vision_out = out_dir / f"{stem}_vision.png"
    legend_out = out_dir / f"{stem}_legend.png"

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"wspace": 0.42})
    draw_metric_panel(axes[0], labels, other_text, anchor_text, colors, show_tick_labels, show_y_ticks)
    draw_metric_panel(axes[1], labels, other_vision, anchor_vision, colors, show_tick_labels, show_y_ticks)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.96, wspace=0.42)
    fig.savefig(combined_out, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {combined_out}")

    fig_text, ax_text = plt.subplots(figsize=(4.4, 3.6))
    draw_metric_panel(ax_text, labels, other_text, anchor_text, colors, show_tick_labels, show_y_ticks)
    fig_text.tight_layout()
    fig_text.savefig(text_out, dpi=300)
    plt.close(fig_text)
    print(f"[SAVE] {text_out}")

    fig_vision, ax_vision = plt.subplots(figsize=(4.4, 3.6))
    draw_metric_panel(ax_vision, labels, other_vision, anchor_vision, colors, show_tick_labels, show_y_ticks)
    fig_vision.tight_layout()
    fig_vision.savefig(vision_out, dpi=300)
    plt.close(fig_vision)
    print(f"[SAVE] {vision_out}")

    save_legend(handles, legend_labels, legend_out)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(".").glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.pattern}")

    for path in paths:
        plot_one_file(path, out_dir, args.show_tick_labels, args.show_y_ticks)


if __name__ == "__main__":
    main()
