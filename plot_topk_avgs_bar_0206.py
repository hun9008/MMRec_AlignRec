import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

tick_fontsize = 20
ytick_fontsize = 20
xtick_fontsize = 20

def parse_avg(path):
    avg_re = re.compile(r"^avg\t-\t-\t([0-9.]+|-)\t([0-9.]+|-)\t([0-9.]+|-)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = avg_re.match(line.strip())
            if m:
                twohop = None if m.group(1) == "-" else float(m.group(1))
                vision = None if m.group(2) == "-" else float(m.group(2))
                text = None if m.group(3) == "-" else float(m.group(3))
                return twohop, vision, text
    raise ValueError(f"avg line not found in {path}")

def _draw_left_panel(ax, labels, values_dict, width, colors, y_top):
    x0 = np.array([0])
    ax.bar(
        x0 - width / 2,
        [values_dict["2hop"][0]],
        width,
        label=labels[0],
        color=colors[0],
    )
    ax.bar(
        x0 + width / 2,
        [values_dict["2hop"][1]],
        width,
        label=labels[1],
        color=colors[1],
    )

    ax.set_xticks([0])
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xticklabels(["2-hop"], fontsize=xtick_fontsize)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylim(top=y_top)

    for i, v in enumerate(values_dict["2hop"]):
        ax.text(
            x0[0] + (-width / 2 if i == 0 else width / 2),
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
        )

def _draw_right_panel(ax, labels, values_dict, width, colors, y_top):
    x1 = np.arange(2)
    right_align = [values_dict["text"][0], values_dict["vision"][0]]
    right_anchor = [values_dict["text"][1], values_dict["vision"][1]]
    ax.bar(
        x1 - width / 2,
        right_align,
        width,
        label=labels[0],
        color=colors[0],
    )
    ax.bar(
        x1 + width / 2,
        right_anchor,
        width,
        label=labels[1],
        color=colors[1],
    )

    ax.set_xticks(x1)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xticklabels(["Text", "Vision"], fontsize=xtick_fontsize)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylim(top=y_top)

    for i, v in enumerate(right_align):
        ax.text(
            x1[i] - width / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    for i, v in enumerate(right_anchor):
        ax.text(
            x1[i] + width / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
        )

def plot_two_panel(labels, values_dict, out_path):
    fig = plt.figure(figsize=(8.6, 3.6))

    # 좌:1 / 우:2 비율로 영역 자체를 나눔
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.63)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1:])

    width = 0.35
    colors = ["#4C78A8", "#F28E2B"]
    left_y_top = max(values_dict["2hop"]) * 1.15
    right_y_top = max(values_dict["text"] + values_dict["vision"]) * 1.4

    _draw_left_panel(ax_left, labels, values_dict, width, colors, left_y_top)
    _draw_right_panel(ax_right, labels, values_dict, width, colors, right_y_top)

    handles, legend_labels = ax_right.get_legend_handles_labels()

    fig.tight_layout()
    fig.subplots_adjust(right=0.97)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

    # Save left/right panels separately while preserving current scale settings.
    left_out_path = f"{os.path.splitext(out_path)[0]}_left.png"
    right_out_path = f"{os.path.splitext(out_path)[0]}_right.png"

    fig_left, ax_left_only = plt.subplots(figsize=(8.6 / 3.0, 3.6))
    _draw_left_panel(ax_left_only, labels, values_dict, width, colors, left_y_top)
    fig_left.tight_layout()
    fig_left.savefig(left_out_path, dpi=300)
    plt.close(fig_left)
    print(f"[SAVE] {left_out_path}")

    fig_right, ax_right_only = plt.subplots(figsize=(8.6 * 2.0 / 3.0, 3.6))
    _draw_right_panel(ax_right_only, labels, values_dict, width, colors, right_y_top)
    fig_right.tight_layout()
    fig_right.savefig(right_out_path, dpi=300)
    plt.close(fig_right)
    print(f"[SAVE] {right_out_path}")

    legend_fig = plt.figure(figsize=(2.6, 1.0), facecolor="white")
    legend_fig.legend(
        handles,
        legend_labels,
        loc="center",
        frameon=False,
        fontsize=16,
        prop={"weight": "bold"},
        ncol=1,
    )
    legend_fig.tight_layout()
    legend_out_path = f"{os.path.splitext(out_path)[0]}_legend.png"
    legend_fig.savefig(legend_out_path, dpi=300, facecolor="white", transparent=False)
    plt.close(legend_fig)
    print(f"[SAVE] {legend_out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignrec", default="topk_alignrec_item5016_k500.txt")
    ap.add_argument("--anchorrec", default="topk_anchorrec_item5016_k500.txt")
    ap.add_argument("--out_prefix", default="topk_avg_compare")
    args = ap.parse_args()

    a_twohop, a_vis, a_text = parse_avg(args.alignrec)
    b_twohop, b_vis, b_text = parse_avg(args.anchorrec)

    labels = ["AlignRec", "AnchorRec"]

    values_dict = {
        "2hop": [a_twohop, b_twohop],
        "text": [a_text, b_text],
        "vision": [a_vis, b_vis],
    }

    if None not in values_dict["2hop"] + values_dict["text"] + values_dict["vision"]:
        plot_two_panel(labels, values_dict,
                       f"{args.out_prefix}_500_split.png")


if __name__ == "__main__":
    main()
