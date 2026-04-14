import argparse
import re
import numpy as np
import matplotlib.pyplot as plt


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


def plot_all_scaled(labels, values_dict, out_path, cos_scale=500.0):
    metrics = [
        ("2-hop", values_dict["2hop"], False),
        ("Text", values_dict["text"], True),
        ("Vision", values_dict["vision"], True),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(metrics))
    width = 0.35
    colors = ["#4C78A8", "#F28E2B"]

    scale_factor = float(cos_scale)

    vals_a = []
    vals_b = []
    for _, vals, scale_cos in metrics:
        if scale_cos:
            vals_a.append(vals[0] * scale_factor)
            vals_b.append(vals[1] * scale_factor)
        else:
            vals_a.append(vals[0])
            vals_b.append(vals[1])

    ax.bar(x - width / 2, vals_a, width, label=labels[0], color=colors[0])
    ax.bar(x + width / 2, vals_b, width, label=labels[1], color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel(f"Average (2-hop raw, cos × {scale_factor:.0f})")
    ax.set_title("Avg Metrics (Top-K) with Cosine Scaling")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    for i, (_, vals, scale_cos) in enumerate(metrics):
        if scale_cos:
            a_txt = f"{vals[0]:.4f}"
            b_txt = f"{vals[1]:.4f}"
        else:
            a_txt = f"{vals[0]:.2f}"
            b_txt = f"{vals[1]:.2f}"
        ax.text(x[i] - width / 2, vals_a[i], a_txt, ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, vals_b[i], b_txt, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def plot_two_panel(labels, values_dict, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    width = 0.35
    colors = ["#4C78A8", "#F28E2B"]

    # 2-hop panel
    x0 = np.arange(1)
    axes[0].bar(x0 - width / 2, [values_dict["2hop"][0]], width, label=labels[0], color=colors[0])
    axes[0].bar(x0 + width / 2, [values_dict["2hop"][1]], width, label=labels[1], color=colors[1])
    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(["2-hop"])
    axes[0].set_ylabel("Average")
    axes[0].set_title("Avg 2-hop (raw)")
    axes[0].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    for i, v in enumerate(values_dict["2hop"]):
        axes[0].text(x0[0] + (-width / 2 if i == 0 else width / 2), v, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)

    # text/vision panel
    x1 = np.arange(2)
    axes[1].bar(x1 - width / 2, values_dict["text"], width, label=labels[0], color=colors[0])
    axes[1].bar(x1 + width / 2, values_dict["vision"], width, label=labels[1], color=colors[1])
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels(["Text", "Vision"])
    axes[1].set_ylabel("Average cosine")
    axes[1].set_title("Avg Text/Vision (raw)")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    # annotate text/vision
    for i, v in enumerate(values_dict["text"]):
        axes[1].text(x1[0] + (-width / 2 if i == 0 else width / 2), v, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(values_dict["vision"]):
        axes[1].text(x1[1] + (-width / 2 if i == 0 else width / 2), v, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=8)

    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignrec", default="topk_alignrec_item5016_k500.txt")
    ap.add_argument("--anchorrec", default="topk_anchorrec_item5016_k500.txt")
    ap.add_argument("--out_prefix", default="topk_avg_compare")
    args = ap.parse_args()

    a_twohop, a_vis, a_text = parse_avg(args.alignrec)
    b_twohop, b_vis, b_text = parse_avg(args.anchorrec)

    labels = ["AlignRec", "AnchorRec"]

    if a_twohop is None or b_twohop is None:
        print("[WARN] 2-hop avg missing; skipping 2-hop chart")
    if a_text is None or b_text is None:
        print("[WARN] text avg missing; skipping text chart")
    if a_vis is None or b_vis is None:
        print("[WARN] vision avg missing; skipping vision chart")

    values_dict = {
        "2hop": [a_twohop, b_twohop],
        "text": [a_text, b_text],
        "vision": [a_vis, b_vis],
    }
    if None not in values_dict["2hop"] + values_dict["text"] + values_dict["vision"]:
        plot_all_scaled(labels, values_dict, f"{args.out_prefix}_500_scaled500.png", cos_scale=500.0)
        plot_two_panel(labels, values_dict, f"{args.out_prefix}_500_split.png")


if __name__ == "__main__":
    main()
