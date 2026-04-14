import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_topk(path: str, topk: int = 3) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    # Skip header and parse data rows with flexible whitespace.
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue

        rank = int(parts[0])
        item = int(parts[1])
        twohop = float(parts[2])
        vision = float(parts[3])
        text = float(parts[4])

        rows.append(
            {
                "rank": rank,
                "item": item,
                "2hop": twohop,
                "vision_cos": vision,
                "text_cos": text,
            }
        )

    rows.sort(key=lambda r: r["rank"])
    if len(rows) < topk:
        raise ValueError(f"{path} has only {len(rows)} rows; need at least top-{topk}.")
    return rows[:topk]


def _plot_metric(
    align_rows: List[Dict[str, float]],
    anchor_rows: List[Dict[str, float]],
    metric: str,
    out_path: str,
) -> None:
    x = [0, 1, 2]
    width = 0.36

    align_vals = [row[metric] for row in align_rows]
    anchor_vals = [row[metric] for row in anchor_rows]

    left_positions = [v - width / 2 for v in x]
    right_positions = [v + width / 2 for v in x]

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
    ax.bar(left_positions, align_vals, width=width, color="#4C78A8")
    ax.bar(right_positions, anchor_vals, width=width, color="#F28E2B")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    y_top = max(align_vals + anchor_vals) * 1.2
    if y_top <= 0:
        y_top = 1.0
    ax.set_ylim(0, y_top)

    # Requested style: remove value text/title/x labels/legend.
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare top-1/2/3 item similarity metrics with 3 bar charts.")
    parser.add_argument("--alignrec", default="topk_AlignRec_item_similarity.txt")
    parser.add_argument("--anchorrec", default="topk_AnchorRec_item_similarity.txt")
    parser.add_argument("--out", default="topk_bar_compare.png")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if args.topk != 3:
        raise ValueError("This plot is defined for top-1, top-2, top-3 only. Use --topk 3.")

    align_rows = parse_topk(args.alignrec, topk=args.topk)
    anchor_rows = parse_topk(args.anchorrec, topk=args.topk)

    out = Path(args.out)
    stem = out.stem
    suffix = out.suffix if out.suffix else ".png"
    parent = out.parent if out.parent != Path("") else Path(".")

    _plot_metric(align_rows, anchor_rows, "2hop", str(parent / f"{stem}_2hop{suffix}"))
    _plot_metric(align_rows, anchor_rows, "vision_cos", str(parent / f"{stem}_vision_cos{suffix}"))
    _plot_metric(align_rows, anchor_rows, "text_cos", str(parent / f"{stem}_text_cos{suffix}"))


if __name__ == "__main__":
    main()
