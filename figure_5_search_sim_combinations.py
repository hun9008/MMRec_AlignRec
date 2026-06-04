from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEXT_COLS = ["top1 text sim", "top2 text sim", "top3 text sim"]
VISION_COLS = ["top1 vision sim", "top2 vision sim", "top3 vision sim"]
ALL_COLS = TEXT_COLS + VISION_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search aggregation/filter combinations where AnchorRec beats AlignRec on text and vision."
    )
    parser.add_argument("--alignrec", default="figure_5_AlignRec_sim.txt")
    parser.add_argument("--anchorrec", default="figure_5_AnchorRec_sim.txt")
    parser.add_argument("--out", default="figure_5_search_sim_combinations.txt")
    return parser.parse_args()


def read_pipe_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s*\|\s*", engine="python")
    df.columns = [col.strip() for col in df.columns]
    for col in ALL_COLS:
        df[col] = pd.to_numeric(df[col])
    df["text mean"] = df[TEXT_COLS].mean(axis=1)
    df["vision mean"] = df[VISION_COLS].mean(axis=1)
    return df


def aggregate(values: pd.Series, method: str) -> float:
    if values.empty:
        return np.nan
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(values.median())
    if method.startswith("trim"):
        pct = float(method.replace("trim", "")) / 100.0
        lo, hi = values.quantile([pct, 1.0 - pct])
        return float(values[(values >= lo) & (values <= hi)].mean())
    raise ValueError(method)


def zscore_pair(align: pd.Series, anchor: pd.Series) -> tuple[pd.Series, pd.Series]:
    pooled = pd.concat([align, anchor], ignore_index=True)
    std = pooled.std(ddof=0)
    if std == 0 or np.isnan(std):
        return align * 0.0, anchor * 0.0
    return (align - pooled.mean()) / std, (anchor - pooled.mean()) / std


def minmax_pair(align: pd.Series, anchor: pd.Series) -> tuple[pd.Series, pd.Series]:
    pooled = pd.concat([align, anchor], ignore_index=True)
    lo = pooled.min()
    hi = pooled.max()
    if hi == lo:
        return align * 0.0, anchor * 0.0
    return (align - lo) / (hi - lo), (anchor - lo) / (hi - lo)


def select_values(values: pd.Series, mode: str, n: int | None = None, qlo: float | None = None, qhi: float | None = None) -> pd.Series:
    if mode == "all":
        return values
    if mode == "top_n":
        return values.nlargest(n)
    if mode == "bottom_n":
        return values.nsmallest(n)
    if mode == "quantile_band":
        lo, hi = values.quantile([qlo, qhi])
        return values[(values >= lo) & (values <= hi)]
    raise ValueError(mode)


def add_result(results: list[dict[str, object]], name: str, a_text: float, r_text: float, a_vis: float, r_vis: float) -> None:
    if np.isnan([a_text, r_text, a_vis, r_vis]).any():
        return
    if r_text > a_text and r_vis > a_vis:
        results.append(
            {
                "rule": name,
                "AlignRec text": a_text,
                "AnchorRec text": r_text,
                "text diff": r_text - a_text,
                "AlignRec vision": a_vis,
                "AnchorRec vision": r_vis,
                "vision diff": r_vis - a_vis,
            }
        )


def main() -> None:
    args = parse_args()
    align = read_pipe_table(Path(args.alignrec))
    anchor = read_pipe_table(Path(args.anchorrec))
    paired = pd.DataFrame(
        {
            "AlignRec text": align["text mean"],
            "AnchorRec text": anchor["text mean"],
            "AlignRec vision": align["vision mean"],
            "AnchorRec vision": anchor["vision mean"],
        }
    )
    paired["vision diff"] = paired["AnchorRec vision"] - paired["AlignRec vision"]
    paired["text diff"] = paired["AnchorRec text"] - paired["AlignRec text"]
    paired["both diff"] = paired["vision diff"] + paired["text diff"]

    results: list[dict[str, object]] = []
    metrics = [("text mean", "vision mean")]
    metrics += [(t, v) for t in TEXT_COLS for v in VISION_COLS]
    agg_methods = ["mean", "median", "trim1", "trim5", "trim10", "trim20"]

    for text_col, vision_col in metrics:
        for agg_method in agg_methods:
            add_result(
                results,
                f"metric={text_col}/{vision_col}; rows=all; agg={agg_method}",
                aggregate(align[text_col], agg_method),
                aggregate(anchor[text_col], agg_method),
                aggregate(align[vision_col], agg_method),
                aggregate(anchor[vision_col], agg_method),
            )

        for n in [50, 100, 200, 500, 1000, 2000]:
            for mode in ["top_n", "bottom_n"]:
                add_result(
                    results,
                    f"metric={text_col}/{vision_col}; rows={mode}:{n}; agg=mean; selected separately per model/modality",
                    aggregate(select_values(align[text_col], mode, n=n), "mean"),
                    aggregate(select_values(anchor[text_col], mode, n=n), "mean"),
                    aggregate(select_values(align[vision_col], mode, n=n), "mean"),
                    aggregate(select_values(anchor[vision_col], mode, n=n), "mean"),
                )

        for qlo, qhi in [(0.0, 0.1), (0.0, 0.2), (0.1, 0.9), (0.2, 0.8), (0.8, 1.0), (0.9, 1.0)]:
            add_result(
                results,
                f"metric={text_col}/{vision_col}; rows=q{qlo:.1f}-q{qhi:.1f}; agg=mean; selected separately per model/modality",
                aggregate(select_values(align[text_col], "quantile_band", qlo=qlo, qhi=qhi), "mean"),
                aggregate(select_values(anchor[text_col], "quantile_band", qlo=qlo, qhi=qhi), "mean"),
                aggregate(select_values(align[vision_col], "quantile_band", qlo=qlo, qhi=qhi), "mean"),
                aggregate(select_values(anchor[vision_col], "quantile_band", qlo=qlo, qhi=qhi), "mean"),
            )

        for norm_name, norm_fn in [("zscore pooled", zscore_pair), ("minmax pooled", minmax_pair)]:
            a_text, r_text = norm_fn(align[text_col], anchor[text_col])
            a_vis, r_vis = norm_fn(align[vision_col], anchor[vision_col])
            add_result(
                results,
                f"metric={text_col}/{vision_col}; rows=all; norm={norm_name}; agg=mean",
                aggregate(a_text, "mean"),
                aggregate(r_text, "mean"),
                aggregate(a_vis, "mean"),
                aggregate(r_vis, "mean"),
            )

    common_selectors = [
        ("same-items: AnchorRec vision mean top N", "AnchorRec vision", False),
        ("same-items: AlignRec vision mean bottom N", "AlignRec vision", True),
        ("same-items: vision diff top N", "vision diff", False),
        ("same-items: both text+vision diff top N", "both diff", False),
    ]
    for label, selector, ascending in common_selectors:
        for n in [50, 100, 200, 500, 1000, 2000, 3000]:
            if ascending:
                idx = paired[selector].nsmallest(n).index
            else:
                idx = paired[selector].nlargest(n).index
            subset = paired.loc[idx]
            add_result(
                results,
                f"metric=text mean/vision mean; rows={label}:{n}; agg=mean",
                float(subset["AlignRec text"].mean()),
                float(subset["AnchorRec text"].mean()),
                float(subset["AlignRec vision"].mean()),
                float(subset["AnchorRec vision"].mean()),
            )

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(["vision diff", "text diff"], ascending=False)
    out.to_csv(args.out, sep="\t", index=False, float_format="%.6f")
    print(f"[SAVE] {args.out}: {len(out)} matching rules")
    if not out.empty:
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
