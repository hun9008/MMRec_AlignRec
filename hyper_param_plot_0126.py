# hyper_param_plot_0126.py
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


def safe_sort_key(x: Any):
    # 숫자는 숫자대로, 문자열은 문자열대로 정렬되게
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))


def plot_hp_mean(df: pd.DataFrame, hp: str, metric: str, out_path: Path):
    # seed 포함 다른 파라미터 모두 "마진" 처리: hp별 metric 평균
    g = df.groupby(hp, dropna=False)[metric].mean().reset_index()
    g = g.sort_values(by=hp, key=lambda col: col.map(safe_sort_key))

    plt.figure()
    plt.plot(g[hp].values, g[metric].values, marker="o")
    plt.xlabel(hp)
    plt.ylabel(metric)
    plt.title(f"{metric} mean by {hp} (avg over seeds & other params)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument(
        "--metric",
        type=str,
        default="valid_r20",
        choices=["valid_r20", "test_r20", "valid_n20", "test_n20"],
    )
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
        out_dir = csv_path.parent
    elif args.log_path:
        log_path = Path(args.log_path)
        out_dir = log_path.parent / (log_path.name + "_plots_r20")
        csv_path = out_dir / "parsed_r20.csv"
    else:
        raise RuntimeError("Provide --csv_path or --log_path.")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("CSV is empty.")

    # metric 없는 row 제거
    df = df.dropna(subset=[args.metric]).copy()

    # seed 제외하고 hp 목록 구성
    hp_cols = [c for c in df.columns if c not in {"seed", "valid_r20", "test_r20", "valid_n20", "test_n20"}]

    out_dir.mkdir(parents=True, exist_ok=True)
    for hp in hp_cols:
        out_path = out_dir / f"{args.metric}_mean_by_{hp}.png"
        plot_hp_mean(df, hp, args.metric, out_path)

    print("[OK] loaded csv :", csv_path)
    print("[OK] saved plots:", out_dir)


if __name__ == "__main__":
    main()
