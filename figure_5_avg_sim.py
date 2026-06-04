from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODELS = ["AlignRec", "AnchorRec", "FREEDOM", "DAMRS", "BM3", "LATTICE", "LGMRec", "SMORE", "VBPR"]
TWO_HOP_COLS = ["top1 2-hop items", "top2 2-hop items", "top3 2-hop items"]
TEXT_COLS = ["top1 text sim", "top2 text sim", "top3 text sim"]
VISION_COLS = ["top1 vision sim", "top2 vision sim", "top3 vision sim"]

OUTPUT_COLUMNS = [
    "model",
    "avg 2hop",
    "text 1",
    "text 2",
    "text 3",
    "vision 1",
    "vision 2",
    "vision 3",
]


INPUT_TO_OUTPUT = {
    "top1 text sim": "text 1",
    "top2 text sim": "text 2",
    "top3 text sim": "text 3",
    "top1 vision sim": "vision 1",
    "top2 vision sim": "vision 2",
    "top3 vision sim": "vision 3",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average figure 5 top-k 2-hop/text/vision similarity files.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sim-template", default="figure_5_{model}_sim.txt")
    parser.add_argument("--saved-emb-dir", default="saved_emb")
    parser.add_argument("--interactions", default="/home/hun/data/data/baby/baby.inter")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out", default="figure_5_avg_sim.txt")
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


def read_pipe_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=r"\s*\|\s*", engine="python")
    df.columns = [col.strip() for col in df.columns]
    return df


def l2_normalize(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def read_interactions(path: Path) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=None, engine="python")
    if "userID" not in df.columns or "itemID" not in df.columns:
        raise ValueError(f"{path} must contain userID and itemID columns.")

    item_to_users: dict[int, set[int]] = {}
    user_to_items: dict[int, set[int]] = {}
    for user_id, item_id in df[["userID", "itemID"]].dropna().itertuples(index=False):
        user_id = int(user_id)
        item_id = int(item_id)
        item_to_users.setdefault(item_id, set()).add(user_id)
        user_to_items.setdefault(user_id, set()).add(item_id)
    return item_to_users, user_to_items


def count_two_hop_items(
    item_id: int,
    item_to_users: dict[int, set[int]],
    user_to_items: dict[int, set[int]],
) -> int:
    users = item_to_users.get(item_id)
    if not users:
        return 0

    two_hop_items: set[int] = set()
    for user_id in users:
        two_hop_items.update(user_to_items.get(user_id, ()))
    two_hop_items.discard(item_id)
    return len(two_hop_items)


def build_two_hop_counts(
    n_items: int,
    item_to_users: dict[int, set[int]],
    user_to_items: dict[int, set[int]],
) -> np.ndarray:
    counts = np.zeros(n_items, dtype=np.int32)
    for item_id in range(n_items):
        counts[item_id] = count_two_hop_items(item_id, item_to_users, user_to_items)
    return counts


def topk_neighbors_by_cosine(emb: np.ndarray, topk: int, block_size: int) -> np.ndarray:
    n_items = emb.shape[0]
    neighbors = np.empty((n_items, topk), dtype=np.int32)

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        sims = emb[start:end] @ emb.T
        row_idx = np.arange(end - start)
        sims[row_idx, np.arange(start, end)] = -np.inf

        part = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        neighbors[start:end] = np.take_along_axis(part, order, axis=1).astype(np.int32)

    return neighbors


def avg_two_hop_from_final(
    model: str,
    saved_emb_dir: Path,
    two_hop_counts: np.ndarray,
    topk: int,
    block_size: int,
    dtype: np.dtype,
) -> float:
    final_item_path = saved_emb_dir / model / "final_item.npy"
    if not final_item_path.exists():
        raise FileNotFoundError(final_item_path)

    final_norm = l2_normalize(np.load(final_item_path), dtype)
    if final_norm.shape[0] != two_hop_counts.shape[0]:
        raise ValueError(
            f"{model}: final item count {final_norm.shape[0]} does not match 2-hop count "
            f"{two_hop_counts.shape[0]}."
        )

    neighbors = topk_neighbors_by_cosine(final_norm, topk=topk, block_size=block_size)
    return float(two_hop_counts[neighbors].mean())


def average_row(
    model: str,
    path: Path,
    saved_emb_dir: Path,
    two_hop_counts: np.ndarray,
    topk: int,
    block_size: int,
    dtype: np.dtype,
) -> dict[str, float | str]:
    df = read_pipe_table(path)
    missing = [col for col in INPUT_TO_OUTPUT if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    row: dict[str, float | str] = {"model": model}
    if all(col in df.columns for col in TWO_HOP_COLS):
        two_hop_values = df[TWO_HOP_COLS].apply(pd.to_numeric)
        row["avg 2hop"] = float(two_hop_values.to_numpy().mean())
    else:
        row["avg 2hop"] = avg_two_hop_from_final(
            model,
            saved_emb_dir,
            two_hop_counts,
            topk,
            block_size,
            dtype,
        )

    for input_col, output_col in INPUT_TO_OUTPUT.items():
        row[output_col] = float(pd.to_numeric(df[input_col]).mean())
    return row


def save_rows(rows: list[dict[str, float | str]], out_path: Path, precision: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(" | ".join(OUTPUT_COLUMNS) + "\n")
        for row in rows:
            values = []
            for col in OUTPUT_COLUMNS:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.{precision}f}")
                else:
                    values.append(str(value))
            f.write(" | ".join(values) + "\n")


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)
    item_to_users, user_to_items = read_interactions(Path(args.interactions))
    first_path = Path(args.sim_template.format(model=args.models[0]))
    n_items = len(read_pipe_table(first_path))
    two_hop_counts = build_two_hop_counts(n_items, item_to_users, user_to_items)

    rows = []
    for model in args.models:
        sim_path = Path(args.sim_template.format(model=model))
        rows.append(
            average_row(
                model,
                sim_path,
                Path(args.saved_emb_dir),
                two_hop_counts,
                args.topk,
                args.block_size,
                dtype,
            )
        )
    save_rows(rows, Path(args.out), args.precision)
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
