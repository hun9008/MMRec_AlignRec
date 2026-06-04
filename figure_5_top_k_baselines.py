from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODELS = ["DAMRS", "BM3", "LATTICE", "LGMRec", "SMORE", "VBPR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figure 5 top-k text/vision similarity files for baseline models."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--saved-emb-dir", default="saved_emb")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--text-feat", default="/home/hun/data/data/baby/text_feat.npy")
    parser.add_argument("--vision-feat", default="/home/hun/data/data/baby/image_feat.npy")
    parser.add_argument("--interactions", default="/home/hun/data/data/baby/baby.inter")
    parser.add_argument("--item-mapping", default="/home/hun/data/data/baby/i_id_mapping.csv")
    parser.add_argument("--out-prefix", default="figure_5")
    return parser.parse_args()


def l2_normalize(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def validate_item_count(n_items: int, mapping_path: Path) -> None:
    if not mapping_path.exists():
        return

    mapping = pd.read_csv(mapping_path, sep=None, engine="python")
    if "itemID" not in mapping.columns:
        raise ValueError(f"{mapping_path} must contain an itemID column.")

    item_ids = mapping["itemID"].astype(int)
    if len(mapping) != n_items:
        raise ValueError(f"item mapping count={len(mapping)} does not match embedding count={n_items}.")
    if item_ids.min() != 0 or item_ids.max() != n_items - 1:
        raise ValueError(
            f"itemID range must be 0..{n_items - 1}, got {item_ids.min()}..{item_ids.max()}."
        )


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


def save_similarity_file(
    model_name: str,
    final_item_path: Path,
    out_path: Path,
    text_norm: np.ndarray,
    vision_norm: np.ndarray,
    two_hop_counts: np.ndarray,
    topk: int,
    block_size: int,
    dtype: np.dtype,
) -> None:
    final_item = np.load(final_item_path)
    if final_item.shape[0] != text_norm.shape[0] or final_item.shape[0] != vision_norm.shape[0]:
        raise ValueError(
            f"{model_name}: final/text/vision item counts must match. "
            f"got final={final_item.shape}, text={text_norm.shape}, vision={vision_norm.shape}"
        )

    final_norm = l2_normalize(final_item, dtype)
    neighbors = topk_neighbors_by_cosine(final_norm, topk=topk, block_size=block_size)

    item_idx = np.arange(final_norm.shape[0])[:, None]
    vision_sims = np.sum(vision_norm[item_idx] * vision_norm[neighbors], axis=2)
    text_sims = np.sum(text_norm[item_idx] * text_norm[neighbors], axis=2)
    two_hop_items = two_hop_counts[neighbors]

    header_cols = ["itemID"]
    header_cols += [f"top{i} itemID" for i in range(1, topk + 1)]
    header_cols += [f"top{i} 2-hop items" for i in range(1, topk + 1)]
    header_cols += [f"top{i} vision sim" for i in range(1, topk + 1)]
    header_cols += [f"top{i} text sim" for i in range(1, topk + 1)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(" | ".join(header_cols) + "\n")
        for item_id in range(final_norm.shape[0]):
            values = [str(item_id)]
            values += [str(int(v)) for v in neighbors[item_id]]
            values += [str(int(v)) for v in two_hop_items[item_id]]
            values += [f"{v:.6f}" for v in vision_sims[item_id]]
            values += [f"{v:.6f}" for v in text_sims[item_id]]
            f.write(" | ".join(values) + "\n")

    print(f"[SAVE] {model_name}: {out_path} ({final_norm.shape[0]} items)")


def generate_for_models(
    models: list[str],
    saved_emb_dir: Path,
    out_prefix: str,
    text_feat: Path,
    vision_feat: Path,
    interactions: Path,
    item_mapping: Path,
    topk: int,
    block_size: int,
    dtype: np.dtype,
) -> None:
    text_norm = l2_normalize(np.load(text_feat), dtype)
    vision_norm = l2_normalize(np.load(vision_feat), dtype)
    if text_norm.shape[0] != vision_norm.shape[0]:
        raise ValueError(f"text/vision item counts differ: {text_norm.shape} vs {vision_norm.shape}")
    if topk < 1:
        raise ValueError("--topk must be >= 1.")
    if topk >= text_norm.shape[0]:
        raise ValueError(f"--topk must be smaller than item count ({text_norm.shape[0]}).")

    validate_item_count(text_norm.shape[0], item_mapping)
    item_to_users, user_to_items = read_interactions(interactions)
    two_hop_counts = build_two_hop_counts(text_norm.shape[0], item_to_users, user_to_items)

    for model_name in models:
        final_item_path = saved_emb_dir / model_name / "final_item.npy"
        if not final_item_path.exists():
            raise FileNotFoundError(final_item_path)
        out_path = Path(f"{out_prefix}_{model_name}_sim.txt")
        save_similarity_file(
            model_name,
            final_item_path,
            out_path,
            text_norm,
            vision_norm,
            two_hop_counts,
            topk,
            block_size,
            dtype,
        )


def main() -> None:
    args = parse_args()
    generate_for_models(
        models=args.models,
        saved_emb_dir=Path(args.saved_emb_dir),
        out_prefix=args.out_prefix,
        text_feat=Path(args.text_feat),
        vision_feat=Path(args.vision_feat),
        interactions=Path(args.interactions),
        item_mapping=Path(args.item_mapping),
        topk=args.topk,
        block_size=args.block_size,
        dtype=np.dtype(args.dtype),
    )


if __name__ == "__main__":
    main()
