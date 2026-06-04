from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each item, find top-k similar items from FREEDOM final item embeddings "
            "and save text/vision cosine similarities."
        )
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--dtype", default="float32")

    parser.add_argument("--freedom-final", default="saved_emb/FREEDOM/final_item.npy")
    parser.add_argument("--text-feat", default="/home/hun/data/data/baby/text_feat.npy")
    parser.add_argument("--vision-feat", default="/home/hun/data/data/baby/image_feat.npy")
    parser.add_argument("--item-mapping", default="/home/hun/data/data/baby/i_id_mapping.csv")
    parser.add_argument("--out", default="figure_5_FREEDOM_sim.txt")
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
    final_item_path: Path,
    out_path: Path,
    text_norm: np.ndarray,
    vision_norm: np.ndarray,
    topk: int,
    block_size: int,
    dtype: np.dtype,
) -> None:
    final_item = np.load(final_item_path)
    if final_item.shape[0] != text_norm.shape[0] or final_item.shape[0] != vision_norm.shape[0]:
        raise ValueError(
            "FREEDOM final/text/vision item counts must match. "
            f"got final={final_item.shape}, text={text_norm.shape}, vision={vision_norm.shape}"
        )

    final_norm = l2_normalize(final_item, dtype)
    neighbors = topk_neighbors_by_cosine(final_norm, topk=topk, block_size=block_size)

    item_idx = np.arange(final_norm.shape[0])[:, None]
    vision_sims = np.sum(vision_norm[item_idx] * vision_norm[neighbors], axis=2)
    text_sims = np.sum(text_norm[item_idx] * text_norm[neighbors], axis=2)

    header_cols = ["itemID"]
    header_cols += [f"top{i} vision sim" for i in range(1, topk + 1)]
    header_cols += [f"top{i} text sim" for i in range(1, topk + 1)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(" | ".join(header_cols) + "\n")
        for item_id in range(final_norm.shape[0]):
            values = [str(item_id)]
            values += [f"{v:.6f}" for v in vision_sims[item_id]]
            values += [f"{v:.6f}" for v in text_sims[item_id]]
            f.write(" | ".join(values) + "\n")

    print(f"[SAVE] FREEDOM: {out_path} ({final_norm.shape[0]} items)")


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    if args.topk < 1:
        raise ValueError("--topk must be >= 1.")

    text_norm = l2_normalize(np.load(args.text_feat), dtype)
    vision_norm = l2_normalize(np.load(args.vision_feat), dtype)
    if text_norm.shape[0] != vision_norm.shape[0]:
        raise ValueError(f"text/vision item counts differ: {text_norm.shape} vs {vision_norm.shape}")
    if args.topk >= text_norm.shape[0]:
        raise ValueError(f"--topk must be smaller than item count ({text_norm.shape[0]}).")

    validate_item_count(text_norm.shape[0], Path(args.item_mapping))
    save_similarity_file(
        Path(args.freedom_final),
        Path(args.out),
        text_norm,
        vision_norm,
        args.topk,
        args.block_size,
        dtype,
    )


if __name__ == "__main__":
    main()
