#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Overlap@K (and Jaccard@K) between multiple recommendation lists.

- Auto-discovers files: <dir>/rec_top{K}_*.npy  where each is shape (n_users, K)
- Aliases inferred from filenames (e.g., rec_top20_final.npy -> 'final')
- Computes pairwise mean Overlap@K and Jaccard@K over users
- Saves CSVs and heatmap PNGs

Usage:
  python rec_overlap.py --dir saved_emb/rec_output --k 20
  python rec_overlap.py --dir saved_emb/rec_output --k 20 --only final id text vision
  python rec_overlap.py --dir saved_emb/rec_output --k 10 20   # multiple K

Outputs (under <dir>/images):
  overlap_k{K}.csv, jaccard_k{K}.csv, overlap_k{K}.png, jaccard_k{K}.png
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True, help="Directory containing rec_top{K}_*.npy files.")
    p.add_argument("--k", type=int, nargs="+", required=True, help="K values to evaluate, e.g., --k 20 or --k 10 20 50")
    p.add_argument("--only", type=str, nargs="*", help="Restrict to these aliases (e.g., final id text vision idmm).")
    p.add_argument("--img_dpi", type=int, default=200)
    return p.parse_args()

def find_rec_files(base_dir: str, K: int) -> Dict[str, str]:
    """Return mapping alias -> filepath for rec_top{K}_{alias}.npy"""
    pattern = os.path.join(base_dir, f"rec_top{K}_*.npy")
    files = glob.glob(pattern)
    out = {}
    for f in files:
        name = os.path.basename(f)
        # expect: rec_top{K}_{alias}.npy
        try:
            alias = name.rsplit(".npy", 1)[0].split(f"rec_top{K}_", 1)[1]
            out[alias] = f
        except Exception:
            continue
    return out

def load_rec_lists(filemap: Dict[str, str], only: List[str] = None) -> Tuple[List[str], List[np.ndarray]]:
    aliases = sorted(filemap.keys())
    if only:
        aliases = [a for a in aliases if a in set(only)]
    if not aliases:
        raise SystemExit("No recommendation files matched the selection.")
    recs = []
    shapes = set()
    for a in aliases:
        arr = np.load(filemap[a])
        if arr.ndim != 2:
            raise ValueError(f"{filemap[a]} is not (n_users, K). Got shape {arr.shape}")
        recs.append(arr)
        shapes.add(arr.shape)
        print(f"[LOAD] {a}: shape={arr.shape} from {filemap[a]}")
    if len({s[0] for s in shapes}) != 1:
        raise ValueError(f"All rec files must have same n_users. Got {shapes}")
    if len({s[1] for s in shapes}) != 1:
        raise ValueError(f"All rec files must have same K. Got {shapes}")
    return aliases, recs

def pairwise_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean Overlap@K over users for two (n_users, K) arrays."""
    K = A.shape[1]
    vals = []
    for a, b in zip(A, B):
        vals.append(len(set(a) & set(b)) / float(K))
    return float(np.mean(vals))

def pairwise_jaccard(A: np.ndarray, B: np.ndarray) -> float:
    """Mean Jaccard@K over users for two (n_users, K) arrays."""
    vals = []
    for a, b in zip(A, B):
        sa, sb = set(a), set(b)
        inter = len(sa & sb); uni = len(sa | sb)
        vals.append(0.0 if uni == 0 else inter / float(uni))
    return float(np.mean(vals))

def compute_matrix(func, arrays: List[np.ndarray]) -> np.ndarray:
    m = len(arrays)
    M = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i == j:
                M[i, j] = 1.0
            else:
                M[i, j] = func(arrays[i], arrays[j])
    return M

def save_csv(path: str, labels: List[str], M: np.ndarray):
    with open(path, "w") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "," + ",".join(f"{M[i,j]:.6f}" for j in range(len(labels))) + "\n")
    print(f"[SAVE] CSV -> {path}")

def save_heatmap(path: str, title: str, labels: List[str], M: np.ndarray, dpi: int = 200):
    m = len(labels)
    fig, ax = plt.subplots(figsize=(1.2 + 0.6*m, 1.2 + 0.6*m))
    im = ax.imshow(M, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(m)); ax.set_yticks(range(m))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if m <= 20:
        for i in range(m):
            for j in range(m):
                ax.text(j, i, ("1.00" if i == j else f"{M[i,j]:.2f}"),
                        ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi); plt.close(fig)
    print(f"[SAVE] Heatmap -> {path}")

def main():
    args = parse_args()
    img_dir = os.path.join(args.dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    for K in args.k:
        filemap = find_rec_files(args.dir, K)
        if not filemap:
            print(f"[WARN] No files found for K={K} in {args.dir}")
            continue

        labels, arrays = load_rec_lists(filemap, only=args.only)
        print(f"[INFO] aliases = {labels}")

        M_overlap = compute_matrix(pairwise_overlap, arrays)
        M_jaccard = compute_matrix(pairwise_jaccard, arrays)

        # Save CSV
        save_csv(os.path.join(img_dir, f"rec_overlap_k{K}.csv"), labels, M_overlap)
        save_csv(os.path.join(img_dir, f"rec_jaccard_k{K}.csv"), labels, M_jaccard)

        # Save heatmaps
        save_heatmap(os.path.join(img_dir, f"rec_overlap_k{K}.png"),
                     f"Recommendation Overlap@{K} (mean over users)", labels, M_overlap, dpi=args.img_dpi)
        save_heatmap(os.path.join(img_dir, f"rec_jaccard_k{K}.png"),
                     f"Recommendation Jaccard@{K} (mean over users)", labels, M_jaccard, dpi=args.img_dpi)

        # Console table
        def print_table(name, M):
            header = [""] + labels
            widths = [max(12, max(len(h) for h in header))]
            for h in header[1:]:
                widths.append(max(12, len(h)))
            fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
            print(f"\n=== {name}@{K} ===")
            print(fmt.format(*header))
            for i, lab in enumerate(labels):
                row = [lab] + [("--" if i==j else f"{M[i,j]:.4f}") for j in range(len(labels))]
                print(fmt.format(*row))

        print_table("Overlap", M_overlap)
        print_table("Jaccard", M_jaccard)

if __name__ == "__main__":
    main()


"""
python rec_overlap.py --dir saved_emb/rec_output --k 20
# 또는 특정 조합만
python rec_overlap.py --dir saved_emb/rec_output --k 20 --only final id text vision idmm
# 여러 K
python rec_overlap.py --dir saved_emb/rec_output --k 10 20 50
"""