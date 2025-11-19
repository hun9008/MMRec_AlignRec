#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Overlap@K between multiple recommendation lists.

- Auto-discovers files: <dir>/rec_top{K}_*.npy  where each is shape (n_users, K)
- Aliases inferred from filenames (e.g., rec_top20_final.npy -> 'final')
- Computes pairwise mean Overlap@K over users
- Saves CSVs and heatmap PNGs

Usage:
  python rec_overlap.py --dir saved_emb/rec_output --k 20
  python rec_overlap.py --dir saved_emb/rec_output --k 20 --only final id text vision
  python rec_overlap.py --dir saved_emb/rec_output --k 10 20   # multiple K
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


# --------------------------- Args ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True,
                   help="Directory containing rec_top{K}_*.npy files.")
    p.add_argument("--k", type=int, nargs="+", required=True,
                   help="K values to evaluate, e.g., --k 20 or --k 10 20 50")
    p.add_argument("--only", type=str, nargs="*",
                   help="Restrict to these aliases (e.g., final id text vision idmm).")
    p.add_argument("--img_dpi", type=int, default=200)
    return p.parse_args()


# --------------------- File discovery ----------------------

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


def load_rec_lists(filemap: Dict[str, str],
                   only: List[str] = None) -> Tuple[List[str], List[np.ndarray]]:
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


# ---------------------- Metrics ----------------------------

def pairwise_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean Overlap@K over users for two (n_users, K) arrays."""
    K = A.shape[1]
    vals = []
    for a, b in zip(A, B):
        vals.append(len(set(a) & set(b)) / float(K))
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


# ---------------- Label & I/O helpers ----------------------

def pretty_label(alias: str) -> str:
    """
    Heatmap 축에 표시할 라벨 (TeX 수식 등으로 바꾸고 싶으면 여기 수정).
    예) 'final' -> r"$\\text{final}$" 처럼 마음대로 매핑 가능.
    """
    mapping = {
        # 예시 (원하면 채워서 사용)
        "final": r"$h^i$",
        "id":    r"$h^i_{id}$",
        "mm":    r"$h^i_{mm}$",
        "text":  r"$h^i_{t}$",
        "vision":r"$h^i_{v}$",
    }
    return mapping.get(alias, alias)


def save_csv(path: str, labels: List[str], M: np.ndarray):
    with open(path, "w") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "," + ",".join(f"{M[i,j]:.6f}" for j in range(len(labels))) + "\n")
    print(f"[SAVE] CSV -> {path}")


def save_heatmap(path: str, title: str,
                 labels: List[str],
                 M: np.ndarray,
                 dpi: int = 200):
    m = len(labels)
    fig, ax = plt.subplots(figsize=(1.2 + 0.6*m, 1.2 + 0.6*m))
    im = ax.imshow(M, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(m))
    ax.set_yticks(range(m))

    display_labels = [pretty_label(a) for a in labels]

    # x축: 기본 0도, final(/final_emb)만 45도
    ax.set_xticklabels(display_labels, fontsize=14)
    for tick, alias in zip(ax.get_xticklabels(), labels):
        if alias in ("final", "final_emb"):
            tick.set_rotation(0)
            tick.set_ha("right")
        else:
            tick.set_rotation(0)
            tick.set_ha("center")

    # y축: 같은 라벨, 오른쪽 정렬
    ax.set_yticklabels(display_labels, fontsize=14)
    for tick in ax.get_yticklabels():
        tick.set_ha("right")

    # ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if m <= 20:
        for i in range(m):
            for j in range(m):
                ax.text(
                    j, i,
                    ("1.00" if i == j else f"{M[i,j]:.2f}"),
                    ha="center", va="center", fontsize=8
                )
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[SAVE] Heatmap -> {path}")


# --------------------------- Main ---------------------------

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

        # Save CSV
        save_csv(os.path.join(img_dir, f"rec_overlap_k{K}.csv"), labels, M_overlap)

        # Save heatmap
        save_heatmap(
            os.path.join(img_dir, f"rec_overlap_k{K}_final.png"),
            f"Recommendation Overlap@{K} (mean over users)",
            labels,
            M_overlap,
            dpi=args.img_dpi,
        )

        # Console table
        header = [""] + labels
        widths = [max(12, max(len(h) for h in header))]
        for h in header[1:]:
            widths.append(max(12, len(h)))
        fmt = "  ".join("{:<" + str(w) + "}" for w in widths)

        print(f"\n=== Overlap@{K} ===")
        print(fmt.format(*header))
        for i, lab in enumerate(labels):
            row = [lab] + [("--" if i == j else f"{M_overlap[i,j]:.4f}")
                           for j in range(len(labels))]
            print(fmt.format(*row))


if __name__ == "__main__":
    main()