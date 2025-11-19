#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Top-K neighbor overlaps between multiple item-embedding files (.npy).

- Loads multiple (N, D) item-embedding matrices
- L2-normalizes
- Finds Top-K nearest neighbors per item (cosine; dot after L2)
- Computes Overlap@K:
    * Pairwise between every pair (mean over items)
    * Optional n-way (--multi aliases) intersection overlap
- Saves neighbor indices (optional), a CSV, and heatmaps

Usage:
  # auto-discover inside <DIR>/0907_all_anchor
  python overlap.py --variant anchor --dir saved_emb --k 20

  # auto-discover inside <DIR>/0907_all_alignrec
  python overlap.py --variant alignrec --dir saved_emb --k 20

  # explicit files (overrides variant/dir)
  python overlap.py --files a.npy b.npy c.npy --k 50

  # n-way overlap for specific aliases (must exist among discovered files)
  python overlap.py --variant anchor --k 20 --multi proj_id proj_text final_emb
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


# --------------------------- Args ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["anchor", "alignrec"], default="anchor",
                   help="Which bundle of embeddings to use when auto-discovering.")
    p.add_argument("--dir", type=str, default="saved_emb",
                   help="Base directory. The script will search <dir>/0907_all_<variant>/ unless --files is given.")
    p.add_argument("--files", type=str, nargs="*",
                   help="Explicit list of .npy embeddings. If set, overrides --variant/--dir discovery.")
    p.add_argument("--k", type=int, default=20, help="Top-K neighbors (self excluded).")
    p.add_argument("--block_size", type=int, default=4096,
                   help="Block size for similarity computation to control memory.")
    p.add_argument("--save_neighbors", action="store_true",
                   help="Save neighbor indices as .npy next to each embedding.")
    p.add_argument("--suffix", type=str, default="neighbors",
                   help="Suffix for saved neighbor files (e.g., file.neighbors.npy).")
    p.add_argument("--seed", type=int, default=42, help="Seed for determinism (ties/argpartition).")
    p.add_argument("--multi", nargs="+",
                   help="Compute n-way overlap for the given aliases (e.g. proj_id proj_text final_emb).")
    return p.parse_args()


# ---------------------- Core utilities ----------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return X / norms


def topk_neighbors_blockwise(X: np.ndarray, k: int, block_size: int = 4096) -> np.ndarray:
    """
    Efficient Top-K nearest neighbors for cosine similarity using block-wise matmul.
    - X must be L2-normalized.
    - Returns indices shape (N, k), self excluded.
    """
    N, d = X.shape
    if k >= N:
        raise ValueError(f"k ({k}) must be < N ({N}).")

    neighbors = np.empty((N, k), dtype=np.int32)
    XT = X.T

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Q = X[start:end]               # (B, d)
        S = Q @ XT                     # (B, N)

        # exclude self
        for i in range(start, end):
            S[i - start, i] = -np.inf

        # unordered top-k via argpartition
        part = np.argpartition(-S, kth=range(k), axis=1)[:, :k]   # (B, k)
        row_idx = np.arange(end - start)[:, None]
        part_scores = S[row_idx, part]
        order = np.argsort(-part_scores, axis=1)                  # sort desc
        topk = part[row_idx, order]
        neighbors[start:end] = topk.astype(np.int32)

    return neighbors


def pairwise_overlap_at_k(neigh_a: np.ndarray, neigh_b: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """
    Overlap@K between two neighbor index arrays (N, k).
    Returns per-item overlaps and mean overlap.
    """
    assert neigh_a.shape == neigh_b.shape, "Neighbor arrays must have same shape"
    N = neigh_a.shape[0]
    overlaps = np.empty(N, dtype=np.float32)
    for i in range(N):
        overlaps[i] = len(set(neigh_a[i]).intersection(neigh_b[i])) / float(k)
    return overlaps, float(np.mean(overlaps))


def multi_overlap_at_k(neigh_list: List[np.ndarray], k: int) -> Tuple[np.ndarray, float]:
    """
    n-way Overlap@K for a list of neighbor arrays [(N,k), ...].
    Per item i, compute size( ⋂_m TopK_m(i) ) / k. Returns per-item and mean.
    """
    assert len(neigh_list) >= 2, "Need >=2 neighbor arrays for multi-overlap."
    N, kk = neigh_list[0].shape
    assert kk == k
    for A in neigh_list[1:]:
        assert A.shape == (N, k)
    overlaps = np.empty(N, dtype=np.float32)
    for i in range(N):
        sets = [set(A[i]) for A in neigh_list]
        inter = set.intersection(*sets)
        overlaps[i] = len(inter) / float(k)
    return overlaps, float(overlaps.mean())


# -------------------- Discovery (aliases) -------------------

def discover_default_files(base_dir: str, variant: str):
    """Return (files, alias_map, search_dir)."""
    search_dir = os.path.join(base_dir)

    if variant == "anchor":
        candidates = [
            "item_emb_align_id.npy",
            "item_emb_align_mm.npy",
            # "item_emb_align_projavg.npy",
            "item_emb_align_text.npy",
            "item_emb_align_vision.npy",
            "item_emb_final_alignrec.npy",
            "item_emb_mm_out.npy",
            "item_emb_raw_id.npy",
            "item_feat_raw_text.npy",
            "item_feat_raw_vision.npy",
            "../../data/baby_beit3_128token_add_title_brand_to_text/image_feat.npy",
            "../../data/baby/text_feat.npy",
            "../../data/baby/image_feat.npy",
        ]
        aliases = [
            "proj_id",
            "proj_mm",
            # "align_proj",
            "proj_text",
            "proj_vision",
            "final_emb",
            "origin_mm",
            "origin_id",
            "origin_text",
            "origin_vision",
            "raw_mm",
            "raw_text",
            "raw_vision",
        ]
    else:  # alignrec
        candidates = [
            "item_emb_final_alignrec.npy",
            "item_emb_mm_out.npy",
            "item_emb_raw_id.npy",
            "../../data/baby_beit3_128token_add_title_brand_to_text/image_feat.npy",
            "../../data/baby/text_feat.npy",
            "../../data/baby/image_feat.npy",
        ]
        aliases = [
            "final_emb",
            "origin_mm",
            "origin_id",
            "raw_mm",
            "raw_text",
            "raw_vision",
        ]

    found = []
    alias_map: Dict[str, str] = {}

    # 후보들 중 존재하는 파일만 수집 (절대경로 키, alias 값)
    for name, alias in zip(candidates, aliases):
        p = os.path.normpath(os.path.join(search_dir, name))
        if os.path.isfile(p):
            p_abs = os.path.abspath(p)
            found.append(p_abs)
            alias_map[p_abs] = alias

    # 후보가 하나도 안 잡히면 디렉토리 내 *.npy 폴백
    if not found:
        for p in sorted(glob.glob(os.path.join(search_dir, "*.npy"))):
            p_abs = os.path.abspath(p)
            found.append(p_abs)
            alias_map[p_abs] = os.path.splitext(os.path.basename(p_abs))[0]

    return found, alias_map, search_dir


# ---------- ordering & label helpers ----------

def anchor_alias_sort_key(alias: str) -> Tuple[int, int]:
    """
    anchor 용 정렬 기준:
      group: raw(0), origin(1), proj(2), final(3+)
      modality: id(0), mm(1), text(2), vision(3), 기타(4)
    """
    # group
    if alias.startswith("raw_"):
        g = 0
    elif alias.startswith("origin_"):
        g = 1
    elif alias.startswith("proj_"):
        g = 2
    elif alias == "final_emb":
        g = 3
    else:
        g = 4

    # modality
    if alias.endswith("_id"):
        m = 0
    elif alias.endswith("_mm"):
        m = 1
    elif alias.endswith("_text"):
        m = 2
    elif alias.endswith("_vision"):
        m = 3
    else:
        m = 4

    return (g, m)


def alignrec_alias_sort_key(alias: str) -> Tuple[int, int]:
    """
    alignrec 용 정렬 기준 (비슷한 규칙, 있는 것만 순서 강제).
    """
    if alias.startswith("raw_"):
        g = 0
    elif alias.startswith("origin_"):
        g = 1
    elif alias == "final_emb":
        g = 2
    else:
        g = 3

    if alias.endswith("_id"):
        m = 0
    elif alias.endswith("_mm"):
        m = 1
    elif alias.endswith("_text"):
        m = 2
    elif alias.endswith("_vision"):
        m = 3
    else:
        m = 4

    return (g, m)


def pretty_label(alias: str) -> str:
    """
    heatmap 축에 표시할 라벨 (TeX 수식 포함 가능).
    요청: proj_id -> $p_{id}$
    나머지는 필요하면 여기서 추가해서 쓰면 됨.
    """
    mapping = {
        "final_emb": r"$\mathbf{h^i}$",
        "proj_id": r"$\mathbf{p^i_{id}}$",
        # 필요하면 아래처럼 확장:
        "proj_mm":   r"$\mathbf{p^i_{mm}}$",
        "proj_text": r"$\mathbf{p^i_{t}}$",
        "proj_vision": r"$\mathbf{p^i_{v}}$",
        "origin_id": r"$\mathbf{h^i_{id}}$",
        "origin_mm": r"$\mathbf{h^i_{mm}}$",
        "origin_text": r"$\mathbf{h^i_{t}}$",
        "origin_vision": r"$\mathbf{h^i_{v}}$",
        "raw_mm": r"$\mathbf{f^i_{mm}}$",
        "raw_text": r"$\mathbf{f^i_{t}}$",
        "raw_vision": r"$\mathbf{f^i_{v}}$",
        # ...
    }
    return mapping.get(alias, alias)


# --------------------------- Main ---------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # 파일 목록 및 alias 설정
    if args.files:
        files = [os.path.abspath(f) for f in args.files]
        alias_map = {f: os.path.splitext(os.path.basename(f))[0] for f in files}
        search_dir = os.path.commonpath(files) if len(files) > 1 else os.path.dirname(files[0])
        variant_suffix = "custom"
    else:
        files, alias_map, search_dir = discover_default_files(args.dir, args.variant)
        variant_suffix = args.variant

    if not files:
        raise SystemExit(f"No embedding files found. Checked: {search_dir}")

    # where to save artifacts
    img_dir = os.path.join(search_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Load & normalize
    embs: Dict[str, np.ndarray] = {}
    shapes = set()
    for f in files:
        X = np.load(f)
        if X.ndim != 2:
            raise ValueError(f"{f} is not 2D (N, D). Got shape {X.shape}")
        embs[f] = l2_normalize(X.astype(np.float32))
        shapes.add(X.shape[0])
        name = alias_map.get(f, os.path.basename(f))
        print(f"[LOAD] {name} shape={X.shape}")

    if len(shapes) != 1:
        raise ValueError(f"All embeddings must have the same number of items. Got Ns={shapes}")
    N = next(iter(shapes))
    if args.k >= N:
        raise ValueError(f"--k ({args.k}) must be < number of items ({N})")

    # ---------- 정렬: raw → origin → proj → final, modality 순서 ----------
    keys = list(embs.keys())

    if variant_suffix == "anchor":
        keys.sort(key=lambda k: anchor_alias_sort_key(alias_map.get(k, os.path.basename(k))))
    elif variant_suffix == "alignrec":
        keys.sort(key=lambda k: alignrec_alias_sort_key(alias_map.get(k, os.path.basename(k))))
    # custom이면 순서 유지

    # # Neighbors
    # neighbors: Dict[str, np.ndarray] = {}
    # for f in keys:
    #     X = embs[f]
    #     name = alias_map.get(f, os.path.basename(f))
    #     print(f"[NEIGHBORS] Computing Top-{args.k} for {name} ...")
    #     neigh = topk_neighbors_blockwise(X, k=args.k, block_size=args.block_size)
    #     neighbors[f] = neigh
    #     if args.save_neighbors:
    #         out_path = f"{f.rsplit('.npy', 1)[0]}.{args.suffix}.k{args.k}.npy"
    #         np.save(out_path, neigh)
    #         print(f"  -> saved neighbors: {out_path}")
    # Neighbors
    neighbors: Dict[str, np.ndarray] = {}
    for f in keys:
        X = embs[f]
        base_name = alias_map.get(f, os.path.basename(f))
        neigh_path = f"{f.rsplit('.npy', 1)[0]}.{args.suffix}.k{args.k}.npy"

        # 1) 캐시가 있으면 로드
        if os.path.exists(neigh_path):
            neigh = np.load(neigh_path)
            # 안전하게 shape 체크 (item 수나 k가 바뀐 경우 다시 계산)
            if neigh.shape[0] != N or neigh.shape[1] != args.k:
                print(f"[NEIGHBORS] Cached shape mismatch for {base_name}, recomputing ...")
                neigh = topk_neighbors_blockwise(X, k=args.k, block_size=args.block_size)
                np.save(neigh_path, neigh)
            else:
                print(f"[NEIGHBORS] Loaded cached Top-{args.k} for {base_name} from {neigh_path}")
        else:
            # 2) 없으면 새로 계산
            print(f"[NEIGHBORS] Computing Top-{args.k} for {base_name} ...")
            neigh = topk_neighbors_blockwise(X, k=args.k, block_size=args.block_size)
            # --save_neighbors 옵션과 상관없이 항상 저장하고 싶으면 if 제거
            if args.save_neighbors:
                np.save(neigh_path, neigh)
                print(f"  -> saved neighbors: {neigh_path}")

        neighbors[f] = neigh

    # Overlap matrix (pairwise)
    # alias 중복 방지: 동일 이름이 있으면 뒤에 (2), (3) 붙이기
    seen = {}
    short_names: List[str] = []
    for k in keys:
        base_alias = alias_map.get(k, os.path.basename(k))
        name = base_alias
        if name in seen:
            seen[name] += 1
            name = f"{name}({seen[name]})"
        else:
            seen[name] = 1
        short_names.append(name)

    m = len(keys)
    overlap_mat = np.zeros((m, m), dtype=np.float32)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                overlap_mat[i, j] = 1.0
            else:
                _, mean_ov = pairwise_overlap_at_k(neighbors[a], neighbors[b], args.k)
                overlap_mat[i, j] = mean_ov

    # Console table (평문 alias)
    print("\n=== Pairwise Overlap@{} (mean over items) ===".format(args.k))
    header = [""] + short_names
    widths = [max(12, max(len(h) for h in header))]
    for h in header[1:]:
        widths.append(max(12, len(h)))
    fmt_row = "  ".join("{:<" + str(w) + "}" for w in widths)

    print(fmt_row.format(*header))
    for i, name in enumerate(short_names):
        row = [name] + [("--" if i == j else f"{overlap_mat[i,j]:.4f}") for j in range(m)]
        print(fmt_row.format(*row))

    # Save CSV (pairwise, 평문 alias)
    csv_path = os.path.join(img_dir, f"overlap_k{args.k}_{variant_suffix}.csv")
    with open(csv_path, "w") as f:
        f.write("," + ",".join(short_names) + "\n")
        for i, name in enumerate(short_names):
            f.write(name + "," + ",".join(f"{overlap_mat[i,j]:.6f}" for j in range(m)) + "\n")
    print(f"\n[SAVE] CSV -> {csv_path}")

    # Full heatmap (pairwise) — 축 라벨만 TeX 수식 변환
    fig, ax = plt.subplots(figsize=(1.2 + 0.6*m, 1.2 + 0.6*m))
    im = ax.imshow(overlap_mat, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(m)); ax.set_yticks(range(m))

    display_labels = [pretty_label(alias) for alias in short_names]
    ax.set_xticks(range(m))
    ax.set_xticklabels(display_labels, ha="right", fontsize=14)

    # 개별 tick label 객체 가져오기
    for label, alias in zip(ax.get_xticklabels(), short_names):
        if alias == "final_emb":
            label.set_rotation(0)
            label.set_ha("right")
        else:
            label.set_rotation(0)
            label.set_ha("center") 

    ax.set_yticklabels(display_labels, fontsize=14)
    for label in ax.get_yticklabels():
        label.set_ha("right")

    # ax.set_title(f"Overlap@{args.k} (mean over items) - {variant_suffix}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if m <= 20:
        for i in range(m):
            for j in range(m):
                ax.text(j, i, ("1.00" if i == j else f"{overlap_mat[i,j]:.2f}"),
                        ha="center", va="center", fontsize=8)
    plt.tight_layout()
    img_path = os.path.join(img_dir, f"overlap_k{args.k}_{variant_suffix}_final.png")
    plt.savefig(img_path, dpi=200); plt.close(fig)
    print(f"[SAVE] Heatmap -> {img_path}")

    # ------------------ n-way overlap (optional) ------------------
    if args.multi:
        # alias 이름 → 파일 path
        alias_to_key = {alias_map.get(k, os.path.basename(k)): k for k in keys}
        missing = [a for a in args.multi if a not in alias_to_key]
        if missing:
            print(f"[WARN] --multi aliases not found: {missing}")
        chosen = [neighbors[alias_to_key[a]] for a in args.multi if a in alias_to_key]
        if len(chosen) >= 2:
            per_item, mean_val = multi_overlap_at_k(chosen, args.k)
            print(f"\n=== {len(chosen)}-way Overlap@{args.k} for {args.multi} ===")
            print(f"Mean overlap = {mean_val:.6f}")

            # save summary
            txt_path = os.path.join(
                img_dir,
                f"multi_overlap_k{args.k}_{variant_suffix}__{'__'.join(args.multi)}.txt"
            )
            with open(txt_path, "w") as f:
                f.write(f"aliases: {args.multi}\n")
                f.write(f"mean_overlap@{args.k}: {mean_val:.6f}\n")
            print(f"[SAVE] Multi-way summary -> {txt_path}")
        else:
            print("[WARN] Need >=2 valid aliases in --multi to compute n-way overlap.")

    print()


if __name__ == "__main__":
    main()