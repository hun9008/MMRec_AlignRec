import os
import argparse
import numpy as np


def read_interactions_csv(path_csv):
    assert os.path.exists(path_csv), f"missing interactions: {path_csv}"
    with open(path_csv, "r", encoding="utf-8") as f:
        head = f.readline()
    sep = "\t" if "\t" in head else ","

    # lazy import to keep deps minimal
    import pandas as pd
    df = pd.read_csv(path_csv, sep=sep, low_memory=False)

    cols = {c.lower(): c for c in df.columns}

    def pick(name_opts):
        for opt in name_opts:
            for c in cols:
                import re
                if re.fullmatch(opt, c):
                    return cols[c]
        return None

    user_col = pick([r"user[_\s]*id", r"user", r"uid"])
    item_col = pick([r"item[_\s]*id", r"item", r"iid", r"product[_\s]*id"])
    if user_col is None or item_col is None:
        raise ValueError(f"Cannot detect user/item columns. columns={list(df.columns)}")

    df = df[[user_col, item_col]].dropna()
    df[user_col] = df[user_col].astype(int)
    df[item_col] = df[item_col].astype(int)

    item2users = {}
    user2items = {}
    for u, it in df.itertuples(index=False):
        item2users.setdefault(it, set()).add(u)
        user2items.setdefault(u, set()).add(it)

    return item2users, user2items


def twohop_item_count(item_id, item2users, user2items):
    users = item2users.get(item_id)
    if not users:
        return 0
    neigh_items = set()
    for u in users:
        neigh_items.update(user2items.get(u, ()))
    neigh_items.discard(item_id)
    return len(neigh_items)


def l2_normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def topk_by_cosine(emb, anchor_idx, k):
    emb_n = l2_normalize(emb)
    anchor = emb_n[anchor_idx]
    sims = emb_n @ anchor
    sims[anchor_idx] = -np.inf
    if k >= len(sims):
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
    return top_idx, sims


def compute_stats(name, final_emb, text_feat, vision_feat, item_id, k, twohop):
    top_idx, sims = topk_by_cosine(final_emb, item_id, k)
    text_n = l2_normalize(text_feat)
    vision_n = l2_normalize(vision_feat)
    anchor_text = text_n[item_id]
    anchor_vision = vision_n[item_id]

    rows = []
    for rank, it in enumerate(top_idx, 1):
        twohop_cnt = twohop_item_count(it, *twohop) if twohop is not None else None
        text_sim = float(np.dot(text_n[it], anchor_text))
        vision_sim = float(np.dot(vision_n[it], anchor_vision))
        rows.append((rank, int(it), float(sims[it]), twohop_cnt, vision_sim, text_sim))

    # averages
    avg_twohop = None
    if twohop is not None:
        vals = [r[3] for r in rows]
        avg_twohop = float(np.mean(vals)) if vals else 0.0
    avg_vision = float(np.mean([r[4] for r in rows])) if rows else 0.0
    avg_text = float(np.mean([r[5] for r in rows])) if rows else 0.0

    print(f"\n[{name}] top-{k} for item {item_id}")
    print("rank\titem\tfinal_cos\t2hop\tvision_cos\ttext_cos")
    for r in rows:
        twohop_str = str(r[3]) if r[3] is not None else "-"
        print(f"{r[0]}\t{r[1]}\t{r[2]:.4f}\t{twohop_str}\t{r[4]:.4f}\t{r[5]:.4f}")
    print("avg\t-\t-\t{}\t{:.4f}\t{:.4f}".format(
        f"{avg_twohop:.2f}" if avg_twohop is not None else "-",
        avg_vision,
        avg_text,
    ))

    # save .txt
    out_path = f"topk_{name.lower()}_item{item_id}_k{k}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Top-{k} similar items for item {item_id} by {name}\n")
        f.write("rank\titem\tfinal_cos\t2hop\tvision_cos\ttext_cos\n")
        for r in rows:
            twohop_str = str(r[3]) if r[3] is not None else "-"
            f.write(f"{r[0]}\t{r[1]}\t{r[2]:.4f}\t{twohop_str}\t{r[4]:.4f}\t{r[5]:.4f}\n")
        f.write("avg\t-\t-\t{}\t{:.4f}\t{:.4f}\n".format(
            f"{avg_twohop:.2f}" if avg_twohop is not None else "-",
            avg_vision,
            avg_text,
        ))
    print(f"Saved to {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_id", type=int, default=5016)
    parser.add_argument("--topk", type=int, default=500)

    parser.add_argument("--alignrec_final", default="saved_emb/item_emb_alignrec_final.npy")
    parser.add_argument("--anchorrec_final", default="saved_emb/item_emb_alignrec_anchor_final.npy")
    parser.add_argument("--text_feat", default="saved_emb/alignrec_anchor_1101/item_feat_raw_text.npy")
    parser.add_argument("--vision_feat", default="saved_emb/alignrec_anchor_1101/item_feat_raw_vision.npy")
    parser.add_argument("--interactions", default="data/baby/baby.inter")
    parser.add_argument("--no_interactions", action="store_true")
    args = parser.parse_args()

    for p in [args.alignrec_final, args.anchorrec_final, args.text_feat, args.vision_feat]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    align_final = np.load(args.alignrec_final)
    anchor_final = np.load(args.anchorrec_final)
    text_feat = np.load(args.text_feat)
    vision_feat = np.load(args.vision_feat)

    if args.item_id < 0 or args.item_id >= align_final.shape[0]:
        raise ValueError(f"item_id out of range: 0..{align_final.shape[0]-1}")

    if align_final.shape[0] != anchor_final.shape[0]:
        raise ValueError("alignrec_final and anchorrec_final must have same #items")
    if text_feat.shape[0] != align_final.shape[0] or vision_feat.shape[0] != align_final.shape[0]:
        raise ValueError("text/vision feature count must match final embeddings")

    twohop = None
    if not args.no_interactions and os.path.exists(args.interactions):
        item2users, user2items = read_interactions_csv(args.interactions)
        twohop = (item2users, user2items)

    compute_stats("AlignRec", align_final, text_feat, vision_feat, args.item_id, args.topk, twohop)
    compute_stats("AnchorRec", anchor_final, text_feat, vision_feat, args.item_id, args.topk, twohop)


if __name__ == "__main__":
    main()
