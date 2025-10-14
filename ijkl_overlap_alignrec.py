# scripts/check_overlap_projection_final.py
import os
import numpy as np
import torch
import csv

def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-9)

@torch.no_grad()
def split_neighbors(E: torch.Tensor, k_close: int = 10, k_far: int = 10):
    n = E.size(0)
    S = E @ E.T  # cosine

    # close: self 불가
    S_close = S.clone()
    S_close.fill_diagonal_(-1e9)
    _, close_idx = torch.topk(S_close, k=k_close, dim=1, largest=True, sorted=False)

    # far: self 불가
    S_far = S.clone()
    S_far.fill_diagonal_(+1e9)
    _, far_idx = torch.topk(S_far, k=k_far, dim=1, largest=False, sorted=False)

    middle_mask = torch.ones((n, n), dtype=torch.bool, device=E.device)
    middle_mask.fill_diagonal_(False)
    middle_mask.scatter_(1, close_idx, False)
    middle_mask.scatter_(1, far_idx,   False)
    return close_idx, far_idx, middle_mask

@torch.no_grad()
def category_overlap(Ep: torch.Tensor,
                     Ef: torch.Tensor,
                     k_close: int = 10,
                     k_far: int = 10):
    Ep = normalize_rows(Ep)
    Ef = normalize_rows(Ef)

    p_close, p_far, p_middle_mask = split_neighbors(Ep, k_close, k_far)
    f_close, f_far, f_middle_mask = split_neighbors(Ef, k_close, k_far)

    n = Ep.size(0)
    p_close_mask = torch.zeros((n, n), dtype=torch.bool, device=Ep.device)
    p_far_mask   = torch.zeros((n, n), dtype=torch.bool, device=Ep.device)
    f_close_mask = torch.zeros((n, n), dtype=torch.bool, device=Ep.device)
    f_far_mask   = torch.zeros((n, n), dtype=torch.bool, device=Ef.device)

    p_close_mask.scatter_(1, p_close, True)
    p_far_mask.scatter_(1, p_far, True)
    f_close_mask.scatter_(1, f_close, True)
    f_far_mask.scatter_(1, f_far, True)

    has_close_overlap  = (p_close_mask  & f_close_mask).any(dim=1)
    has_far_overlap    = (p_far_mask    & f_far_mask).any(dim=1)
    has_middle_overlap = (p_middle_mask & f_middle_mask).any(dim=1)

    mask_all_true = has_close_overlap & has_far_overlap & has_middle_overlap

    stats = {
        "count_all_true": int(mask_all_true.sum().item()),
        "ratio_all_true": float(mask_all_true.float().mean().item()),
        "ratio_close_overlap": float(has_close_overlap.float().mean().item()),
        "ratio_far_overlap": float(has_far_overlap.float().mean().item()),
        "ratio_middle_overlap": float(has_middle_overlap.float().mean().item()),
    }
    aux = {
        "p_close": p_close, "p_far": p_far, "p_middle_mask": p_middle_mask,
        "f_close": f_close, "f_far": f_far, "f_middle_mask": f_middle_mask,
        "has_close_overlap": has_close_overlap,
        "has_far_overlap": has_far_overlap,
        "has_middle_overlap": has_middle_overlap,
    }
    return mask_all_true, stats, aux

# @torch.no_grad()
# def export_ijkl_csv(save_csv_path: str,
#                     mask_all_true: torch.Tensor,
#                     aux: dict,
#                     Ef: torch.Tensor):
#     """
#     (i, j, l, k)를 CSV로 저장.
#       - j: (p_close ∩ f_close) 중 Ef에서 i와 코사인 유사도 최대
#       - k: (p_far   ∩ f_far)   중 Ef에서 i와 코사인 유사도 최소
#       - l: (p_middle ∩ f_middle) 전체 후보를 모두 저장
#     """
#     os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

#     Ef = normalize_rows(Ef)  # (n, d)
#     n = Ef.size(0)

#     p_close = aux["p_close"]
#     f_close = aux["f_close"]
#     p_far   = aux["p_far"]
#     f_far   = aux["f_far"]
#     p_mid_m = aux["p_middle_mask"]
#     f_mid_m = aux["f_middle_mask"]

#     def idxs_to_mask(idxs, N):
#         m = torch.zeros((N, N), dtype=torch.bool, device=idxs.device)
#         m.scatter_(1, idxs, True)
#         return m

#     p_close_mask = idxs_to_mask(p_close, n)
#     f_close_mask = idxs_to_mask(f_close, n)
#     p_far_mask   = idxs_to_mask(p_far,   n)
#     f_far_mask   = idxs_to_mask(f_far,   n)

#     both_close = (p_close_mask & f_close_mask)
#     both_far   = (p_far_mask   & f_far_mask)
#     both_mid   = (p_mid_m      & f_mid_m)

#     rows = []
#     close_set = set()
#     far_set   = set()

#     true_idxs = mask_all_true.nonzero(as_tuple=False).flatten().tolist()
#     for i in true_idxs:
#         sim_row = (Ef[i] @ Ef.T)  # (n,)

#         # j: close 교집합 중 Ef에서 i와 코사인 유사도 최대
#         close_candidates = both_close[i].nonzero(as_tuple=False).flatten()
#         if len(close_candidates) > 0:
#             j = int(close_candidates[sim_row[close_candidates].argmax().item()])
#         else:
#             j = -1

#         # k: far 교집합 중 Ef에서 i와 코사인 유사도 최소
#         far_candidates = both_far[i].nonzero(as_tuple=False).flatten()
#         if len(far_candidates) > 0:
#             k = int(far_candidates[sim_row[far_candidates].argmin().item()])
#         else:
#             k = -1

#         # l: middle 교집합 중 모든 후보
#         mid_candidates = both_mid[i].nonzero(as_tuple=False).flatten().tolist()
#         if len(mid_candidates) == 0:
#             rows.append((i, j, -1, k))
#         else:
#             for l in mid_candidates:
#                 rows.append((i, j, int(l), k))

#         if j >= 0:
#             close_set.add(j)
#         if k >= 0:
#             far_set.add(k)

#     # CSV 저장
#     with open(save_csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["i", "j(close)", "l(middle)", "k(far)"])
#         writer.writerows(rows)

#     # 세트 저장
#     out_dir = os.path.dirname(save_csv_path)
#     close_txt = os.path.join(out_dir, "close_item.txt")
#     far_txt   = os.path.join(out_dir, "far_item.txt")
#     with open(close_txt, "w") as f:
#         for x in sorted(close_set):
#             f.write(f"{x}\n")
#     with open(far_txt, "w") as f:
#         for x in sorted(far_set):
#             f.write(f"{x}\n")

#     print(f"Saved ijkl csv: {save_csv_path} (rows: {len(rows)})")
#     print(f"Saved close set to {close_txt} (|close|={len(close_set)})")
#     print(f"Saved far   set to {far_txt}   (|far|={len(far_set)})")
@torch.no_grad()
def export_ijkl_csv(save_csv_path: str,
                    mask_all_true: torch.Tensor,
                    aux: dict,
                    Ef: torch.Tensor,
                    top_close: int = 5,
                    top_far: int = 5):
    """
    CSV 컬럼: i, j(close), l(middle), k(far)

    저장 정책 (행 단위, 폭발 방지):
      1) CLOSE 블록: (i, j, -1, -1)  for j in top-`top_close` (교집합 후보 중 Ef 유사도 내림차순)
      2) MIDDLE 블록: (i, -1, l, -1) for l in 모든 교집합 후보
      3) FAR   블록: (i, -1, -1, k)  for k in top-`top_far` (교집합 후보 중 Ef 유사도 오름차순)
    """
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

    Ef = normalize_rows(Ef)  # (n, d)
    n = Ef.size(0)

    p_close = aux["p_close"]
    f_close = aux["f_close"]
    p_far   = aux["p_far"]
    f_far   = aux["f_far"]
    p_mid_m = aux["p_middle_mask"]
    f_mid_m = aux["f_middle_mask"]

    def idxs_to_mask(idxs, N):
        m = torch.zeros((N, N), dtype=torch.bool, device=idxs.device)
        m.scatter_(1, idxs, True)
        return m

    p_close_mask = idxs_to_mask(p_close, n)
    f_close_mask = idxs_to_mask(f_close, n)
    p_far_mask   = idxs_to_mask(p_far,   n)
    f_far_mask   = idxs_to_mask(f_far,   n)

    both_close = (p_close_mask & f_close_mask)
    both_far   = (p_far_mask   & f_far_mask)
    both_mid   = (p_mid_m      & f_mid_m)

    rows = []
    close_set = set()
    far_set   = set()

    true_idxs = mask_all_true.nonzero(as_tuple=False).flatten().tolist()
    for i in true_idxs:
        # i와의 유사도 벡터 (Ef 기준)
        sim_row = (Ef[i] @ Ef.T)  # (n,)

        # ---- CLOSE: top_close개 (유사도 내림차순) ----
        close_candidates = both_close[i].nonzero(as_tuple=False).flatten()
        if close_candidates.numel() > 0:
            # 값이 큰 게 가까움 → 내림차순 정렬
            sims = sim_row[close_candidates]
            topc = min(top_close, close_candidates.numel())
            # torch.topk는 기본 내림차순 largest=True
            vals, idxs = torch.topk(sims, k=topc, largest=True, sorted=True)
            top_close_ids = close_candidates[idxs].tolist()
            for j in top_close_ids:
                rows.append((i, int(j), -1, -1))
                close_set.add(int(j))

        # ---- MIDDLE: 전부 ----
        mid_candidates = both_mid[i].nonzero(as_tuple=False).flatten()
        if mid_candidates.numel() > 0:
            for l in mid_candidates.tolist():
                rows.append((i, -1, int(l), -1))

        # ---- FAR: top_far개 (유사도 오름차순) ----
        far_candidates = both_far[i].nonzero(as_tuple=False).flatten()
        if far_candidates.numel() > 0:
            sims = sim_row[far_candidates]
            topf = min(top_far, far_candidates.numel())
            # 가장 먼 것 → 유사도 오름차순. topk로 하면 largest=False가 없으니 argsort 사용
            sorted_idx = torch.argsort(sims, dim=0, descending=False)  # 작은 값부터
            top_far_ids = far_candidates[sorted_idx[:topf]].tolist()
            for k in top_far_ids:
                rows.append((i, -1, -1, int(k)))
                far_set.add(int(k))

    # CSV 저장
    with open(save_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j(close)", "l(middle)", "k(far)"])
        writer.writerows(rows)

    # 세트 저장
    out_dir = os.path.dirname(save_csv_path)
    close_txt = os.path.join(out_dir, "close_item.txt")
    far_txt   = os.path.join(out_dir, "far_item.txt")
    with open(close_txt, "w") as f:
        for x in sorted(close_set):
            f.write(f"{x}\n")
    with open(far_txt, "w") as f:
        for x in sorted(far_set):
            f.write(f"{x}\n")

    print(f"Saved ijkl csv: {save_csv_path} (rows: {len(rows)})")
    print(f"Saved close set to {close_txt} (|close|={len(close_set)})")
    print(f"Saved far   set to {far_txt}   (|far|={len(far_set)})")

def main(
    save_dir="saved_emb/epoch_last",
    k_close=10, k_far=10,
    SAVE_MASKS=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    ef_path  = os.path.join(save_dir, "item_emb_final_alignrec.npy")

    assert os.path.exists(ef_path), f"missing: {ef_path}"

    Ef     = torch.tensor(np.load(ef_path), dtype=torch.float32, device=device)

    # ----- 겹침 계산 -----
    mask_all_true, stats, aux = category_overlap(Ef, Ef, k_close=k_close, k_far=k_far)
    n_items = Ef.size(0)
    print("\n=== Overlap stats (projection vs final) ===")
    print(f"items with all-3 overlaps True: {stats['count_all_true']} / {n_items} "
          f"({stats['ratio_all_true']*100:.2f}%)")
    print(f"  close-overlap ratio : {stats['ratio_close_overlap']*100:.2f}%")
    print(f"  far-overlap ratio   : {stats['ratio_far_overlap']*100:.2f}%")
    print(f"  middle-overlap ratio: {stats['ratio_middle_overlap']*100:.2f}%")

    # ----- CSV 내보내기 -----
    export_ijkl_csv(
        save_csv_path=os.path.join("./ijkl_overlap_alignrec", "result.csv"),
        mask_all_true=mask_all_true,
        aux=aux,
        Ef=Ef,
    )

    # ----- (선택) 마스크/이웃 저장 -----
    if SAVE_MASKS:
        np.save(os.path.join(save_dir, "proj_close_idx.npy"), aux["p_close"].detach().cpu().numpy())
        np.save(os.path.join(save_dir, "proj_far_idx.npy"),   aux["p_far"].detach().cpu().numpy())
        np.save(os.path.join(save_dir, "final_close_idx.npy"), aux["f_close"].detach().cpu().numpy())
        np.save(os.path.join(save_dir, "final_far_idx.npy"),   aux["f_far"].detach().cpu().numpy())
        print("Saved close/far indices to .npy")

if __name__ == "__main__":
    main(
        save_dir="saved_emb/0927_all_alignrec",
        k_close=10, k_far=10,
        SAVE_MASKS=False,
    )