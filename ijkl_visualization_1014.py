import os, json, html, argparse, time, logging, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Optional, Tuple, Set, Dict, List

# ---------- optional deps ----------
_HAS_TORCH = False
_HAS_TORCHVISION = False
_HAS_TRANSFORMERS = False
_HAS_SENTENCE_T = False
_HAS_PIL_REQ = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    pass

try:
    import torchvision
    from torchvision import models
    _HAS_TORCHVISION = True
except Exception:
    pass

try:
    from PIL import Image
    import requests
    _HAS_PIL_REQ = True
except Exception:
    pass

try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_T = True
except Exception:
    pass

RESULT_CSV    = "./ijkl_overlap_1104/result.csv"
MAPPING_CSV   = "./data/baby/i_id_mapping.csv"
METADATA_JSON = "./data/baby/metadata_baby.json"
INTER_PATH    = "./data/baby/baby.inter"   # TSV/CSV 자동 인식
CACHE_DIR     = "./cache_images"

# ---------------- logging & timer ----------------
def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

class tick:
    def __init__(self, msg: str):
        self.msg, self.t0 = msg, None
    def __enter__(self):
        logging.info(f"▶ {self.msg} 시작"); self.t0 = time.perf_counter(); return self
    def __exit__(self, et, ev, tb):
        dt = (time.perf_counter() - self.t0) * 1000
        if et is None: logging.info(f"✔ {self.msg} 완료 ({dt:.1f} ms)")
        else: logging.exception(f"✘ {self.msg} 실패 ({dt:.1f} ms) - {ev}")

# ---------------- utils ----------------
def read_mapping(path_csv: str) -> dict:
    with tick("i_id_mapping 로드"):
        with open(path_csv, "r", encoding="utf-8") as f:
            head = f.readline()
        delim = "\t" if "\t" in head else ","
        df = pd.read_csv(path_csv, sep=delim, usecols=["itemID","asin"],
                         dtype={"itemID":"int32","asin":"string"})
        mapping = dict(zip(df["itemID"].astype(int), df["asin"].astype(str)))
        logging.debug(f"mapping size={len(mapping)}")
        return mapping

def read_result_two_cols(path_csv: str) -> pd.DataFrame:
    with tick("result.csv(i, j(close)) 로드"):
        df = pd.read_csv(
            path_csv,
            usecols=["i","j(close)"],
            dtype={"i":"int32","j(close)":"int32"},
            low_memory=False
        )
        logging.debug(f"result two-cols shape={df.shape}")
        return df

def read_metadata_all(path_json: str) -> dict:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d.get("asin"): d for d in data if d.get("asin")}

def read_metadata_filtered(path_json: str, needed_asins: Set[str]) -> dict:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["asin"]: d for d in data if d.get("asin") in needed_asins}

def fetch_image(url: str, timeout: int = 7, log_images: bool = False, noimg: bool = False):
    if noimg or not (_HAS_PIL_REQ and url): return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = url.replace("://","_").replace("/","_").replace("?","_").replace("&","_").replace("=","_")
    local = os.path.join(CACHE_DIR, safe)
    try:
        if os.path.exists(local):
            if log_images: logging.debug(f"[IMG] cache hit: {local}")
            return Image.open(local).convert("RGB")
        if log_images: logging.debug(f"[IMG] downloading: {url}")
        r = requests.get(url, timeout=timeout); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(local); return img
    except Exception as e:
        if log_images: logging.warning(f"[IMG] failed: {url} ({e})")
        return None

def wrap_text(s: str, width: int = 60, max_lines: int = 6) -> str:
    if not s: return ""
    import textwrap
    s = html.unescape(str(s))
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines: lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)

def get_item_text_and_img(item_id: int, id2asin: dict, meta: dict,
                          log_images: bool = False, noimg: bool = False):
    asin = id2asin.get(int(item_id))
    if asin is None: return "(unknown)", "N/A", None
    m = meta.get(asin, {})
    title = m.get("title",""); desc = m.get("description","")
    text = f"{title}\n\n{desc}" if desc else title
    img  = fetch_image(m.get("imUrl",""), log_images=log_images, noimg=noimg)
    return asin, text, img

# ---------- interactions ----------
def read_interactions_csv(path_csv: str) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Returns:
      item2users: {item_id -> {user_ids}}
      user2items:{user_id -> {item_ids}}
    자동 컬럼 탐지: userID/itemID/user_id/item_id 등
    """
    with tick("interactions 로드"):
        assert os.path.exists(path_csv), f"missing interactions: {path_csv}"
        with open(path_csv, "r", encoding="utf-8") as f:
            head = f.readline()
        sep = "\t" if "\t" in head else ","
        df = pd.read_csv(path_csv, sep=sep, low_memory=False)

        cols = {c.lower(): c for c in df.columns}
        def pick(name_opts):
            for opt in name_opts:
                for c in cols:
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

        item2users: Dict[int, Set[int]] = {}
        user2items: Dict[int, Set[int]] = {}
        for u, it in df.itertuples(index=False):
            item2users.setdefault(it, set()).add(u)
            user2items.setdefault(u, set()).add(it)

        logging.info(f"interactions loaded: |users|={len(user2items)}, |items|={len(item2users)}")
        return item2users, user2items

def twohop_item_count(item_id: int, item2users: Dict[int, Set[int]], user2items: Dict[int, Set[int]]) -> int:
    users = item2users.get(item_id)
    if not users: return 0
    neigh_items: Set[int] = set()
    for u in users:
        neigh_items.update(user2items.get(u, ()))
    neigh_items.discard(item_id)
    return len(neigh_items)

# ------------- FAST close-index -------------
def build_close_index_fast(df_two_cols: pd.DataFrame, topk: int, dedup: str = "global") -> dict:
    with tick(f"close 인덱스 생성 (dedup={dedup})"):
        if dedup == "consecutive":
            same_i = df_two_cols["i"].eq(df_two_cols["i"].shift())
            same_j = df_two_cols["j(close)"].eq(df_two_cols["j(close)"].shift())
            mask   = ~(same_i & same_j)
            df = df_two_cols[mask].copy()
        else:
            df = df_two_cols.drop_duplicates(subset=["i","j(close)"], keep="first").copy()

        # 유효 j만
        df = df[(df["j(close)"] >= 0) & (df["i"] != df["j(close)"])]

        df["rk"] = df.groupby("i").cumcount()
        df = df[df["rk"] < topk]

        agg = df.groupby("i", sort=False)["j(close)"].apply(list)
        i2js = {int(i): lst for i, lst in agg.items()}

        # 순서 유지 중복 제거
        for i, lst in list(i2js.items()):
            seen, deduped = set(), []
            for x in lst:
                if x not in seen:
                    seen.add(x); deduped.append(x)
            i2js[i] = deduped[:topk]

        logging.debug(f"anchors: {len(i2js)} (valid-only, topk-truncated)")
        return i2js

def choose_anchor_with_fallback(
    i2js: dict, want_i: Optional[int], need_k: int, rng: np.random.Generator
) -> Tuple[int, int]:
    with tick("anchor 선택(자동 대체/축소 포함)"):
        counts = sorted(((i, len(js)) for i, js in i2js.items()),
                        key=lambda x: x[1], reverse=True)
        logging.info("i별 close 개수 TOP5: " + ", ".join([f"{i}:{c}" for i, c in counts[:5]]))
        candidates = [i for i, js in i2js.items() if len(js) >= need_k]

        if want_i is not None and want_i in i2js and len(i2js[want_i]) >= need_k:
            logging.info(f"anchor 고정: i={want_i}, k={need_k}")
            return want_i, need_k

        if candidates:
            picked = int(rng.choice(candidates))
            if want_i is not None:
                logging.warning(f"지정 anchor i={want_i}는 j<{need_k} → 대체 i={picked}, k={need_k}")
            else:
                logging.info(f"anchor 무작위 선택: i={picked}, k={need_k}")
            return picked, need_k

        if not counts or counts[0][1] == 0:
            raise ValueError("모든 i에 대해 close j가 0개입니다. result.csv 내용을 확인하세요.")
        best_i, best_c = counts[0]
        eff_k = min(need_k, best_c)
        logging.warning(f"유효 j가 {need_k}개 이상인 i가 없습니다 → i={best_i}, 가용 {best_c}개로 k={eff_k}로 축소")
        return best_i, eff_k

# ---------- embedding helpers ----------
_IMG_MODEL = None
_IMG_TRANS = None
_TXT_MODEL = None
_TXT_TOK   = None
_SBERT     = None

def setup_image_model(device:str):
    """ResNet50 (ImageNet), features vector"""
    global _IMG_MODEL, _IMG_TRANS
    if not (_HAS_TORCH and _HAS_TORCHVISION): return False
    with tick("ResNet50 로드"):
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        _IMG_MODEL = model
        _IMG_TRANS = weights.transforms()
    return True

def embed_image(img_pil, device:str) -> Optional[np.ndarray]:
    if _IMG_MODEL is None or _IMG_TRANS is None or not _HAS_TORCH: return None
    x = _IMG_TRANS(img_pil).unsqueeze(0)  # CPU tensor
    if device != "cpu":
        x = x.to(device)
    with torch.no_grad():
        x = _IMG_MODEL(x)
        if x.dim() == 4:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1)).squeeze()
        v = x.flatten().float()
        v = v / (v.norm() + 1e-9)
    return v.detach().cpu().numpy()

def setup_text_model_bert(device:str, model_name:str="bert-base-uncased"):
    global _TXT_MODEL, _TXT_TOK
    if not (_HAS_TORCH and _HAS_TRANSFORMERS): return False
    with tick(f"BERT 로드 ({model_name})"):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval().to(device)
        _TXT_MODEL, _TXT_TOK = mdl, tok
    return True

def embed_text_bert(texts: List[str], device:str, max_len:int=128) -> Optional[np.ndarray]:
    if _TXT_MODEL is None or _TXT_TOK is None or not _HAS_TORCH: return None
    # mean pooling on last hidden states
    from torch.nn.functional import normalize
    embs = []
    with torch.no_grad():
        for t in texts:
            enc = _TXT_TOK(t if t else "", return_tensors="pt", truncation=True, max_length=max_len)
            if device != "cpu":
                for k in enc: enc[k] = enc[k].to(device)
            out = _TXT_MODEL(**enc)
            last = out.last_hidden_state  # (1,L,H)
            mask = enc["attention_mask"].unsqueeze(-1)  # (1,L,1)
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1)
            v = (summed / denom).squeeze(0).float()
            v = normalize(v, dim=0)
            embs.append(v.detach().cpu().numpy())
    return np.stack(embs, axis=0) if embs else None

def setup_text_model_sbert(device:str, model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
    global _SBERT
    if not (_HAS_TORCH and _HAS_SENTENCE_T): return False
    with tick(f"SBERT 로드 ({model_name})"):
        mdl = SentenceTransformer(model_name, device=device if device != "cpu" else "cpu")
        _SBERT = mdl
    return True

def embed_text_sbert(texts: List[str]) -> Optional[np.ndarray]:
    if _SBERT is None: return None
    # SentenceTransformer는 내부에서 알아서 디바이스 사용
    embs = _SBERT.encode(texts, batch_size=8, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embs

def cosine(a:np.ndarray, b:np.ndarray) -> Optional[float]:
    if a is None or b is None: return None
    return float(np.dot(a, b))

# ------------- viz -------------
def visualize_anchor_with_topk_close(anchor_i: int, js_topk: list, id2asin: dict, meta: dict,
                                     out_png: str, log_images: bool = False, noimg: bool = False,
                                     twohop_counts: Optional[Dict[int, int]] = None,
                                     img_sims: Optional[Dict[int,float]] = None,
                                     txt_sims: Optional[Dict[int,float]] = None):
    with tick("시각화 생성"):
        k = len(js_topk)
        fig, axes = plt.subplots(nrows=1, ncols=1 + k, figsize=(4*(1+k), 6))
        if (1 + k) == 1:
            import numpy as _np
            axes = _np.array([axes])

        titles = [f"i (anchor)\nID: {anchor_i}"] + [f"j#{r} (close)\nID: {jid}" for r, jid in enumerate(js_topk, start=1)]
        item_ids = [anchor_i] + js_topk

        for c, item_id in enumerate(item_ids):
            ax = axes[c]; ax.axis("off")
            asin, text, img = get_item_text_and_img(item_id, id2asin, meta, log_images=log_images, noimg=noimg)
            if img is not None: ax.imshow(img)

            # 부가 정보(2-hop / cos)
            extras = []
            if twohop_counts is not None and item_id in twohop_counts:
                extras.append(f"2-hop: {twohop_counts[item_id]}")
            if img_sims is not None:
                val = img_sims.get(item_id, None)
                extras.append(f"img cos: {'-' if val is None else f'{val:.3f}'}")
            if txt_sims is not None:
                val = txt_sims.get(item_id, None)
                extras.append(f"txt cos: {'-' if val is None else f'{val:.3f}'}")
            extra_txt = ("\n" + "\n".join(extras)) if extras else ""

            ax.set_title(f"{titles[c]}\nASIN: {asin}{extra_txt}", fontsize=10)

            if text:
                ax.text(
                    0.02, 0.02,  # 아래쪽 고정
                    wrap_text(text, width=70, max_lines=6),
                    transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
                )

            if twohop_counts is not None and item_id in twohop_counts:
                ax.text(
                    0.02, 0.98, f"2-hop {twohop_counts[item_id]}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8)
                )

        plt.tight_layout()
        with tick(f"이미지 저장: {out_png}"):
            plt.savefig(out_png, dpi=200)
        plt.close(fig)

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="Visualize top-k close neighbors (ijkl_overlap) with 2-hop + image/text cosine.")
    ap.add_argument("--anchor", type=int, default=9, help="고정할 anchor i (부족하면 자동 대체/축소)")
    ap.add_argument("--topk", type=int, default=5, help="가까운 이웃 j 개수")
    ap.add_argument("--out", type=str, default="ijkl_visualization_topkclose", help="출력 파일명")
    ap.add_argument("--dedup", type=str, default="global", choices=["global","consecutive"],
                    help="중복 제거 방식: global=전역 중복 제거, consecutive=연속 중복 제거")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--log-images", action="store_true")
    ap.add_argument("--noimg", action="store_true")
    ap.add_argument("--filter-meta", action="store_true")
    ap.add_argument("--no-interactions", action="store_true", help="2-hop 계산 끄기")

    # 새 옵션들
    default_device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    ap.add_argument("--device", type=str, default=default_device)
    ap.add_argument("--no-img-sim", action="store_true", help="ResNet 임베딩 코사인 계산 생략")
    ap.add_argument("--no-txt-sim", action="store_true", help="텍스트 임베딩 코사인 계산 생략")
    ap.add_argument("--sbert-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="문장 임베딩 모델(SBERT). sentence-transformers 없으면 BERT로 폴백.")
    ap.add_argument("--bert-model", type=str, default="bert-base-uncased", help="폴백 BERT 모델")
    args = ap.parse_args()

    setup_logger(args.log_level)

    with tick("입력 파일 존재 확인"):
        assert os.path.exists(RESULT_CSV), f"missing {RESULT_CSV}"
        assert os.path.exists(MAPPING_CSV), f"missing {MAPPING_CSV}"
        assert os.path.exists(METADATA_JSON), f"missing {METADATA_JSON}"

    id2asin = read_mapping(MAPPING_CSV)
    df_two = read_result_two_cols(RESULT_CSV)
    i2js   = build_close_index_fast(df_two, topk=args.topk, dedup=args.dedup)

    rng_seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(int(rng_seed) % (2**32 - 1))
    anchor_i, eff_k = choose_anchor_with_fallback(i2js, want_i=args.anchor, need_k=args.topk, rng=rng)

    js_topk  = i2js[anchor_i][:eff_k]
    logging.info(f"[INFO] anchor i={anchor_i}, close j(top-{eff_k})={js_topk}")

    # interactions → 2-hop
    twohop_counts = None
    if not args.no_interactions and os.path.exists(INTER_PATH):
        item2users, user2items = read_interactions_csv(INTER_PATH)
        targets = [anchor_i] + js_topk
        with tick("2-hop 이웃 개수 계산"):
            twohop_counts = {it: twohop_item_count(it, item2users, user2items) for it in targets}
        covered = sum(1 for it in targets if it in item2users)
        logging.info("2-hop counts: " + ", ".join([f"{it}:{twohop_counts[it]}" for it in targets]))
        logging.info(f"coverage: {covered}/{len(targets)} targets appear in interactions")
        missing = [it for it in targets if it not in item2users]
        if missing:
            logging.warning(f"missing-in-interactions: {missing}")
    else:
        if not os.path.exists(INTER_PATH):
            logging.warning(f"interactions 파일이 없어 2-hop 계산 생략: {INTER_PATH}")
        else:
            logging.info("요청에 의해 2-hop 계산 생략 (--no-interactions)")

    # 메타데이터
    if args.filter_meta:
        with tick("필요 asin 집합 구성"):
            ids = [anchor_i] + js_topk
            needed = {id2asin.get(int(x)) for x in ids if id2asin.get(int(x))}
        with tick("metadata 로드(필터)"):
            meta = read_metadata_filtered(METADATA_JSON, needed)
    else:
        with tick("metadata 로드(전체)"):
            meta = read_metadata_all(METADATA_JSON)

    # ---- 임베딩 코사인 계산 (anchor vs js) ----
    img_sims: Dict[int, float] = {}
    txt_sims: Dict[int, float] = {}

    # anchor의 원재료
    _, anchor_text, anchor_img = get_item_text_and_img(anchor_i, id2asin, meta,
                                                       log_images=args.log_images, noimg=args.noimg)

    # 이미지 코사인 (ResNet50)
    if not args.no_img_sim:
        if not (_HAS_TORCH and _HAS_TORCHVISION):
            logging.warning("torch/torchvision 미탑재 → img cos 생략")
        else:
            ok = setup_image_model(args.device)
            if ok and anchor_img is not None:
                with tick("이미지 임베딩 계산"):
                    va = embed_image(anchor_img, args.device)
                    for j in js_topk:
                        _, _, j_img = get_item_text_and_img(j, id2asin, meta,
                                                            log_images=args.log_images, noimg=args.noimg)
                        vj = embed_image(j_img, args.device) if j_img is not None else None
                        img_sims[j] = cosine(va, vj) if (va is not None and vj is not None) else None
            else:
                logging.warning("ResNet 초기화 실패 또는 anchor 이미지 없음 → img cos 생략")

    # 텍스트 코사인 (SBERT 우선, 없으면 BERT 폴백)
    if not args.no_txt_sim:
        texts_anchor = [anchor_text or ""]
        texts_js = []
        for j in js_topk:
            _, j_text, _ = get_item_text_and_img(j, id2asin, meta, log_images=False, noimg=True)
            texts_js.append(j_text or "")

        va = None
        vjs = None

        # SBERT 시도
        if _HAS_TORCH and _HAS_SENTENCE_T:
            ok = setup_text_model_sbert(args.device, model_name=args.sbert_model)
            if ok:
                with tick("텍스트 임베딩 계산(SBERT)"):
                    all_emb = embed_text_sbert(texts_anchor + texts_js)
                    if all_emb is not None and all_emb.shape[0] == (1 + len(js_topk)):
                        va = all_emb[0]
                        vjs = all_emb[1:]

        # 폴백 BERT
        if (va is None or vjs is None) and _HAS_TORCH and _HAS_TRANSFORMERS:
            ok = setup_text_model_bert(args.device, model_name=args.bert_model)
            if ok:
                with tick("텍스트 임베딩 계산(BERT 폴백)"):
                    va_b = embed_text_bert(texts_anchor, args.device)
                    vj_b = embed_text_bert(texts_js, args.device)
                    if va_b is not None and vj_b is not None:
                        va = va_b[0]
                        vjs = vj_b

        if va is None or vjs is None:
            logging.warning("텍스트 임베딩 불가 → txt cos 생략")
        else:
            for j, v in zip(js_topk, vjs):
                txt_sims[j] = cosine(va, v)

    # anchor 칸은 표기용으로 '-'
    img_sims[anchor_i] = None
    txt_sims[anchor_i] = None

    visualize_anchor_with_topk_close(
        anchor_i, js_topk, id2asin, meta, "1104" + args.out + f"{anchor_i}.png",
        log_images=args.log_images, noimg=args.noimg,
        twohop_counts=twohop_counts,
        img_sims=img_sims, txt_sims=txt_sims
    )
    logging.info(f"[DONE] saved to: {args.out}")

if __name__ == "__main__":
    main()