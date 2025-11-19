import os, json, html, argparse, time, logging, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from io import BytesIO
from typing import Dict, Set, Tuple, Optional, List

# ---------- optional deps ----------
_HAS_PIL_REQ = False
_HAS_TORCH = False
_HAS_TORCHVISION = False
_HAS_TRANSFORMERS = False
_HAS_SBERT = False

try:
    import requests
    from PIL import Image
    _HAS_PIL_REQ = True
except Exception:
    pass

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
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    pass

# === 기본 경로 설정 ===
MAPPING_CSV    = "./data/baby/i_id_mapping.csv"
METADATA_JSON  = "./data/baby/metadata_baby.json"
INTER_PATH     = "./data/baby/baby.inter"
CACHE_DIR      = "./cache_images"

ALIGNREC_RESULT = "./ijkl_overlap_alignrec/result.csv"
ANCHOR_RESULT   = "./ijkl_overlap_1104/result.csv"

# -------------------------------------------------
# logging & timer
# -------------------------------------------------
def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

class tick:
    def __init__(self, msg: str):
        self.msg = msg
        self.t0 = None
    def __enter__(self):
        logging.info(f"▶ {self.msg} 시작")
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, et, ev, tb):
        dt = (time.perf_counter() - self.t0) * 1000
        if et is None:
            logging.info(f"✔ {self.msg} 완료 ({dt:.1f} ms)")
        else:
            logging.exception(f"✘ {self.msg} 실패 ({dt:.1f} ms) - {ev}")

# -------------------------------------------------
# basic utils
# -------------------------------------------------
def read_mapping(path_csv: str) -> Dict[int, str]:
    with open(path_csv, "r", encoding="utf-8") as f:
        head = f.readline()
    delim = "\t" if "\t" in head else ","
    df = pd.read_csv(
        path_csv,
        sep=delim,
        usecols=["itemID", "asin"],
        dtype={"itemID": "int32", "asin": "string"},
    )
    return dict(zip(df["itemID"].astype(int), df["asin"].astype(str)))

def read_result_two_cols(path_csv: str) -> pd.DataFrame:
    return pd.read_csv(
        path_csv,
        usecols=["i", "j(close)"],
        dtype={"i": "int32", "j(close)": "int32"},
        low_memory=False,
    )

def read_metadata_all(path_json: str) -> Dict[str, dict]:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d.get("asin"): d for d in data if d.get("asin")}

def fetch_image(url: str, timeout: int = 7):
    if not (_HAS_PIL_REQ and url):
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = (
        url.replace("://", "_")
        .replace("/", "_")
        .replace("?", "_")
        .replace("&", "_")
        .replace("=", "_")
    )
    local = os.path.join(CACHE_DIR, safe)
    try:
        if os.path.exists(local):
            from PIL import Image
            return Image.open(local).convert("RGB")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        from PIL import Image
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(local)
        return img
    except Exception:
        return None

def wrap_text(s: str, width: int = 60, max_lines: int = 5) -> str:
    if not s:
        return ""
    import textwrap
    s = html.unescape(str(s))
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)

def get_item_text_and_img(
    item_id: int,
    id2asin: Dict[int, str],
    meta: Dict[str, dict],
    noimg: bool = False,
):
    asin = id2asin.get(int(item_id))
    if asin is None:
        return "(unknown)", "", None
    m = meta.get(asin, {})
    title = m.get("title", "")
    desc = m.get("description", "")
    text = f"{title}\n\n{desc}" if desc else title
    img = None
    if not noimg:
        img = fetch_image(m.get("imUrl", ""))
    return asin, text, img

# -------------------------------------------------
# close index
# -------------------------------------------------
def build_close_index(df_two_cols: pd.DataFrame, topk: int = 3) -> Dict[int, list]:
    df = df_two_cols[
        (df_two_cols["j(close)"] >= 0)
        & (df_two_cols["i"] != df_two_cols["j(close)"])
    ].copy()
    df["rk"] = df.groupby("i").cumcount()
    df = df[df["rk"] < topk]
    agg = df.groupby("i", sort=False)["j(close)"].apply(list)
    return {int(i): list(lst) for i, lst in agg.items()}

# -------------------------------------------------
# anchor pick (공통 anchor)
# -------------------------------------------------
def pick_common_anchor(
    want_i: int,
    i2js_alignrec: Dict[int, list],
    i2js_anchor: Dict[int, list],
    topk: int,
) -> int:
    has_a = want_i in i2js_alignrec and len(i2js_alignrec[want_i]) >= topk
    has_b = want_i in i2js_anchor and len(i2js_anchor[want_i]) >= topk

    if has_a and has_b:
        logging.info(f"요청 anchor {want_i} 사용 (두 모델 모두 top-{topk} OK)")
        return want_i

    if not has_a:
        logging.warning(f"AlignRec 결과에 anchor {want_i} 없음 또는 close 부족")
    if not has_b:
        logging.warning(f"Anchor 결과에 anchor {want_i} 없음 또는 close 부족")

    common = set(i2js_alignrec.keys()) & set(i2js_anchor.keys())
    candidates = [
        i
        for i in common
        if len(i2js_alignrec[i]) >= topk and len(i2js_anchor[i]) >= topk
    ]
    if candidates:
        chosen = sorted(candidates)[0]
        logging.warning(f"→ 공통 anchor {chosen} 사용")
        return chosen

    raise ValueError("공통 anchor를 찾을 수 없습니다.")

# -------------------------------------------------
# interactions & 2-hop
# -------------------------------------------------
def read_interactions_csv(path_csv: str) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
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
            raise ValueError(
                f"Cannot detect user/item columns. columns={list(df.columns)}"
            )

        df = df[[user_col, item_col]].dropna()
        df[user_col] = df[user_col].astype(int)
        df[item_col] = df[item_col].astype(int)

        item2users: Dict[int, Set[int]] = {}
        user2items: Dict[int, Set[int]] = {}
        for u, it in df.itertuples(index=False):
            item2users.setdefault(it, set()).add(u)
            user2items.setdefault(u, set()).add(it)

        logging.info(
            f"interactions loaded: |users|={len(user2items)}, |items|={len(item2users)}"
        )
        return item2users, user2items

def twohop_item_count(
    item_id: int,
    item2users: Dict[int, Set[int]],
    user2items: Dict[int, Set[int]],
) -> int:
    users = item2users.get(item_id)
    if not users:
        return 0
    neigh_items: Set[int] = set()
    for u in users:
        neigh_items.update(user2items.get(u, ()))
    neigh_items.discard(item_id)
    return len(neigh_items)

# -------------------------------------------------
# embedding helpers (img + text)
# -------------------------------------------------
_IMG_MODEL = None
_IMG_TRANS = None
_SBERT_MODEL = None
_TXT_MODEL = None
_TXT_TOK = None
_DEVICE = "cpu"

def cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))

def setup_image_model(device: str) -> bool:
    global _IMG_MODEL, _IMG_TRANS, _DEVICE
    _DEVICE = device
    if not (_HAS_TORCH and _HAS_TORCHVISION):
        return False
    with tick("ResNet50 로드"):
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(_DEVICE)
        _IMG_MODEL = model
        _IMG_TRANS = weights.transforms()
    return True

def embed_image(img_pil) -> Optional[np.ndarray]:
    if _IMG_MODEL is None or _IMG_TRANS is None or not _HAS_TORCH:
        return None
    x = _IMG_TRANS(img_pil).unsqueeze(0)
    if _DEVICE != "cpu":
        x = x.to(_DEVICE)
    with torch.no_grad():
        x = _IMG_MODEL(x)
        if x.dim() == 4:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        v = x.flatten().float()
        v = v / (v.norm() + 1e-9)
    return v.detach().cpu().numpy()

def setup_sbert_model(device: str, model_name: str) -> bool:
    global _SBERT_MODEL, _DEVICE
    _DEVICE = device
    if not (_HAS_TORCH and _HAS_SBERT):
        return False
    with tick(f"SBERT 로드 ({model_name})"):
        mdl = SentenceTransformer(model_name, device=_DEVICE)
        _SBERT_MODEL = mdl
    return True

def embed_text_sbert(texts: List[str]) -> Optional[np.ndarray]:
    if _SBERT_MODEL is None:
        return None
    vecs = _SBERT_MODEL.encode(
        texts,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs

def setup_text_model(device: str, model_name: str) -> bool:
    global _TXT_MODEL, _TXT_TOK, _DEVICE
    _DEVICE = device
    if not (_HAS_TORCH and _HAS_TRANSFORMERS):
        return False
    with tick(f"BERT 로드 ({model_name})"):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval().to(_DEVICE)
        _TXT_MODEL, _TXT_TOK = mdl, tok
    return True

def embed_text_bert(texts: List[str], max_len: int = 128) -> Optional[np.ndarray]:
    if _TXT_MODEL is None or _TXT_TOK is None or not _HAS_TORCH:
        return None
    outs = []
    bs = 8
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = _TXT_TOK(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        )
        if _DEVICE != "cpu":
            for k in enc:
                enc[k] = enc[k].to(_DEVICE)
        with torch.no_grad():
            out = _TXT_MODEL(**enc).last_hidden_state  # (B,L,H)
            mask = enc["attention_mask"].unsqueeze(-1)  # (B,L,1)
            v = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
            outs.append(v.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else None

# -------------------------------------------------
# stat line helper
# -------------------------------------------------
def build_stat_line(
    item_id: int,
    twohop_counts: Optional[Dict[int, int]],
    img_sims: Optional[Dict[int, float]],
    txt_sims: Optional[Dict[int, float]],
    min_width: int = 40,   # 텍스트 width와 맞춰줌
) -> str:
    parts = []
    if twohop_counts is not None and item_id in twohop_counts:
        parts.append(f"2-hop: {twohop_counts[item_id]}")
    if img_sims is not None and item_id in img_sims:
        v = img_sims[item_id]
        if v is not None:
            parts.append(f"img cos: {v:.3f}")
    if txt_sims is not None and item_id in txt_sims:
        v = txt_sims[item_id]
        if v is not None:
            parts.append(f"txt cos: {v:.3f}")

    if not parts:
        return ""

    line = " \n ".join(parts)

    # 텍스트 박스 폭과 비슷하게 보이도록 오른쪽에 공백 padding
    if len(line) < min_width:
        line = line.ljust(min_width)

    return line

# -------------------------------------------------
# visualization
# -------------------------------------------------
def visualize_alignrec_vs_anchor(
    anchor_i: int,
    js_alignrec: list,
    js_anchor: list,
    id2asin: Dict[int, str],
    meta: Dict[str, dict],
    out_png: str,
    twohop_counts: Optional[Dict[int, int]] = None,
    img_sims: Optional[Dict[int, float]] = None,
    txt_sims: Optional[Dict[int, float]] = None,
):
    with tick("시각화 생성"):
        topk = max(len(js_alignrec), len(js_anchor))
        cols = 1 + topk  # col0: anchor, col1~: neighbors

        fig = plt.figure(figsize=(4 * cols, 9))
        gs = GridSpec(2, cols, figure=fig, wspace=0.3, hspace=1.8)

        # ----- anchor column (공유) : title → image → text (stat 없음)
        ax_a = fig.add_subplot(gs[:, 0])
        ax_a.axis("off")
        _, text_a, img_a = get_item_text_and_img(anchor_i, id2asin, meta)

        # anchor title (위)
        ax_a.text(
            0.5,
            1.3,
            f"[Item ID] : {anchor_i}",
            transform=ax_a.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

        # anchor image
        if img_a is not None:
            ax_a.imshow(img_a)

        # anchor text (아래)
        if text_a:
            ax_a.text(
                0.02,
                -0.20,
                wrap_text(text_a, width=40, max_lines=8),
                transform=ax_a.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        # ----- row titles
        fig.text(0.62, 0.97, "AlignRec",  ha="center", fontsize=14, fontweight="bold")
        fig.text(0.62, 0.43, "AnchorRec", ha="center", fontsize=14, fontweight="bold")

        # 좌표 설정: 타이틀 / stat / 이미지 / 텍스트 분리
        TITLE_Y = 1.3   # 축 위쪽
        STAT_Y  = 1.02   # 타이틀 아래
        TEXT_Y  = -0.20  # 아래쪽 텍스트 시작

        # ----- top row: AlignRec
        for idx in range(topk):
            if idx >= len(js_alignrec):
                continue
            item_id = js_alignrec[idx]
            asin, text, img = get_item_text_and_img(item_id, id2asin, meta)

            ax = fig.add_subplot(gs[0, idx + 1])
            ax.axis("off")

            # title (위, 중앙)
            ax.text(
                0.5,
                TITLE_Y,
                f"[Item j#{idx+1} ID] : {item_id}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

            # stat (그 아래 한 줄)
            stat_line = build_stat_line(item_id, twohop_counts, img_sims, txt_sims)
            if stat_line:
                ax.text(
                    0.02,
                    STAT_Y,
                    stat_line,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                )

            # image
            if img is not None:
                ax.imshow(img)

            # text
            if text:
                ax.text(
                    0.02,
                    TEXT_Y,
                    wrap_text(text, width=40, max_lines=6),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.5,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                )

        # ----- bottom row: AnchorRec
        for idx in range(topk):
            if idx >= len(js_anchor):
                continue
            item_id = js_anchor[idx]
            asin, text, img = get_item_text_and_img(item_id, id2asin, meta)

            ax = fig.add_subplot(gs[1, idx + 1])
            ax.axis("off")

            # title
            ax.text(
                0.5,
                TITLE_Y + 0.05,
                f"[Item j#{idx+1} ID] : {item_id}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

            # stat
            stat_line = build_stat_line(item_id, twohop_counts, img_sims, txt_sims)
            if stat_line:
                ax.text(
                    0.02,
                    STAT_Y,
                    stat_line,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                )

            # image
            if img is not None:
                ax.imshow(img)

            # text
            if text:
                ax.text(
                    0.02,
                    TEXT_Y,
                    wrap_text(text, width=40, max_lines=6),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.5,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                )

        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[DONE] saved figure → {out_png}")

# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Visualize AlignRec vs Anchor close items with shared anchor column"
    )
    ap.add_argument("--anchor", type=int, required=True, help="시각화할 anchor i")
    ap.add_argument("--topk", type=int, default=3, help="각 모델당 close item 개수")
    ap.add_argument("--out", type=str, default="visualize_alignrec_vs_anchor_final.png")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-img-sim", action="store_true")
    ap.add_argument("--no-txt-sim", action="store_true")
    ap.add_argument("--sbert-model", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--bert-model", type=str, default="bert-base-uncased")
    ap.add_argument("--no-interactions", action="store_true")
    ap.add_argument("--log-level", type=str, default="INFO")

    args = ap.parse_args()
    setup_logger(args.log_level)

    if not (_HAS_TORCH and torch.cuda.is_available()) or not args.device.startswith("cuda"):
        args.device = "cpu"

    with tick("입력 파일 확인"):
        assert os.path.exists(MAPPING_CSV)
        assert os.path.exists(METADATA_JSON)
        assert os.path.exists(ALIGNREC_RESULT)
        assert os.path.exists(ANCHOR_RESULT)

    id2asin = read_mapping(MAPPING_CSV)
    meta = read_metadata_all(METADATA_JSON)

    df_alignrec = read_result_two_cols(ALIGNREC_RESULT)
    df_anchor   = read_result_two_cols(ANCHOR_RESULT)

    i2js_alignrec = build_close_index(df_alignrec, topk=args.topk)
    i2js_anchor   = build_close_index(df_anchor,   topk=args.topk)

    anchor_i = pick_common_anchor(args.anchor, i2js_alignrec, i2js_anchor, args.topk)
    js_alignrec = i2js_alignrec[anchor_i][:args.topk]
    js_anchor   = i2js_anchor[anchor_i][:args.topk]

    logging.info(
        f"최종 anchor={anchor_i}, AlignRec close={js_alignrec}, Anchor close={js_anchor}"
    )

    targets = [anchor_i] + js_alignrec + js_anchor
    targets = list(dict.fromkeys(targets))

    # 2-hop
    twohop_counts = None
    if not args.no_interactions and os.path.exists(INTER_PATH):
        item2users, user2items = read_interactions_csv(INTER_PATH)
        with tick("2-hop 이웃 개수 계산"):
            twohop_counts = {
                it: twohop_item_count(it, item2users, user2items) for it in targets
            }
    elif not os.path.exists(INTER_PATH):
        logging.warning(f"interactions 파일 없음 → 2-hop 계산 생략 ({INTER_PATH})")

    # img cos
    img_sims = None
    if (not args.no_img_sim) and _HAS_TORCH and _HAS_TORCHVISION and _HAS_PIL_REQ:
        if setup_image_model(args.device):
            _, _, anchor_img = get_item_text_and_img(anchor_i, id2asin, meta)
            if anchor_img is not None:
                img_sims = {}
                with tick("이미지 임베딩 계산"):
                    va = embed_image(anchor_img)
                    for it in targets:
                        if it == anchor_i:
                            img_sims[it] = None
                            continue
                        _, _, img = get_item_text_and_img(it, id2asin, meta)
                        v = embed_image(img) if img is not None else None
                        img_sims[it] = cosine(va, v) if (va is not None and v is not None) else None
            else:
                logging.warning("anchor 이미지 없음 → img cos 생략")
        else:
            logging.warning("torch/torchvision 미탑재 → img cos 생략")

    # txt cos
    txt_sims = None
    if not args.no_txt_sim:
        texts = [(get_item_text_and_img(it, id2asin, meta, noimg=True)[1] or "") for it in targets]
        vecs = None
        if _HAS_SBERT and _HAS_TORCH and setup_sbert_model(args.device, args.sbert_model):
            with tick("텍스트 임베딩 계산 (SBERT)"):
                vecs = embed_text_sbert(texts)
        if vecs is None and _HAS_TRANSFORMERS and _HAS_TORCH and setup_text_model(args.device, args.bert_model):
            with tick("텍스트 임베딩 계산 (BERT)"):
                vecs = embed_text_bert(texts)
        if vecs is None:
            logging.warning("텍스트 임베딩 불가 → txt cos 생략")
        else:
            txt_sims = {}
            va = vecs[0]
            for idx, it in enumerate(targets):
                txt_sims[it] = None if it == anchor_i else cosine(va, vecs[idx])

    base, ext = os.path.splitext(args.out)
    if not ext:
        ext = ".png"
    out_png = f"{base}_{anchor_i}{ext}"

    visualize_alignrec_vs_anchor(
        anchor_i,
        js_alignrec,
        js_anchor,
        id2asin,
        meta,
        out_png,
        twohop_counts=twohop_counts,
        img_sims=img_sims,
        txt_sims=txt_sims,
    )

if __name__ == "__main__":
    main()