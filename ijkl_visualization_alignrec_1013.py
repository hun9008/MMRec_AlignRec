import os, json, html, argparse, time, logging, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Optional, Tuple, Set, Dict

try:
    import requests
    from PIL import Image
    _HAS_PIL_REQ = True
except Exception:
    _HAS_PIL_REQ = False

RESULT_CSV    = "./ijkl_overlap_alignrec/result.csv"
MAPPING_CSV   = "./data/baby/i_id_mapping.csv"
METADATA_JSON = "./data/baby/metadata_baby.json"
INTER_PATH    = "./data/baby/baby.inter"   # ✅ 기본 interactions 경로 (TSV/CSV 자동 인식)
CACHE_DIR     = "./cache_images"

# ---------------- logging & timer ----------------
def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

class tick:
    def __init__(self, msg: str):
        self.msg, self.t0 = msg, None
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
    """result.csv에서 i와 j(close)만 읽음 → 빠르고 가볍게."""
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
    if noimg or not (_HAS_PIL_REQ and url):
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = url.replace("://","_").replace("/","_").replace("?","_").replace("&","_").replace("=","_")
    local = os.path.join(CACHE_DIR, safe)
    try:
        from PIL import Image
        if os.path.exists(local):
            if log_images:
                logging.debug(f"[IMG] cache hit: {local}")
            return Image.open(local).convert("RGB")
        if log_images:
            logging.debug(f"[IMG] downloading: {url}")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(local)
        return img
    except Exception as e:
        if log_images:
            logging.warning(f"[IMG] failed: {url} ({e})")
        return None

def wrap_text(s: str, width: int = 60, max_lines: int = 6) -> str:
    if not s:
        return ""
    import textwrap
    s = html.unescape(str(s))
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)

def get_item_text_and_img(item_id: int, id2asin: dict, meta: dict,
                          log_images: bool = False, noimg: bool = False):
    asin = id2asin.get(int(item_id))
    if asin is None:
        return "(unknown)", "N/A", None
    m = meta.get(asin, {})
    title = m.get("title","")
    desc  = m.get("description","")
    text  = f"{title}\n\n{desc}" if desc else title
    img   = fetch_image(m.get("imUrl",""), log_images=log_images, noimg=noimg)
    return asin, text, img

# ---------- interactions ----------
def read_interactions_csv(path_csv: str) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Returns:
      item2users: {item_id -> {user_ids}}
      user2items:{user_id -> {item_ids}}
    컬럼 자동 탐지: userID/itemID/user_id/item_id/uid/iid 등
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
        # itertuples는 우리가 지정한 두 컬럼 순서로 나옴
        for u, it in df.itertuples(index=False):
            item2users.setdefault(it, set()).add(u)
            user2items.setdefault(u, set()).add(it)

        logging.info(f"interactions loaded: |users|={len(user2items)}, |items|={len(item2users)}")
        return item2users, user2items

def twohop_item_count(item_id: int, item2users: Dict[int, Set[int]], user2items: Dict[int, Set[int]]) -> int:
    users = item2users.get(item_id)
    if not users:
        return 0
    neigh_items: Set[int] = set()
    for u in users:
        neigh_items.update(user2items.get(u, ()))
    neigh_items.discard(item_id)
    return len(neigh_items)

# ------------- FAST close-index (vectorized, dedup selectable) -------------
def build_close_index_fast(df_two_cols: pd.DataFrame, topk: int, dedup: str = "global") -> dict:
    """
    dedup:
      - "consecutive": 같은 i에서 바로 앞 j와 동일하면 제거
      - "global": 같은 i에서 동일 j는 한 번만 유지
    반환: dict[i] = [j1, ..., j_topk<=K]
    """
    with tick(f"close 인덱스 생성 (dedup={dedup})"):
        if dedup == "consecutive":
            same_i = df_two_cols["i"].eq(df_two_cols["i"].shift())
            same_j = df_two_cols["j(close)"].eq(df_two_cols["j(close)"].shift())
            mask   = ~(same_i & same_j)
            df = df_two_cols[mask].copy()
        else:
            df = df_two_cols.drop_duplicates(subset=["i","j(close)"], keep="first").copy()

        # alignrec CSV는 close/middle/far를 분리 저장 → j가 -1인 행이 있을 수 있음. 걸러낸다.
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
                    seen.add(x)
                    deduped.append(x)
            i2js[i] = deduped[:topk]

        logging.debug(f"anchors: {len(i2js)} (valid-only, topk-truncated)")
        return i2js

def choose_anchor_with_fallback(
    i2js: dict, want_i: Optional[int], need_k: int, rng: np.random.Generator
) -> Tuple[int, int]:
    """
    반환: (anchor_i, effective_k)
    - need_k 이상 가진 i가 없으면: 가장 많은 j를 가진 i 선택 + k 자동 축소
    - want_i가 부족하면: 충분 후보로 대체, 그마저 없으면 위와 동일
    """
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

# ------------- viz -------------
def visualize_anchor_with_topk_close(anchor_i: int, js_topk: list, id2asin: dict, meta: dict,
                                     out_png: str, log_images: bool = False, noimg: bool = False,
                                     twohop_counts: Optional[Dict[int, int]] = None):
    with tick("시각화 생성"):
        k = len(js_topk)
        fig, axes = plt.subplots(nrows=1, ncols=1 + k, figsize=(4*(1+k), 6))
        if (1 + k) == 1:
            import numpy as _np
            axes = _np.array([axes])

        titles = [f"i (anchor)\nID: {anchor_i}"] + [
            f"j#{r} (close)\nID: {jid}" for r, jid in enumerate(js_topk, start=1)
        ]
        item_ids = [anchor_i] + js_topk

        for c, item_id in enumerate(item_ids):
            ax = axes[c]
            ax.axis("off")
            asin, text, img = get_item_text_and_img(item_id, id2asin, meta,
                                                    log_images=log_images, noimg=noimg)
            if img is not None:
                ax.imshow(img)

            # 제목 + 2-hop 줄
            extra = ""
            if twohop_counts is not None and item_id in twohop_counts:
                extra = f"\n2-hop: {twohop_counts[item_id]}"
            ax.set_title(f"{titles[c]}\nASIN: {asin}{extra}", fontsize=10)

            # 본문 텍스트
            if text:
                ax.text(
                    0.02, 0.02,
                    wrap_text(text, width=70, max_lines=6),
                    transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
                )

            # 좌상단 배지
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
    ap = argparse.ArgumentParser(description="Visualize top-k close neighbors for a fixed anchor i (AlignRec, with 2-hop counts).")
    ap.add_argument("--anchor", type=int, default=1942, help="고정할 anchor i (부족하면 자동 대체/축소)")
    ap.add_argument("--topk", type=int, default=4, help="가까운 이웃 j 개수")
    ap.add_argument("--out", type=str, default="ijkl_visualization_topkclose_alignrec.png", help="출력 파일명")
    ap.add_argument("--dedup", type=str, default="global", choices=["global","consecutive"],
                    help="중복 제거 방식: global=전역 중복 제거, consecutive=연속 중복 제거")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--log-images", action="store_true")
    ap.add_argument("--noimg", action="store_true")
    ap.add_argument("--filter-meta", action="store_true")
    ap.add_argument("--no-interactions", action="store_true",
                    help="2-hop 계산 끄기 (기본: 켜짐, 경로는 INTER_PATH)")
    args = ap.parse_args()

    setup_logger(args.log_level)

    with tick("입력 파일 존재 확인"):
        assert os.path.exists(RESULT_CSV), f"missing {RESULT_CSV}"
        assert os.path.exists(MAPPING_CSV), f"missing {MAPPING_CSV}"
        assert os.path.exists(METADATA_JSON), f"missing {METADATA_JSON}"

    id2asin = read_mapping(MAPPING_CSV)

    # i,j 두 컬럼만 읽고 벡터화 인덱스 생성
    df_two = read_result_two_cols(RESULT_CSV)
    i2js   = build_close_index_fast(df_two, topk=args.topk, dedup=args.dedup)

    # anchor 선택 (대체/축소 포함)
    rng_seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(int(rng_seed) % (2**32 - 1))
    anchor_i, eff_k = choose_anchor_with_fallback(i2js, want_i=args.anchor, need_k=args.topk, rng=rng)

    js_topk  = i2js[anchor_i][:eff_k]
    logging.info(f"[INFO] anchor i={anchor_i}, close j(top-{eff_k})={js_topk}")

    # (기본 ON) interactions 로드 & 2-hop 계산
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
            logging.warning(f"interactions 파일이 없어 2-hop 계산을 생략합니다: {INTER_PATH}")
        else:
            logging.info("요청에 의해 2-hop 계산을 생략합니다 (--no-interactions)")

    # 메타데이터 로딩 (필요 asin만 선택적으로)
    if args.filter_meta:
        with tick("필요 asin 집합 구성"):
            ids = [anchor_i] + js_topk
            needed = {id2asin.get(int(x)) for x in ids if id2asin.get(int(x))}
        with tick("metadata 로드(필터)"):
            meta = read_metadata_filtered(METADATA_JSON, needed)
    else:
        with tick("metadata 로드(전체)"):
            meta = read_metadata_all(METADATA_JSON)

    visualize_anchor_with_topk_close(
        anchor_i, js_topk, id2asin, meta, args.out,
        log_images=args.log_images, noimg=args.noimg,
        twohop_counts=twohop_counts
    )
    logging.info(f"[DONE] saved to: {args.out}")

if __name__ == "__main__":
    main()