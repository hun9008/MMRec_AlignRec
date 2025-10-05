# ijkl_visualization_ten.py
import os
import json
import csv
import textwrap
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import BytesIO

# 네트워크가 안 되거나 라이브러리 없으면 이미지 없이 진행
try:
    import requests
    from PIL import Image
    HAS_NET = True
except Exception:
    HAS_NET = False

RESULT_CSV   = "./ijkl_overlap_alignrec/result.csv"
MAPPING_CSV  = "./data/baby/i_id_mapping.csv"
METADATA_JSON= "./data/baby/metadata_baby.json"
CACHE_DIR    = "./cache_images"

def read_mapping(path_csv):
    """itemID -> asin 매핑 (탭/콤마 자동 인식)"""

    with open(path_csv, "r", encoding="utf-8") as f:
        head = f.readline()
    delim = "\t" if "\t" in head else ","
    df = pd.read_csv(path_csv, sep=delim)
    df.columns = [c.strip() for c in df.columns]
    if not {"asin","itemID"}.issubset(df.columns):
        raise ValueError(f"mapping csv 컬럼을 확인하세요: {df.columns.tolist()}")
    return dict(zip(df["itemID"].astype(int), df["asin"].astype(str)))

def read_metadata(path_json):
    """asin -> metadata dict"""
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d.get("asin"): d for d in data if d.get("asin")}

def read_ijkl_rows(path_csv):
    """(i, j, l, k) tuple 리스트"""
    df = pd.read_csv(path_csv)
    needed = ["i","j(close)","l(middle)","k(far)"]
    if not set(needed).issubset(df.columns):
        raise ValueError(f"result.csv 컬럼을 확인하세요: {df.columns.tolist()}")
    rows = df[needed].astype(int).to_records(index=False).tolist()
    return rows

def fetch_image(url, timeout=7):
    """간단 캐시 사용하여 이미지 로드. 실패시 None"""
    if not (HAS_NET and url):
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = url.replace("://","_").replace("/","_").replace("?","_").replace("&","_").replace("=","_")
    local = os.path.join(CACHE_DIR, safe)
    try:
        from PIL import Image
        if os.path.exists(local):
            return Image.open(local).convert("RGB")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(local)
        return img
    except Exception:
        return None

def wrap_text(s, width=60, max_lines=6):
    if not s:
        return ""
    s = html.unescape(str(s))
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)

def get_item_text_and_img(item_id, id2asin, meta):
    asin = id2asin.get(int(item_id))
    if asin is None:
        return "(unknown)", "N/A", None
    m = meta.get(asin, {})
    title = m.get("title","")
    desc  = m.get("description","")
    text  = f"{title}\n\n{desc}" if desc else title
    img   = fetch_image(m.get("imUrl", ""))
    return asin, text, img

def main():
    assert os.path.exists(RESULT_CSV), f"missing {RESULT_CSV}"
    assert os.path.exists(MAPPING_CSV), f"missing {MAPPING_CSV}"
    assert os.path.exists(METADATA_JSON), f"missing {METADATA_JSON}"

    id2asin = read_mapping(MAPPING_CSV)
    meta    = read_metadata(METADATA_JSON)
    rows    = read_ijkl_rows(RESULT_CSV)

    # 무작위 10개 샘플 (실행마다 달라지도록 시드도 무작위)
    rng_seed = np.random.SeedSequence().entropy  # 임의 시드
    rng = np.random.default_rng(int(rng_seed) % (2**32 - 1))
    pick_n = min(10, len(rows))
    picked_idx = rng.choice(len(rows), size=pick_n, replace=False)
    picked = [rows[i] for i in picked_idx]

    # 10행 × 4열 (i, j, l, k)
    fig, axes = plt.subplots(nrows=pick_n, ncols=4, figsize=(20, 4*pick_n))
    if pick_n == 1:
        axes = np.array([axes])  # shape 통일

    titles = ["i (anchor)","j (close)","l (middle)","k (far)"]

    for r, (i, j, l, k) in enumerate(picked):
        for c, item_id in enumerate([i, j, l, k]):
            ax = axes[r, c]
            ax.axis("off")

            asin, text, img = get_item_text_and_img(item_id, id2asin, meta)
            if img is not None:
                ax.imshow(img)

            ax.set_title(f"{titles[c]}\nASIN: {asin}", fontsize=10)

            if text:
                # 아래쪽에 짧게 오버레이
                ax.text(
                    0.02, 0.02,
                    wrap_text(text, width=70, max_lines=6),
                    transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
                )

    plt.tight_layout()
    
    plt.savefig("ijkl_visualization_ten_alignrec.png", dpi=200)

if __name__ == "__main__":
    main()