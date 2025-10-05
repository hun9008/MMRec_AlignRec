# scripts/visualize_ijkl.py
import os
import csv
import json
import random
import argparse
import textwrap
import html
from io import BytesIO

import matplotlib.pyplot as plt

# 인터넷에서 이미지 받기 (실패해도 전체 파이프라인은 계속)
try: 
    import requests
    from PIL import Image
    HAS_NET_DEPS = True
except Exception:
    HAS_NET_DEPS = False


def load_mapping_csv(path_csv):
    """./data/baby/i_id_mapping.csv -> itemID(int) -> asin(str)"""
    mapping = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t") if "\t" in f.readline() else None
    # 재열기 (첫 줄을 소비했으니)
    with open(path_csv, "r", encoding="utf-8") as f:
        first_line = f.readline()
        # 구분자 자동 판별 (탭/콤마)
        delimiter = "\t" if "\t" in first_line else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            asin = row["asin"]
            item_id = int(row["itemID"])
            mapping[item_id] = asin
    return mapping


def load_metadata_json(path_json):
    """./data/baby/metadata_baby.json -> asin(str) -> meta(dict)"""
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    for d in data:
        asin = d.get("asin")
        if asin:
            meta[asin] = d
    return meta


def load_result_csv(path_csv):
    """
    ./ijkl_overlap/result.csv
    헤더: i, j(close), l(middle), k(far)
    """
    rows = []
    with open(path_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            i = int(r["i"])
            j = int(r["j(close)"])
            l = int(r["l(middle)"])
            k = int(r["k(far)"])
            rows.append((i, j, l, k))
    return rows


def fetch_image(url, cache_dir="./cache_images", timeout=5):
    """
    url 이미지를 받아 PIL Image로. 캐시를 남겨 재사용.
    - 인터넷/의존성 없으면 None 반환
    - 실패해도 None
    """
    if not HAS_NET_DEPS or not url:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    # 파일명 안전화
    safe_name = url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    local_path = os.path.join(cache_dir, safe_name)

    # 캐시 히트
    if os.path.exists(local_path):
        try:
            return Image.open(local_path).convert("RGB")
        except Exception:
            pass

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        # 캐시 저장
        img.save(local_path)
        return img
    except Exception:
        return None


def wrap_text(s, width=50, max_lines=6):
    if not s:
        return ""
    s = html.unescape(str(s))
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)


def item_card(ax, asin, meta, subtitle):
    """하나의 subplot에 이미지(가능하면) + title/desc 텍스트 렌더링"""
    ax.axis("off")
    title = meta.get("title", "")
    desc  = meta.get("description", "")
    img_url = meta.get("imUrl", "")

    # 이미지 먼저
    img = fetch_image(img_url)
    if img is not None:
        ax.imshow(img)
        # 이미지 아래쪽에 텍스트를 넣기 위해 축을 한 번 더 사용
        ax.set_title(f"{subtitle}\nASIN: {asin}", fontsize=10)
    else:
        # 이미지가 없으면 텍스트 박스만
        txt = f"{subtitle}\nASIN: {asin}\n(no image)\nURL:\n{img_url}"
        ax.text(0.02, 0.98, wrap_text(txt, width=60, max_lines=12),
                va="top", ha="left", fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        return  # 아래 텍스트는 공간이 모자라므로 여기서 종료

    # 이미지 성공 시, 하단에 텍스트 오버레이(작게)
    text_block = f"{wrap_text(title, 60, 2)}\n\n{wrap_text(desc, 60, 5)}"
    ax.text(0.02, 0.02, text_block,
            va="bottom", ha="left", fontsize=8, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))


def visualize_case(i, j, l, k, id2asin, meta_dict, figsize=(16, 9)):
    """
    2x2 그리드로 i(앵커)/j(가까움)/l(중간)/k(멀리) 표시
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    items = [
        ("i (anchor)", i),
        ("j (close)", j),
        ("l (middle)", l),
        ("k (far)", k),
    ]

    for ax, (subtitle, item_id) in zip(axes, items):
        asin = id2asin.get(item_id, None)
        if asin is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{subtitle}\nitemID {item_id}\n(ASIN not found)",
                    ha="center", va="center", fontsize=10)
            continue
        meta = meta_dict.get(asin, {})
        item_card(ax, asin, meta, subtitle)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_csv", default="./ijkl_overlap/result.csv")
    parser.add_argument("--mapping_csv", default="./data/baby/i_id_mapping.csv")
    parser.add_argument("--meta_json",  default="./data/baby/metadata_baby.json")
    parser.add_argument("--row_index", type=int, default=None, help="지정하면 해당 행 사용, 미지정 시 무작위")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    assert os.path.exists(args.result_csv), f"result csv not found: {args.result_csv}"
    assert os.path.exists(args.mapping_csv), f"mapping csv not found: {args.mapping_csv}"
    assert os.path.exists(args.meta_json),  f"metadata json not found: {args.meta_json}"

    id2asin = load_mapping_csv(args.mapping_csv)
    meta    = load_metadata_json(args.meta_json)
    rows    = load_result_csv(args.result_csv)
    assert len(rows) > 0, "result.csv에 (i,j,l,k) 행이 없습니다."

    random.seed(args.seed)
    if args.row_index is None:
        idx = random.randrange(len(rows))
    else:
        assert 0 <= args.row_index < len(rows), f"row_index out of range: [0, {len(rows)-1}]"
        idx = args.row_index

    i, j, l, k = rows[idx]
    print(f"[picked row {idx}] i={i}, j={j}, l={l}, k={k}")

    visualize_case(i, j, l, k, id2asin, meta)


if __name__ == "__main__":
    main()