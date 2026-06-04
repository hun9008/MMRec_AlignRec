from __future__ import annotations

import argparse
import ast
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    raise RuntimeError("figure_5_images.py requires pillow and requests.") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_MODELS = ["AlignRec", "FREEDOM", "DAMRS", "BM3", "LATTICE", "LGMRec", "SMORE", "VBPR"]
CACHE_DIR = Path(".cache/figure_5_images")
LOCAL_IMAGE_DIR = Path("cache_images")
TOP_ITEM_COLS = ["top1 itemID", "top2 itemID", "top3 itemID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-item comparison panels containing target, AnchorRec top-3 images, "
            "and another model's top-3 images."
        )
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--anchor-sim", default="figure_5_AnchorRec_sim.txt")
    parser.add_argument("--sim-template", default="figure_5_{model}_sim.txt")
    parser.add_argument("--saved-emb-dir", default="saved_emb")
    parser.add_argument("--mapping", default="/home/hun/data/data/baby/i_id_mapping.csv")
    parser.add_argument("--metadata", default="/home/hun/data/data/baby/meta_baby.json")
    parser.add_argument("--local-image-dir", default=str(LOCAL_IMAGE_DIR))
    parser.add_argument("--out-dir", default="figure_5_images")
    parser.add_argument("--limit", type=int, default=1, help="Max output images total. Use 0 for all.")
    parser.add_argument("--start-item", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=220)
    parser.add_argument("--label-height", type=int, default=34)
    parser.add_argument("--timeout", type=int, default=7)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--block-size", type=int, default=1024)
    return parser.parse_args()


def read_pipe_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=r"\s*\|\s*", engine="python")
    df.columns = [col.strip() for col in df.columns]
    return df


def l2_normalize(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def topk_neighbors_by_cosine(emb: np.ndarray, topk: int, block_size: int) -> np.ndarray:
    n_items = emb.shape[0]
    neighbors = np.empty((n_items, topk), dtype=np.int32)

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        sims = emb[start:end] @ emb.T
        row_idx = np.arange(end - start)
        sims[row_idx, np.arange(start, end)] = -np.inf

        part = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        neighbors[start:end] = np.take_along_axis(part, order, axis=1).astype(np.int32)

    return neighbors


def top_items_from_sim_or_final(
    model: str,
    sim_path: Path,
    saved_emb_dir: Path,
    dtype: np.dtype,
    block_size: int,
) -> np.ndarray:
    df = read_pipe_table(sim_path)
    if all(col in df.columns for col in TOP_ITEM_COLS):
        return df[TOP_ITEM_COLS].apply(pd.to_numeric).to_numpy(dtype=np.int32)

    final_item_path = saved_emb_dir / model / "final_item.npy"
    if not final_item_path.exists():
        raise FileNotFoundError(
            f"{sim_path} has no top itemID columns and final embedding is missing: {final_item_path}"
        )
    final_norm = l2_normalize(np.load(final_item_path), dtype)
    return topk_neighbors_by_cosine(final_norm, topk=3, block_size=block_size)


def load_id2asin(path: Path) -> dict[int, str]:
    df = pd.read_csv(path, sep=None, engine="python")
    if "itemID" not in df.columns or "asin" not in df.columns:
        raise ValueError(f"{path} must contain asin and itemID columns.")
    return {int(row.itemID): str(row.asin) for row in df.itertuples(index=False)}


def load_metadata(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [ast.literal_eval(line) for line in text.splitlines() if line.strip()]
    return {str(item.get("asin")): item for item in data if item.get("asin")}


def cache_name(url: str) -> str:
    return (
        url.replace("://", "_")
        .replace("/", "_")
        .replace("?", "_")
        .replace("&", "_")
        .replace("=", "_")
    )


def fetch_item_image(
    item_id: int,
    id2asin: dict[int, str],
    metadata: dict[str, dict],
    local_image_dir: Path,
    timeout: int,
) -> Image.Image | None:
    asin = id2asin.get(int(item_id))
    if asin is None:
        return None

    url = metadata.get(asin, {}).get("imUrl", "")
    if not url:
        return None

    local_cached = local_image_dir / cache_name(url)
    if local_cached.exists():
        try:
            return Image.open(local_cached).convert("RGB")
        except Exception:
            pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / cache_name(url)
    try:
        if local.exists():
            return Image.open(local).convert("RGB")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(local)
        return image
    except Exception:
        return None


def placeholder(cell_size: int, text: str) -> Image.Image:
    image = Image.new("RGB", (cell_size, cell_size), color=(242, 242, 242))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, cell_size - 1, cell_size - 1), outline=(190, 190, 190), width=2)
    draw.multiline_text((12, cell_size // 2 - 18), text, fill=(80, 80, 80), spacing=4)
    return image


def fit_image(image: Image.Image | None, cell_size: int, fallback_text: str) -> Image.Image:
    if image is None:
        return placeholder(cell_size, fallback_text)

    fitted = Image.new("RGB", (cell_size, cell_size), color="white")
    image = image.copy()
    image.thumbnail((cell_size, cell_size), Image.LANCZOS)
    x = (cell_size - image.width) // 2
    y = (cell_size - image.height) // 2
    fitted.paste(image, (x, y))
    return fitted


def draw_labeled_cell(
    canvas: Image.Image,
    image: Image.Image,
    label: str,
    x: int,
    y: int,
    cell_size: int,
    label_height: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + cell_size - 1, y + label_height - 1), fill=(255, 255, 255))
    draw.text((x + 8, y + 8), label, fill=(25, 25, 25))
    canvas.paste(image, (x, y + label_height))
    draw.rectangle(
        (x, y, x + cell_size - 1, y + label_height + cell_size - 1),
        outline=(210, 210, 210),
        width=1,
    )


def save_panel(
    out_path: Path,
    target_id: int,
    anchor_top: np.ndarray,
    model_name: str,
    model_top: np.ndarray,
    id2asin: dict[int, str],
    metadata: dict[str, dict],
    local_image_dir: Path,
    timeout: int,
    cell_size: int,
    label_height: int,
) -> None:
    gap = 14
    width = 3 * cell_size + 2 * gap
    height = 3 * (cell_size + label_height) + 2 * gap
    canvas = Image.new("RGB", (width, height), color="white")

    target_img = fit_image(
        fetch_item_image(target_id, id2asin, metadata, local_image_dir, timeout),
        cell_size,
        f"missing\nitem {target_id}",
    )
    target_x = cell_size + gap
    draw_labeled_cell(canvas, target_img, f"tgtItemID {target_id}", target_x, 0, cell_size, label_height)

    rows = [
        ("AnchorRec", anchor_top, cell_size + label_height + gap),
        (model_name, model_top, 2 * (cell_size + label_height + gap)),
    ]
    for row_name, item_ids, y in rows:
        for col_idx, item_id in enumerate(item_ids):
            x = col_idx * (cell_size + gap)
            image = fit_image(
                fetch_item_image(int(item_id), id2asin, metadata, local_image_dir, timeout),
                cell_size,
                f"missing\nitem {int(item_id)}",
            )
            label = f"{row_name}_Top{col_idx + 1} {int(item_id)}"
            draw_labeled_cell(canvas, image, label, x, y, cell_size, label_height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def progress_iter(items, total: int):
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc="figure_5_images")


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)
    out_dir = Path(args.out_dir)
    saved_emb_dir = Path(args.saved_emb_dir)
    local_image_dir = Path(args.local_image_dir)
    id2asin = load_id2asin(Path(args.mapping))
    metadata = load_metadata(Path(args.metadata))

    anchor_top = top_items_from_sim_or_final(
        "AnchorRec",
        Path(args.anchor_sim),
        saved_emb_dir,
        dtype,
        args.block_size,
    )
    n_items = anchor_top.shape[0]
    target_ids = range(args.start_item, n_items)

    jobs: list[tuple[str, int]] = []
    for model in args.models:
        sim_path = Path(args.sim_template.format(model=model))
        if not sim_path.exists():
            continue
        for target_id in target_ids:
            jobs.append((model, target_id))
            if args.limit > 0 and len(jobs) >= args.limit:
                break
        if args.limit > 0 and len(jobs) >= args.limit:
            break

    model_top_cache: dict[str, np.ndarray] = {}
    for model, target_id in progress_iter(jobs, total=len(jobs)):
        if model not in model_top_cache:
            sim_path = Path(args.sim_template.format(model=model))
            model_top_cache[model] = top_items_from_sim_or_final(
                model,
                sim_path,
                saved_emb_dir,
                dtype,
                args.block_size,
            )

        out_path = out_dir / f"{model}_{target_id}.png"
        save_panel(
            out_path,
            target_id,
            anchor_top[target_id],
            model,
            model_top_cache[model][target_id],
            id2asin,
            metadata,
            local_image_dir,
            args.timeout,
            args.cell_size,
            args.label_height,
        )

    print(f"[SAVE] {len(jobs)} image panels to {out_dir}")


if __name__ == "__main__":
    main()
