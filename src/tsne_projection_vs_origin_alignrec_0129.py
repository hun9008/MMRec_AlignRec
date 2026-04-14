import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed, get_model


def _unpack_forward_outputs(outputs):
    if len(outputs) < 4:
        raise RuntimeError("ALIGNREC forward(train=True) must return at least 4 outputs")
    _, _, mm_embeds, content_embeds = outputs[:4]
    return mm_embeds, content_embeds


def _tsne_2d(id_items, mm_items, indices, random_state=42):
    id_np = id_items.detach().cpu().numpy()[indices]
    mm_np = mm_items.detach().cpu().numpy()[indices]
    all_data = np.concatenate([id_np, mm_np], axis=0)
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_result = tsne.fit_transform(all_data)

    sample_size = id_np.shape[0]
    id_2d = tsne_result[:sample_size]
    mm_2d = tsne_result[sample_size:]
    return id_2d, mm_2d


def visualize_origin_vs_fused(id_items, mm_items, mm_items_fused, sample_size, title, filename, seed=42):
    n_items = id_items.shape[0]
    sample_size = min(sample_size, n_items)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_items, size=sample_size, replace=False)

    origin_id_2d, origin_mm_2d = _tsne_2d(id_items, mm_items, indices, random_state=seed)
    fused_id_2d, fused_mm_2d = _tsne_2d(id_items, mm_items_fused, indices, random_state=seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("Origin space (ID/MM)")
    axes[1].set_title("Fused space (ID/MM after scaling)")

    axes[0].scatter(origin_id_2d[:, 0], origin_id_2d[:, 1], label="ID", alpha=0.8, marker="o", s=8)
    axes[0].scatter(origin_mm_2d[:, 0], origin_mm_2d[:, 1], label="MM", alpha=0.8, marker="s", s=8)

    axes[1].scatter(fused_id_2d[:, 0], fused_id_2d[:, 1], label="ID", alpha=0.8, marker="o", s=8)
    axes[1].scatter(fused_mm_2d[:, 0], fused_mm_2d[:, 1], label="MM", alpha=0.8, marker="s", s=8)

    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def main(args):
    config_dict = {
        "multimodal_data_dir": args.multimodal_data_dir,
        "save_model": False,
        "side_emb_div": 2,
        "valid_metric": "Recall@20",
        "topk": [20],
        "use_gpu": torch.cuda.is_available(),
    }
    config = Config(args.model, args.dataset, config_dict)
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key in ["knn_k", "seed", "sim_weight", "lambda_weight", "learning_rate", "n_layers"]:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][0]

    init_seed(config["seed"])

    raw_dataset = RecDataset(config)
    raw_dataset.inter_num = len(raw_dataset)
    raw_dataset.user_num = raw_dataset.get_user_num()
    raw_dataset.item_num = raw_dataset.get_item_num()

    dataloader = TrainDataLoader(config, raw_dataset)

    model_cls = get_model(config["model"])
    model = model_cls(config, dataloader).to(config["device"])

    checkpoint = torch.load(args.checkpoint_path, map_location=config["device"])
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    with torch.no_grad():
        outputs = model.forward(model.norm_adj, train=True)
    mm_embeds, content_embeds = _unpack_forward_outputs(outputs)

    n_items = model.n_items
    id_items = content_embeds[-n_items:]
    mm_items = mm_embeds[-n_items:]

    side_div = getattr(model, "side_emb_div", 0)
    if side_div and side_div != 0:
        mm_items_fused = mm_items / side_div
    else:
        mm_items_fused = mm_items

    visualize_origin_vs_fused(
        id_items,
        mm_items,
        mm_items_fused,
        sample_size=args.sample_size,
        title=args.title,
        filename=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ALIGNREC")
    parser.add_argument("--dataset", type=str, default="baby")
    parser.add_argument("--checkpoint_path", type=str, default="./saved/ALIGNREC_best.pth")
    parser.add_argument(
        "--multimodal_data_dir",
        type=str,
        default="data/baby_beit3_128token_add_title_brand_to_text/",
    )
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--title",
        type=str,
        default="t-SNE: origin vs fused (ALIGNREC ID/MM)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_origin_vs_fused_alignrec_0129.png",
    )
    args = parser.parse_args()

    main(args)
