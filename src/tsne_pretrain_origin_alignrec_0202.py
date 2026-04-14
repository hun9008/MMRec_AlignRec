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


def _tsne_2d(mm_items, id_items, indices, random_state=42):
    mm_np = mm_items.detach().cpu().numpy()[indices]
    id_np = id_items.detach().cpu().numpy()[indices]

    all_data = np.concatenate([mm_np, id_np], axis=0)
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_result = tsne.fit_transform(all_data)

    sample_size = mm_np.shape[0]
    mm_2d = tsne_result[:sample_size]
    id_2d = tsne_result[sample_size:]
    return mm_2d, id_2d


def visualize_pretrain_vs_origin(pretrain, origin, sample_size, title, filename, seed=42):
    n_items = pretrain["MM"].shape[0]
    sample_size = min(sample_size, n_items)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_items, size=sample_size, replace=False)

    pre_mm_2d, pre_id_2d = _tsne_2d(
        pretrain["MM"], pretrain["ID"], indices, random_state=seed
    )
    org_mm_2d, org_id_2d = _tsne_2d(
        origin["MM"], origin["ID"], indices, random_state=seed
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("Pre-trained + random ID (MM/ID)")
    axes[1].set_title("Origin + trained ID (MM/ID)")

    axes[0].scatter(pre_mm_2d[:, 0], pre_mm_2d[:, 1], label="MM", alpha=0.8, marker="s", s=8)
    axes[0].scatter(pre_id_2d[:, 0], pre_id_2d[:, 1], label="ID(init)", alpha=0.8, marker="o", s=8)

    axes[1].scatter(org_mm_2d[:, 0], org_mm_2d[:, 1], label="MM", alpha=0.8, marker="s", s=8)
    axes[1].scatter(org_id_2d[:, 0], org_id_2d[:, 1], label="ID(trained)", alpha=0.8, marker="o", s=8)

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


def _extract_pretrain_embeddings(model):
    if not hasattr(model, "mm_embedding"):
        raise RuntimeError("model must have mm_embedding for pre-trained embeddings")

    mm_pre = model.mm_embedding.weight

    # If projection exists, align to embedding_dim
    if hasattr(model, "mm_trs"):
        if getattr(model, "use_ln", False) and hasattr(model, "mm_ln"):
            mm_pre = model.mm_trs(model.mm_ln(mm_pre))
        else:
            mm_pre = model.mm_trs(mm_pre)

    # Random initial ID embedding (before checkpoint load)
    if not hasattr(model, "item_id_embedding"):
        raise RuntimeError("model must have item_id_embedding for ID embeddings")
    id_init = model.item_id_embedding.weight

    return {"MM": mm_pre, "ID": id_init}


def _extract_origin_embeddings(model):
    with torch.no_grad():
        outputs = model.forward(model.norm_adj, train=True)

    if len(outputs) < 3:
        raise RuntimeError("model.forward must return mm embeddings with train=True")

    _, _, mm_embeds = outputs[:3]
    n_items = model.n_items

    mm_items = mm_embeds[-n_items:]

    id_trained = model.item_id_embedding.weight

    return {"MM": mm_items, "ID": id_trained}


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

    # Pretrain (random initial) snapshot BEFORE loading checkpoint
    pretrain = _extract_pretrain_embeddings(model)

    checkpoint = torch.load(args.checkpoint_path, map_location=config["device"])
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    origin = _extract_origin_embeddings(model)

    visualize_pretrain_vs_origin(
        pretrain,
        origin,
        sample_size=args.sample_size,
        title=args.title,
        filename=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ANCHORREC")
    parser.add_argument("--dataset", type=str, default="baby")
    parser.add_argument("--checkpoint_path", type=str, default="./saved/ANCHORREC_baby_best.pth")
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
        default="t-SNE: pretrain vs origin + ID (MM/ID)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_pretrain_origin_alignrec_0202.png",
    )
    args = parser.parse_args()

    main(args)
