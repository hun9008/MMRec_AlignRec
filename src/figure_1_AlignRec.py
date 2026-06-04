import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from utils.configurator import Config
from utils.dataloader import TrainDataLoader
from utils.dataset import RecDataset
from utils.utils import get_model, init_seed


def load_alignrec_embeddings(args, device):
    config_dict = {
        "multimodal_data_dir": args.multimodal_data_dir,
        "save_model": False,
        "side_emb_div": 2,
        "valid_metric": "Recall@20",
        "topk": [20],
        "use_gpu": torch.cuda.is_available(),
    }
    config = Config(args.model, args.dataset, config_dict)
    config["device"] = device

    for key in ["knn_k", "seed", "sim_weight", "lambda_weight", "learning_rate", "n_layers"]:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][0]

    cache_dir = "cache_tsne"
    os.makedirs(cache_dir, exist_ok=True)
    emb_cache_path = os.path.join(cache_dir, f"{args.model}_{args.dataset}_embeddings.pt")

    if os.path.exists(emb_cache_path) and not args.no_cache:
        print(f"[INFO] Load cached embeddings from {emb_cache_path}")
        ckpt = torch.load(emb_cache_path, map_location=device)
        return (
            ckpt["id_items"].to(device),
            ckpt["content_items"].to(device),
            ckpt["final_items"].to(device),
        )

    print(f"[INFO] Compute embeddings and save to {emb_cache_path}")
    init_seed(config["seed"])

    raw_dataset = RecDataset(config)
    raw_dataset.inter_num = len(raw_dataset)
    raw_dataset.user_num = raw_dataset.get_user_num()
    raw_dataset.item_num = raw_dataset.get_item_num()
    dataloader = TrainDataLoader(config, raw_dataset)

    model_cls = get_model(config["model"])
    model = model_cls(config, dataloader).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    with torch.no_grad():
        _, all_items, _, content_embeds = model.forward(model.norm_adj, train=True)
        _, content_items = torch.split(content_embeds, [model.n_users, model.n_items], dim=0)
        final_items = all_items
        id_items = model.item_id_embedding.weight

    torch.save(
        {
            "id_items": id_items.cpu(),
            "content_items": content_items.cpu(),
            "final_items": final_items.cpu(),
        },
        emb_cache_path,
    )

    return id_items.to(device), content_items.to(device), final_items.to(device)


def _save_tsne_plot(id_2d, final_2d, output, show_legend=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    scat_final = ax.scatter(
        final_2d[:, 0],
        final_2d[:, 1],
        label="Final",
        alpha=0.7,
        marker="*",
        s=18,
        edgecolors="none",
        linewidths=0,
        color="red",
    )
    scat_id = ax.scatter(
        id_2d[:, 0],
        id_2d[:, 1],
        label="ID",
        alpha=0.7,
        marker="o",
        s=10,
        edgecolors="none",
        linewidths=0,
        color="#2E4358",
    )

    if show_legend:
        legend_marker_size = np.sqrt(2.5) * 8
        legend_id = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=scat_id.get_facecolor()[0],
            markeredgecolor="none",
            markersize=legend_marker_size,
            label="ID",
        )
        legend_final = Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor=scat_final.get_facecolor()[0],
            markeredgecolor="none",
            markersize=legend_marker_size,
            label="Final",
        )
        ax.legend(handles=[legend_id, legend_final], fontsize=20, loc="lower left")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_top_tsne_only(id_items, final_items, output, no_legend_output, sample_size=5000):
    id_np = id_items.detach().cpu().numpy()
    final_np = final_items.detach().cpu().numpy()

    if id_np.shape != final_np.shape:
        raise ValueError(f"Embedding shape mismatch: id={id_np.shape}, final={final_np.shape}")

    n_items = id_np.shape[0]
    sample_size = min(sample_size, n_items)
    rng = np.random.RandomState(42)
    indices = rng.choice(n_items, size=sample_size, replace=False)

    id_sample = id_np[indices]
    final_sample = final_np[indices]

    all_data = np.concatenate([id_sample, final_sample], axis=0)
    tsne_result = TSNE(n_components=2, random_state=42).fit_transform(all_data)
    id_2d = tsne_result[:sample_size]
    final_2d = tsne_result[sample_size:]

    _save_tsne_plot(id_2d, final_2d, output, show_legend=True)
    _save_tsne_plot(id_2d, final_2d, no_legend_output, show_legend=False)


def main():
    parser = argparse.ArgumentParser(description="Save the top t-SNE view for AlignRec only.")
    parser.add_argument("--model", default="ALIGNREC")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--checkpoint", default="saved/ALIGNREC_best.pth")
    parser.add_argument("--output", default="figure1_AlignRec.png")
    parser.add_argument("--no_legend_output", default="figure1_AlignRec_no_regend.png")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument(
        "--multimodal_data_dir",
        default="data/baby_beit3_128token_add_title_brand_to_text/",
    )
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_items, _, final_items = load_alignrec_embeddings(args, device)
    plot_top_tsne_only(
        id_items,
        final_items,
        args.output,
        args.no_legend_output,
        sample_size=args.sample_size,
    )
    print(f"[SAVE] {args.output}")
    print(f"[SAVE] {args.no_legend_output}")


if __name__ == "__main__":
    main()
