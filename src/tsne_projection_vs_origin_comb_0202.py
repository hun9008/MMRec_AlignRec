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


def _project_text_vision(model, t_items, v_items, device):
    if hasattr(model, "W_t_i"):
        t_proj = model.W_t_i(t_items)
    elif hasattr(model, "W_t"):
        t_feat = getattr(model, "t_feat", None)
        if t_feat is None:
            t_proj = t_items
        else:
            t_proj = model.W_t(t_feat.to(device))
    else:
        t_proj = t_items

    if hasattr(model, "W_v_i"):
        v_proj = model.W_v_i(v_items)
    elif hasattr(model, "W_v"):
        v_feat = getattr(model, "v_feat", None)
        if v_feat is None:
            v_proj = v_items
        else:
            v_proj = model.W_v(v_feat.to(device))
    else:
        v_proj = v_items

    return t_proj, v_proj


def _extract_embeddings(model, norm_adj, device):
    with torch.no_grad():
        outputs = model.forward(norm_adj, train=True)

    if len(outputs) < 6:
        raise RuntimeError("model.forward must return mm/text/vision embeddings with train=True")

    _, _, mm_embeds, content_embeds, t_embeds, v_embeds = outputs[:6]

    n_items = model.n_items

    content_items = content_embeds[-n_items:]
    mm_items = mm_embeds[-n_items:]
    t_items = t_embeds[-n_items:]
    v_items = v_embeds[-n_items:]

    origin = {
        "ID": content_items,
        "MM": mm_items,
        "Text": t_items,
        "Vision": v_items,
    }

    if not hasattr(model, "W_id_i") or not hasattr(model, "W_mm_i"):
        raise RuntimeError("model must have W_id_i and W_mm_i for projection-space embeddings")

    id_proj = model.W_id_i(content_items)
    mm_proj = model.W_mm_i(mm_items)
    t_proj, v_proj = _project_text_vision(model, t_items, v_items, device)

    proj = {
        "ID": id_proj,
        "MM": mm_proj,
        "Text": t_proj,
        "Vision": v_proj,
    }

    return origin, proj


def _tsne_2d_combined(origin, proj, indices, random_state=42):
    keys = ["ID", "MM", "Text", "Vision"]
    arrays = []
    labels = []
    for k in keys:
        arrays.append(origin[k].detach().cpu().numpy()[indices])
        labels.append(f"Origin-{k}")
    for k in keys:
        arrays.append(proj[k].detach().cpu().numpy()[indices])
        labels.append(f"Proj-{k}")

    all_data = np.concatenate(arrays, axis=0)
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_result = tsne.fit_transform(all_data)

    sample_size = arrays[0].shape[0]
    out = {}
    for i, label in enumerate(labels):
        start = i * sample_size
        end = (i + 1) * sample_size
        out[label] = tsne_result[start:end]
    return out


def visualize_origin_vs_projection_combined(origin, proj, sample_size, title, filename, seed=42):
    n_items = origin["ID"].shape[0]
    sample_size = min(sample_size, n_items)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_items, size=sample_size, replace=False)

    tsne_2d = _tsne_2d_combined(origin, proj, indices, random_state=seed)

    # origin: red/orange/yellow tones, projection: blue/purple tones
    origin_colors = {
        "ID": "#ff4d4d",   # bright red-orange
        "MM": "#ffb000",   # deep yellow
        "Text": "#ffd966", # light yellow
        "Vision": "#ff8c42",  # bright orange
    }
    proj_colors = {
        "ID": "#6ec6ff",   # light sky blue
        "MM": "#1b3b6f",   # deep navy
        "Text": "#7b5ce1", # purple
        "Vision": "#2f7ed8",  # strong blue
    }
    markers = {
        "ID": "o",
        "MM": "s",
        "Text": "^",
        "Vision": "x",
    }

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.set_title("Origin + Projection (single t-SNE)")

    for k in ["ID", "MM", "Text", "Vision"]:
        pts = tsne_2d[f"Origin-{k}"]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=f"Origin-{k}",
            alpha=0.8,
            marker=markers[k],
            s=8,
            color=origin_colors[k],
        )

    for k in ["ID", "MM", "Text", "Vision"]:
        pts = tsne_2d[f"Proj-{k}"]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=f"Proj-{k}",
            alpha=0.8,
            marker=markers[k],
            s=8,
            color=proj_colors[k],
        )

    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])

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

    origin, proj = _extract_embeddings(model, model.norm_adj, config["device"])

    visualize_origin_vs_projection_combined(
        origin,
        proj,
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
        default="t-SNE: origin + projection (ID/MM/Text/Vision)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_origin_vs_projection_comb_0202.png",
    )
    args = parser.parse_args()

    main(args)
