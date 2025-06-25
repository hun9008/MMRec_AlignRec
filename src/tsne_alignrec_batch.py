import os
import sys
import argparse

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
import random


def visualize_alignment(content_embeds_items, id_embeds, sample_size=500, lines_to_draw=10, title="", filename="tsne_result.png"):
    content_np = content_embeds_items.detach().cpu().numpy()
    id_np = id_embeds.detach().cpu().numpy()
    assert content_np.shape[0] == id_np.shape[0], "Embedding 개수가 다릅니다"

    indices = random.sample(range(content_np.shape[0]), sample_size)
    content_sample = content_np[indices]
    id_sample = id_np[indices]

    tsne = TSNE(n_components=2, random_state=42)
    all_data = np.concatenate([content_sample, id_sample], axis=0)
    tsne_result = tsne.fit_transform(all_data)
    content_2d = tsne_result[:sample_size]
    id_2d = tsne_result[sample_size:]

    pair_distances = np.linalg.norm(content_2d - id_2d, axis=1)
    top_k_indices = np.argsort(pair_distances)[:lines_to_draw]

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.scatter(content_2d[:, 0], content_2d[:, 1], color='skyblue', label='Content', alpha=0.8, marker='D')
    plt.scatter(id_2d[:, 0], id_2d[:, 1], color='lightcoral', label='ID', alpha=0.8, marker='o')

    for i in top_k_indices:
        plt.plot([content_2d[i, 0], id_2d[i, 0]], [content_2d[i, 1], id_2d[i, 1]], 'k--', linewidth=1.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_tsne_visualization(model_name, checkpoint_path, lines_to_draw):
    config_dict = {
        'multimodal_data_dir': 'data/beit3_128token_add_title_brand_to_text/',
        'save_model': False,
        'side_emb_div': 2
    }
    config = Config(model_name, 'baby', config_dict)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for key in ['knn_k', 'seed', 'sim_weight', 'lambda_weight', 'learning_rate', 'n_layers']:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][0]

    init_seed(config['seed'])

    raw_dataset = RecDataset(config)
    raw_dataset.inter_num = len(raw_dataset)
    raw_dataset.user_num = raw_dataset.get_user_num()
    raw_dataset.item_num = raw_dataset.get_item_num()
    dataloader = TrainDataLoader(config, raw_dataset)

    model_cls = get_model(config['model'])
    model = model_cls(config, dataloader).to(config['device'])
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    norm_adj = model.norm_adj
    n_users = model.n_users
    n_items = model.n_items

    with torch.no_grad():
        content_embeds, _ = model.forward(norm_adj)
        content_embeds_items = content_embeds[-n_items:]
        id_embeds = model.item_id_embedding.weight

    filename = f"tsne_{model_name}_top{lines_to_draw}.png"
    title = f"t-SNE: {model_name}, Top {lines_to_draw} Closest Pairs"
    visualize_alignment(content_embeds_items, id_embeds, lines_to_draw=lines_to_draw, title=title, filename=filename)


if __name__ == '__main__':
    models = [
        ("ALIGNREC", "saved/ALIGNREC_best.pth"),
        ("ALIGNREC_INPUT_CL_ANCHOR_MMTV_0513", "saved/ALIGNREC_INPUT_CL_ANCHOR_MMTV_0513_best.pth")
    ]
    for model_name, ckpt_path in models:
        for k in [10, 100]:
            run_tsne_visualization(model_name, ckpt_path, k)