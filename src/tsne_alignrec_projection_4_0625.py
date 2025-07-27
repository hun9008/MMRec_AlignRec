import os
import sys

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

import random


def visualize_multiple_alignments(h_id, h_mm, h_t, h_v, sample_size=300, title="t-SNE of Projected Embeddings"):
    h_id_np = h_id.detach().cpu().numpy()
    h_mm_np = h_mm.detach().cpu().numpy()
    h_t_np = h_t.detach().cpu().numpy()
    h_v_np = h_v.detach().cpu().numpy()

    assert h_id_np.shape[0] == h_mm_np.shape[0] == h_t_np.shape[0] == h_v_np.shape[0]

    indices = np.random.choice(h_id_np.shape[0], sample_size, replace=False)
    h_id_sample = h_id_np[indices]
    h_mm_sample = h_mm_np[indices]
    h_t_sample = h_t_np[indices]
    h_v_sample = h_v_np[indices]

    all_data = np.concatenate([h_id_sample, h_mm_sample, h_t_sample, h_v_sample], axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_data)

    id_2d = tsne_result[:sample_size]
    mm_2d = tsne_result[sample_size:sample_size*2]
    t_2d = tsne_result[sample_size*2:sample_size*3]
    v_2d = tsne_result[sample_size*3:]

    plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.scatter(id_2d[:, 0], id_2d[:, 1], label='ID', alpha=0.8, marker='o')
    plt.scatter(mm_2d[:, 0], mm_2d[:, 1], label='MM', alpha=0.8, marker='s')
    plt.scatter(t_2d[:, 0], t_2d[:, 1], label='Text', alpha=0.8, marker='^')
    plt.scatter(v_2d[:, 0], v_2d[:, 1], label='Vision', alpha=0.8, marker='x')

    plt.legend()
    plt.tight_layout()
    plt.savefig("tsne_projection_h_id_mm_t_v.png", dpi=300)
    plt.close()


def main(args):
    config_dict = {
        'multimodal_data_dir': args.multimodal_data_dir,
        'save_model': False,
        'side_emb_div': 2
    }
    config = Config(args.model, args.dataset, config_dict)
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

    checkpoint = torch.load(args.checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    norm_adj = model.norm_adj
    n_users = model.n_users
    n_items = model.n_items

    with torch.no_grad():
        # 변경: text_embeds, vision_embeds 도 받기
        _, _, mm_embeds, content_embeds, text_embeds, vision_embeds = model.forward(norm_adj, train=True)

        content_embeds_items = content_embeds[-n_items:]
        mm_embeds_items = mm_embeds[-n_items:]
        text_embeds_items = text_embeds[-n_items:]
        vision_embeds_items = vision_embeds[-n_items:]

        # projection
        h_id_i_fusion = model.W_id_i(content_embeds_items)
        h_mm_i_fusion = model.W_mm_i(mm_embeds_items)
        h_t_i_fusion = model.W_t_i(text_embeds_items)
        h_v_i_fusion = model.W_v_i(vision_embeds_items)

    visualize_multiple_alignments(
        h_id_i_fusion, h_mm_i_fusion, h_t_i_fusion, h_v_i_fusion,
        title="t-SNE of Projected ID, MM, Text, Vision"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ALIGNREC_INPUT_CL_ANCHOR_MMTV_0513')
    parser.add_argument('--dataset', type=str, default='baby')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--multimodal_data_dir', type=str, default='data/beit3_128token_add_title_brand_to_text/')
    args = parser.parse_args()

    main(args)