import os
import sys

# 프로젝트 루트 경로를 PYTHONPATH에 추가
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

def visualize_alignment(content_embeds_items, id_embeds, sample_size=500, lines_to_draw=10, title="t-SNE of Content-ID Feature Alignment"):

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

    # 거리 계산: content[i]와 id[i]의 유클리드 거리
    pair_distances = np.linalg.norm(content_2d - id_2d, axis=1)
    top_k_indices = np.argsort(pair_distances)[:lines_to_draw]

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.scatter(content_2d[:, 0], content_2d[:, 1], color='skyblue', label='Content', alpha=0.8, marker='D')
    plt.scatter(id_2d[:, 0], id_2d[:, 1], color='lightcoral', label='ID', alpha=0.8, marker='o')

    for i in top_k_indices:
        plt.plot([content_2d[i, 0], id_2d[i, 0]], [content_2d[i, 1], id_2d[i, 1]], 'k--', linewidth=1.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig("tsne_top10_projection_0625.png", dpi=300)
    plt.close()


def main(args):
    # config 세팅
    config_dict = {
        'multimodal_data_dir': args.multimodal_data_dir,
        'save_model': False,
        'side_emb_div': 2  # ← 기본적으로 자주 쓰이는 값
    }
    config = Config(args.model, args.dataset, config_dict)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 리스트형 파라미터 방지
    for key in ['knn_k', 'seed', 'sim_weight', 'lambda_weight', 'learning_rate', 'n_layers']:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][0]

    init_seed(config['seed'])

    # 데이터셋 로드 및 필드 보강
    raw_dataset = RecDataset(config)
    raw_dataset.inter_num = len(raw_dataset)
    raw_dataset.user_num = raw_dataset.get_user_num()
    raw_dataset.item_num = raw_dataset.get_item_num()

    dataloader = TrainDataLoader(config, raw_dataset)

    # 모델 로드
    model_cls = get_model(config['model'])
    model = model_cls(config, dataloader).to(config['device'])

    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    # 임베딩 추출
    norm_adj = model.norm_adj
    n_users = model.n_users
    n_items = model.n_items

    with torch.no_grad():
        # train=True로 설정하여 mm_embeds 포함한 output 받음
        _, _, mm_embeds, content_embeds = model.forward(norm_adj, train=True)

        # 아이템 부분만 추출
        content_embeds_items = content_embeds[-n_items:]  # h_id^i
        mm_embeds_items = mm_embeds[-n_items:]            # h_mm^i

        # projection 수행: h_id_i_fusion, h_mm_i_fusion
        h_id_i_fusion = model.W_id_i(content_embeds_items)
        h_mm_i_fusion = model.W_mm_i(mm_embeds_items)

    # 시각화: projection 공간에서의 alignment
    visualize_alignment(h_id_i_fusion, h_mm_i_fusion, title="t-SNE of Projected ID-MM Alignment")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ALIGNREC_INPUT_CL_ANCHOR_MMTV_0513')
    parser.add_argument('--dataset', type=str, default='baby')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--multimodal_data_dir', type=str, default='data/beit3_128token_add_title_brand_to_text/')
    args = parser.parse_args()

    main(args)