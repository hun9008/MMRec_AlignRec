import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from models.alignrec import ALIGNREC
from data import create_dataset
from utils.utils import init_seed, get_model, get_trainer, dict2str
from utils.configurator import Config

def visualize_alignment(content_embeds_items, id_embeds):
    content_np = content_embeds_items.detach().cpu().numpy()
    id_np = id_embeds.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    all_data = np.concatenate([content_np, id_np], axis=0)
    tsne_result = tsne.fit_transform(all_data)
    content_2d = tsne_result[:len(content_np)]
    id_2d = tsne_result[len(content_np):]

    plt.figure(figsize=(10, 5))
    plt.title("t-SNE of Content-ID Feature Alignment")
    plt.scatter(content_2d[:, 0], content_2d[:, 1], color='blue', label='Content', alpha=0.5)
    plt.scatter(id_2d[:, 0], id_2d[:, 1], color='red', label='ID', alpha=0.5)

    for i in range(len(content_2d)):
        plt.plot([content_2d[i, 0], id_2d[i, 0]], [content_2d[i, 1], id_2d[i, 1]], 'k--', linewidth=0.3)

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ALIGNREC')
    parser.add_argument('--dataset', type=str, default='baby')
    parser.add_argument('--config_files', type=str, default='src/configs/overall.yaml')
    parser.add_argument('--multimodal_data_dir', type=str, default='data/beit3_128token_add_title_brand_to_text/')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to saved model .pth file')
    args = parser.parse_args()

    # 설정 및 데이터셋 로드
    config = Config(model=args.model, dataset=args.dataset, config_file_list=[args.config_files])
    config['multimodal_data_dir'] = args.multimodal_data_dir
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_seed(config['seed'], config['reproducibility'])
    dataset = RecDataset(config)
    model = ALIGNREC(config, dataset).to(config['device'])

    # 체크포인트 불러오기
    checkpoint = torch.load(args.checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    # 임베딩 추출 및 시각화
    norm_adj = model.norm_adj
    n_users = model.n_users
    n_items = model.n_items

    with torch.no_grad():
        ua_embeddings, ia_embeddings, side_embeds, content_embeds = model.forward(norm_adj)
        _, content_embeds_items = torch.split(content_embeds, [n_users, n_items], dim=0)
        id_embeds = model.item_id_embedding.weight

    visualize_alignment(content_embeds_items, id_embeds)