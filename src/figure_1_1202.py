import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed, get_model
import random

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def visualize_alignment(
    content_embeds_items,
    id_embeds,
    sample_size=500,
    lines_to_draw=10,
    title="",
    filename="tsne_result_1110.png",
):
    """ID vs Content 2-view t-SNE (기본 버전)"""
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

    plt.scatter(
        content_2d[:, 0], content_2d[:, 1],
        color='skyblue', label='Content', alpha=0.8, marker='D', s=6
    )
    plt.scatter(
        id_2d[:, 0], id_2d[:, 1],
        color='lightcoral', label='ID', alpha=0.8, marker='o', s=6
    )

    for i in top_k_indices:
        plt.plot(
            [content_2d[i, 0], id_2d[i, 0]],
            [content_2d[i, 1], id_2d[i, 1]],
            'k--', linewidth=1.0
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_alignment_3views(
    id_embeds,
    content_embeds_items,
    final_embeds_items,
    sample_size=5000,
    lines_to_draw=5000,
    title="",
    filename="tsne_result_3views.png",
    max_pair_distance=3.0,
):
    """
    ID / Content / Final 3-view t-SNE.
    위: 전체 t-SNE (원본)
    아래: 위에서 선택한 zoom 영역만 동일 크기로 확대해서 표시
    """
    # ===== 1. tensor -> numpy =====
    id_np = id_embeds.detach().cpu().numpy()
    content_np = content_embeds_items.detach().cpu().numpy()
    final_np = final_embeds_items.detach().cpu().numpy()

    assert id_np.shape[0] == content_np.shape[0] == final_np.shape[0], "아이템 개수가 다릅니다"
    assert id_np.shape[1] == content_np.shape[1] == final_np.shape[1], "임베딩 차원이 다릅니다"

    n_items = id_np.shape[0]
    sample_size = min(sample_size, n_items)

    # ===== 2. 샘플링 =====
    indices = np.random.choice(n_items, size=sample_size, replace=False)
    id_sample = id_np[indices]
    content_sample = content_np[indices]
    final_sample = final_np[indices]

    # ===== 3. t-SNE =====
    all_data = np.concatenate([id_sample, content_sample, final_sample], axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_data)

    id_2d = tsne_result[0:sample_size]
    content_2d = tsne_result[sample_size:2 * sample_size]
    final_2d = tsne_result[2 * sample_size:3 * sample_size]

    # ===== 4. final과의 거리 기반으로 선을 그릴 index 선택 =====
    dist_if = np.linalg.norm(id_2d - final_2d, axis=1)
    dist_cf = np.linalg.norm(content_2d - final_2d, axis=1)
    score = dist_if + dist_cf

    mask_pair = (dist_if < max_pair_distance) & (dist_cf < max_pair_distance)
    valid_idx = np.where(mask_pair)[0]

    if len(valid_idx) == 0:
        top_k_indices = np.array([], dtype=int)
    else:
        sorted_valid = valid_idx[np.argsort(score[valid_idx])]
        lines_to_draw = min(lines_to_draw, len(sorted_valid))
        top_k_indices = sorted_valid[:lines_to_draw]

    # ===== 5. zoom 위치: final_2d 에서 선택한 클러스터 중심부 =====
    n_clusters = min(4, sample_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(final_2d)
    counts = np.bincount(labels)
    densest_label = counts.argmax()

    center = kmeans.cluster_centers_[densest_label]
    # 필요하면 수동으로 덮어쓰기 (지금처럼 고정 좌표 쓰고 싶을 때)
    center_x, center_y = -8, 82

    cluster_points = final_2d[labels == densest_label]
    if len(cluster_points) > 0:
        dists_cluster = np.linalg.norm(cluster_points - center, axis=1)
        base_r = np.quantile(dists_cluster, 0.4)
        zoom_r = np.clip(base_r * 1.5, 1.0, 12.0)
    else:
        zoom_r = 6.0

    print(f"center_x : {center_x}, center_y : {center_y}, zoom_r : {zoom_r}")

    zoom_mask = (
        (final_2d[:, 0] >= center_x - zoom_r) &
        (final_2d[:, 0] <= center_x + zoom_r) &
        (final_2d[:, 1] >= center_y - zoom_r) &
        (final_2d[:, 1] <= center_y + zoom_r)
    )

    # ===== 6. 상하 2행 figure 생성 =====
    # 위: 전체, 아래: zoom 영역
    fig, (ax_main, ax_zoom) = plt.subplots(
        2, 1,
        figsize=(10, 8),
        gridspec_kw={'height_ratios': [1.2, 2.0], 'hspace': 0.08}
    )

    # ===== 6-1. 메인 그림 (전체 t-SNE) =====
    ax_main.set_title(title)

    scat_id = ax_main.scatter(
        id_2d[:, 0], id_2d[:, 1],
        label='ID', alpha=0.7, marker='o', s=4
    )
    scat_mm = ax_main.scatter(
        content_2d[:, 0], content_2d[:, 1],
        label='MM', alpha=0.7, marker='s', s=6
    )
    scat_final = ax_main.scatter(
        final_2d[:, 0], final_2d[:, 1],
        label='Final', alpha=0.9, marker='*', s=8
    )

    # ID–Final, MM–Final 선
    for i in top_k_indices:
        ax_main.plot(
            [id_2d[i, 0], final_2d[i, 0]],
            [id_2d[i, 1], final_2d[i, 1]],
            linestyle='-',
            linewidth=0.8,
            color='black',
            alpha=0.7,
            zorder=5,
        )
        ax_main.plot(
            [content_2d[i, 0], final_2d[i, 0]],
            [content_2d[i, 1], final_2d[i, 1]],
            linestyle='--',
            linewidth=0.8,
            color='red',
            alpha=0.7,
            zorder=4,
        )

    # 메인 플롯에 zoom 영역을 원으로 표시
    circle_main = mpatches.Circle(
        (center_x, center_y),
        zoom_r,
        edgecolor='black',
        facecolor='none',
        linewidth=2,
        linestyle='-',
        alpha=0.9,
    )
    ax_main.add_patch(circle_main)

    # legend용 line proxy
    line_id_final = Line2D(
        [0], [0],
        linestyle='-',
        color='black',
        label='ID–Final Pair'
    )
    line_mm_final = Line2D(
        [0], [0],
        linestyle='--',
        color='red',
        label='MM–Final Pair'
    )

    ax_main.legend(
        handles=[scat_id, scat_mm, scat_final, line_id_final, line_mm_final],
        fontsize=10,
        markerscale=3,
        loc='upper right'
    )

    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # ===== 6-2. 아래 zoom 축 =====
    ax_zoom.set_title("Zoom-in of highlighted region", fontsize=11)

    # 전체 점을 다시 그리고, x/y 범위로 확대 효과
    scat1 = ax_zoom.scatter(
        id_2d[:, 0], id_2d[:, 1],
        alpha=0.7, marker='o', s=64
    )
    scat2 = ax_zoom.scatter(
        content_2d[:, 0], content_2d[:, 1],
        alpha=0.7, marker='s', s=64
    )
    scat3 = ax_zoom.scatter(
        final_2d[:, 0], final_2d[:, 1],
        alpha=0.9, marker='*', s=88
    )

    ax_zoom.set_xlim(center_x - zoom_r, center_x + zoom_r)
    ax_zoom.set_ylim(center_y - zoom_r, center_y + zoom_r)

    # zoom 영역 안에 있는 pair들만 선을 그림
    for i in top_k_indices:
        if not zoom_mask[i]:
            continue
        ax_zoom.plot(
            [id_2d[i, 0], final_2d[i, 0]],
            [id_2d[i, 1], final_2d[i, 1]],
            linestyle='-',
            linewidth=1.4,
            color='black',
            alpha=0.8,
            zorder=5,
        )
        ax_zoom.plot(
            [content_2d[i, 0], final_2d[i, 0]],
            [content_2d[i, 1], final_2d[i, 1]],
            linestyle='--',
            linewidth=1.4,
            color='red',
            alpha=0.8,
            zorder=4,
        )

    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def run_tsne_visualization(model_name, checkpoint_path, lines_to_draw):
    config_dict = {
        'multimodal_data_dir': 'data/baby_beit3_128token_add_title_brand_to_text/',
        'save_model': False,
        'side_emb_div': 2,
        'valid_metric': 'Recall@20',
        'topk': [20],
    }
    config = Config(model_name, 'baby', config_dict)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for key in ['knn_k', 'seed', 'sim_weight', 'lambda_weight', 'learning_rate', 'n_layers']:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][0]

    # ----- 1) 캐시 디렉토리 & 파일 경로 -----
    cache_dir = "cache_tsne"
    os.makedirs(cache_dir, exist_ok=True)

    # 모델 이름 + 데이터셋 기준으로 캐시 파일 이름 생성
    emb_cache_path = os.path.join(
        cache_dir,
        f"{model_name}_baby_embeddings.pt"
    )

    # ----- 2) 임베딩 캐시가 있으면 바로 로드 -----
    if os.path.exists(emb_cache_path):
        print(f"[INFO] Load cached embeddings from {emb_cache_path}")
        ckpt = torch.load(emb_cache_path, map_location=config['device'])
        id_items = ckpt['id_items'].to(config['device'])
        content_items = ckpt['content_items'].to(config['device'])
        final_items = ckpt['final_items'].to(config['device'])

    else:
        print(f"[INFO] No cache. Compute embeddings and save to {emb_cache_path}")

        init_seed(config['seed'])

        raw_dataset = RecDataset(config)
        raw_dataset.inter_num = len(raw_dataset)
        raw_dataset.user_num = raw_dataset.get_user_num()
        raw_dataset.item_num = raw_dataset.get_item_num()
        dataloader = TrainDataLoader(config, raw_dataset)

        model_cls = get_model(config['model'])
        model = model_cls(config, dataloader).to(config['device'])

        checkpoint = torch.load(checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        model.eval()

        norm_adj = model.norm_adj
        n_users = model.n_users
        n_items = model.n_items

        with torch.no_grad():
            all_users, all_items, mm_embeds, content_embeds = model.forward(norm_adj, train=True)
            _, content_items = torch.split(content_embeds, [n_users, n_items], dim=0)

            final_items = all_items
            id_items = model.item_id_embedding.weight

        # CPU로 옮겨서 저장 (GPU 텐서는 저장/로드 시 골치 아픔)
        torch.save(
            {
                'id_items': id_items.cpu(),
                'content_items': content_items.cpu(),
                'final_items': final_items.cpu(),
            },
            emb_cache_path
        )

        # 다시 device로
        id_items = id_items.to(config['device'])
        content_items = content_items.to(config['device'])
        final_items = final_items.to(config['device'])

    # ----- 3) t-SNE 그리기 -----
    filename = f"tsne_{model_name}_3view_top{lines_to_draw}_1201.png"
    visualize_alignment_3views(
        id_items,
        content_items,
        final_items,
        lines_to_draw=lines_to_draw,
        filename=filename,
    )

if __name__ == '__main__':
    models = [
        ("ALIGNREC", "saved/ALIGNREC_best.pth"),
    ]
    for model_name, ckpt_path in models:
        for k in [5000]:
            run_tsne_visualization(model_name, ckpt_path, k)