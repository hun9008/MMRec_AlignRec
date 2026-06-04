import os
import sys
import argparse
import csv

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from sklearn.cluster import KMeans

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed, get_model


def sample_embedding(emb_np, sample_size=None, random_state=42):
    n_samples = emb_np.shape[0]

    if sample_size is not None and sample_size < n_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        return emb_np[idx]

    return emb_np


def estimate_elbow_k(k_values, inertias):
    """
    Elbow 지점을 자동 추정한다.
    첫 k와 마지막 k를 잇는 직선에서 가장 멀리 떨어진 k를 elbow로 선택한다.
    """
    points = np.column_stack([k_values, inertias]).astype(np.float64)

    if len(points) < 3:
        return int(k_values[0])

    start = points[0]
    end = points[-1]
    line_vec = end - start
    line_norm = np.linalg.norm(line_vec)

    if line_norm == 0:
        return int(k_values[0])

    distances = np.abs(
        line_vec[0] * (start[1] - points[:, 1])
        - line_vec[1] * (start[0] - points[:, 0])
    ) / line_norm
    return int(k_values[int(np.argmax(distances))])


def compute_elbow_for_embedding(
    emb_np,
    name="embedding",
    k_list=None,
    sample_size=None,
    random_state=42,
):
    """
    emb_np: (N, D) numpy array
    name: 출력용 이름
    k_list: elbow method에서 탐색할 KMeans cluster 개수 목록
    sample_size: elbow 계산에 사용할 샘플 수
    """
    if k_list is None:
        k_list = [50, 100, 200, 400, 800, 1200]

    emb_eval = sample_embedding(
        emb_np=emb_np,
        sample_size=sample_size,
        random_state=random_state,
    )

    max_valid_k = len(emb_eval) - 1
    k_values = [int(k) for k in k_list if 2 <= int(k) <= max_valid_k]

    if not k_values:
        raise ValueError(f"{name}: valid k list is empty. k_list={k_list}, max_valid_k={max_valid_k}")
    inertias = []

    print(f"\n[{name}] Elbow calculation")
    print(f"- samples: {len(emb_eval)}")
    print(f"- k list: {k_values}")

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )
        kmeans.fit(emb_eval)
        inertias.append(float(kmeans.inertia_))
        print(f"  k={k:<4} inertia={kmeans.inertia_:.6f}")

    elbow_k = estimate_elbow_k(k_values, inertias)
    print(f"[{name}] estimated elbow k: {elbow_k}")

    return {
        "name": name,
        "n_samples": len(emb_eval),
        "k_values": k_values,
        "inertias": inertias,
        "elbow_k": elbow_k,
    }


def compute_all_elbow_scores(
    id_items,
    content_items,
    final_items,
    k_list=None,
    sample_size=5000,
):
    """
    id_items, content_items, final_items: torch.Tensor (N, D)
    """
    id_np = id_items.detach().cpu().numpy()
    content_np = content_items.detach().cpu().numpy()
    final_np = final_items.detach().cpu().numpy()

    print("=" * 80)
    print("KMeans Elbow Method Calculation")
    print("NOTE: Elbow method uses KMeans inertia/WCSS in the ORIGINAL embedding space.")
    print("NOTE: Lower inertia is better, but choose the k where decrease starts to flatten.")
    print("=" * 80)

    results = []

    results.append(
        compute_elbow_for_embedding(
            emb_np=id_np,
            name="ID",
            k_list=k_list,
            sample_size=sample_size,
            random_state=42,
        )
    )

    results.append(
        compute_elbow_for_embedding(
            emb_np=content_np,
            name="MM",
            k_list=k_list,
            sample_size=sample_size,
            random_state=42,
        )
    )

    results.append(
        compute_elbow_for_embedding(
            emb_np=final_np,
            name="Final",
            k_list=k_list,
            sample_size=sample_size,
            random_state=42,
        )
    )

    return results


def print_results_table(results):
    print("\n" + "=" * 80)
    print(f"{'Name':<12} {'#Samples':<10} {'Estimated Elbow K':<20}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<12} "
            f"{r['n_samples']:<10} "
            f"{r['elbow_k']:<20}"
        )
    print("=" * 80)


def save_results_csv(results, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "n_samples", "k", "inertia", "estimated_elbow_k"])
        for r in results:
            for k, inertia in zip(r["k_values"], r["inertias"]):
                writer.writerow([r["name"], r["n_samples"], k, inertia, r["elbow_k"]])

    print(f"[INFO] Saved elbow results to {output_path}")


def save_elbow_plot(results, output_path):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.figure(figsize=(9, 6))
    for r in results:
        plt.plot(r["k_values"], r["inertias"], marker="o", label=f"{r['name']} (elbow={r['elbow_k']})")
        elbow_idx = r["k_values"].index(r["elbow_k"])
        plt.scatter(
            [r["elbow_k"]],
            [r["inertias"][elbow_idx]],
            s=80,
        )

    plt.xlabel("n_clusters (k)")
    plt.ylabel("Inertia / WCSS")
    plt.title("KMeans Elbow Method")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved elbow plot to {output_path}")


def run_elbow_evaluation(
    model_name,
    checkpoint_path,
    k_list=None,
    sample_size=5000,
    output_csv=None,
    output_plot=None,
):
    config_dict = {
        'multimodal_data_dir': 'data/baby_beit3_128token_add_title_brand_to_text/',
        'save_model': False,
        'side_emb_div': 2,
        'valid_metric': 'Recall@20',
        'topk': [20],
        'use_gpu': torch.cuda.is_available(),
    }
    config = Config(model_name, 'baby', config_dict)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for key in ['knn_k', 'seed', 'sim_weight', 'lambda_weight', 'learning_rate', 'n_layers']:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][0]

    # ----- 1) 캐시 디렉토리 & 파일 경로 -----
    cache_dir = "cache_tsne"
    os.makedirs(cache_dir, exist_ok=True)

    emb_cache_path = os.path.join(
        cache_dir,
        f"{model_name}_baby_embeddings.pt"
    )

    # ----- 2) 임베딩 캐시가 있으면 로드 -----
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
            all_users, all_items, mm_embeds, content_embeds, t_emb, v_emb = model.forward(norm_adj, train=True)
            _, content_items = torch.split(content_embeds, [n_users, n_items], dim=0)

            final_items = all_items
            id_items = model.item_id_embedding.weight

        torch.save(
            {
                'id_items': id_items.cpu(),
                'content_items': content_items.cpu(),
                'final_items': final_items.cpu(),
            },
            emb_cache_path
        )

        id_items = id_items.to(config['device'])
        content_items = content_items.to(config['device'])
        final_items = final_items.to(config['device'])

    # ----- 3) elbow 계산 -----
    results = compute_all_elbow_scores(
        id_items=id_items,
        content_items=content_items,
        final_items=final_items,
        k_list=k_list,
        sample_size=sample_size,
    )

    print_results_table(results)

    if output_csv is not None:
        save_results_csv(results, output_csv)

    if output_plot is not None:
        save_elbow_plot(results, output_plot)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--k_list',
        type=int,
        nargs='+',
        default=[50, 100, 200, 400, 800, 1200],
        help='KMeans cluster counts for elbow search.',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=70000,
        help='Sample size used for elbow calculation.',
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='cache_tsne/k_means_elbow_0428.csv',
        help='CSV path to save k/inertia results. Use an empty string to skip.',
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default='cache_tsne/k_means_elbow_0428.png',
        help='PNG path to save elbow plot. Use an empty string to skip.',
    )
    args = parser.parse_args()

    models = [
        ("ALIGNREC_ANCHOR_1101", "saved/ALIGNREC_ANCHOR_1101_baby_best.pth"),
    ]

    output_csv = args.output_csv if args.output_csv else None
    output_plot = args.output_plot if args.output_plot else None

    for model_name, ckpt_path in models:
        print(
            f"[INFO] k_list={args.k_list}, "
            f"sample_size={args.sample_size}"
        )
        run_elbow_evaluation(
            model_name=model_name,
            checkpoint_path=ckpt_path,
            k_list=args.k_list,
            sample_size=args.sample_size,
            output_csv=output_csv,
            output_plot=output_plot,
        )
