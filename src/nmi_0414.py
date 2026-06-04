import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed, get_model


def get_common_sample_indices(n_items, sample_size=5000, random_state=42):
    sample_size = min(sample_size, n_items)
    rng = np.random.RandomState(random_state)
    indices = rng.choice(n_items, size=sample_size, replace=False)
    return indices


def cluster_embedding_with_kmeans(
    emb_np,
    indices,
    n_clusters=400,
    random_state=42,
):
    """
    emb_np: (N, D)
    indices: 공통 샘플 인덱스
    """
    emb_sample = emb_np[indices]

    n_clusters = min(n_clusters, len(emb_sample) - 1)
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(emb_sample)

    return cluster_labels


def compute_pairwise_nmi(cluster_dict):
    """
    cluster_dict: {
        "ID": labels_id,
        "MM": labels_mm,
        "Final": labels_final
    }
    """
    names = list(cluster_dict.keys())
    results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]

            labels_a = cluster_dict[name_a]
            labels_b = cluster_dict[name_b]

            nmi = normalized_mutual_info_score(labels_a, labels_b)

            results.append({
                "A": name_a,
                "B": name_b,
                "NMI": nmi,
            })

    return results


def print_nmi_results_table(results):
    print("\n" + "=" * 80)
    print(f"{'Embedding A':<15} {'Embedding B':<15} {'NMI':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['A']:<15} {r['B']:<15} {r['NMI']:<12.6f}")
    print("=" * 80)


def interpret_nmi_results(results):
    """
    Final-ID vs Final-MM 비교 중심 간단 해석
    """
    nmi_final_id = None
    nmi_final_mm = None

    for r in results:
        pair = {r["A"], r["B"]}
        if pair == {"Final", "ID"}:
            nmi_final_id = r["NMI"]
        elif pair == {"Final", "MM"}:
            nmi_final_mm = r["NMI"]

    print("\n[Interpretation]")
    if nmi_final_id is not None and nmi_final_mm is not None:
        print(f"NMI(Final, ID) = {nmi_final_id:.6f}")
        print(f"NMI(Final, MM) = {nmi_final_mm:.6f}")

        if nmi_final_id > nmi_final_mm:
            print("-> Final embedding이 MM보다 ID clustering 구조를 더 많이 따릅니다.")
        elif nmi_final_id < nmi_final_mm:
            print("-> Final embedding이 ID보다 MM clustering 구조를 더 많이 따릅니다.")
        else:
            print("-> Final embedding이 ID와 MM 구조를 비슷한 정도로 따릅니다.")
    else:
        print("-> Final-ID 또는 Final-MM NMI를 찾지 못했습니다.")


def compute_nmi_same_condition(
    id_items,
    content_items,
    final_items,
    n_clusters=10,
    sample_size=5000,
    random_state=42,
):
    """
    같은 조건:
    - 같은 샘플 인덱스
    - 같은 n_clusters
    - 같은 random_state
    """
    id_np = id_items.detach().cpu().numpy()
    mm_np = content_items.detach().cpu().numpy()
    final_np = final_items.detach().cpu().numpy()

    assert id_np.shape[0] == mm_np.shape[0] == final_np.shape[0], "아이템 개수가 다릅니다."

    n_items = id_np.shape[0]
    indices = get_common_sample_indices(
        n_items=n_items,
        sample_size=sample_size,
        random_state=random_state
    )

    labels_id = cluster_embedding_with_kmeans(
        emb_np=id_np,
        indices=indices,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    labels_mm = cluster_embedding_with_kmeans(
        emb_np=mm_np,
        indices=indices,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    labels_final = cluster_embedding_with_kmeans(
        emb_np=final_np,
        indices=indices,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    cluster_dict = {
        "ID": labels_id,
        "MM": labels_mm,
        "Final": labels_final,
    }

    results = compute_pairwise_nmi(cluster_dict)

    return results, cluster_dict, indices


def run_nmi_evaluation(
    model_name,
    checkpoint_path,
    n_clusters=10,
    sample_size=5000,
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

    # ----- 2) 임베딩 로드 -----
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
            # 필요 시 train=False로 바꿔도 됨
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

    # ----- 3) 같은 조건에서 NMI 계산 -----
    print("=" * 80)
    print("NMI Calculation under the same clustering condition")
    print(f"- n_clusters: {n_clusters}")
    print(f"- sample_size: {sample_size}")
    print(f"- random_state: 42")
    print("=" * 80)

    results, cluster_dict, indices = compute_nmi_same_condition(
        id_items=id_items,
        content_items=content_items,
        final_items=final_items,
        n_clusters=n_clusters,
        sample_size=sample_size,
        random_state=42,
    )

    print_nmi_results_table(results)
    interpret_nmi_results(results)

    return results, cluster_dict, indices


if __name__ == '__main__':
    models = [
        ("ALIGNREC", "saved/ALIGNREC_best.pth"),
    ]

    for model_name, ckpt_path in models:
        run_nmi_evaluation(
            model_name=model_name,
            checkpoint_path=ckpt_path,
            n_clusters=400,
            sample_size=70000,
        )