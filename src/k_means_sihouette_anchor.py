import os
import sys
import argparse

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed, get_model


def resolve_multimodal_data_dir(multimodal_data_dir):
    if os.path.isfile(os.path.join(multimodal_data_dir, "image_feat.npy")):
        return multimodal_data_dir

    basename = os.path.basename(os.path.normpath(multimodal_data_dir))
    candidates = [
        os.path.join("..", "data", "data", basename),
        os.path.join("..", "data", basename),
    ]

    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "image_feat.npy")):
            print(f"[INFO] Resolve multimodal_data_dir: {multimodal_data_dir} -> {candidate}")
            return candidate

    return multimodal_data_dir


def compute_silhouette_for_embedding(
    emb_np,
    name="embedding",
    n_clusters=10,
    sample_size=None,
    random_state=42,
    use_tsne_projection=False,
):
    """
    emb_np: (N, D) numpy array
    name: 출력용 이름
    n_clusters: KMeans cluster 개수
    sample_size: silhouette 계산에 사용할 샘플 수
    use_tsne_projection: False 권장. True면 입력이 이미 2D projection일 때만 참고용 사용
    """
    n_samples = emb_np.shape[0]

    if sample_size is not None and sample_size < n_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        emb_eval = emb_np[idx]
    else:
        emb_eval = emb_np

    n_clusters = min(n_clusters, len(emb_eval) - 1)
    if n_clusters < 2:
        raise ValueError(f"{name}: n_clusters must be >= 2")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(emb_eval)

    score = silhouette_score(emb_eval, cluster_labels, metric='euclidean')

    space_type = "t-SNE 2D projection" if use_tsne_projection else "original embedding space"
    print(f"[{name}] silhouette score on {space_type}: {score:.6f}")

    return {
        "name": name,
        "space_type": space_type,
        "n_samples": len(emb_eval),
        "n_clusters": n_clusters,
        "score": score,
    }


def compute_all_silhouette_scores(
    id_items,
    content_items,
    final_items,
    n_clusters=10,
    sample_size=5000,
):
    """
    id_items, content_items, final_items: torch.Tensor (N, D)
    """
    id_np = id_items.detach().cpu().numpy()
    content_np = content_items.detach().cpu().numpy()
    final_np = final_items.detach().cpu().numpy()

    print("=" * 80)
    print("ANCHORREC Silhouette Score Calculation")
    print("NOTE: silhouette score should be computed in the ORIGINAL embedding space.")
    print("=" * 80)

    results = []

    results.append(
        compute_silhouette_for_embedding(
            emb_np=id_np,
            name="ID",
            n_clusters=n_clusters,
            sample_size=sample_size,
            random_state=42,
            use_tsne_projection=False,
        )
    )

    results.append(
        compute_silhouette_for_embedding(
            emb_np=content_np,
            name="MM",
            n_clusters=n_clusters,
            sample_size=sample_size,
            random_state=42,
            use_tsne_projection=False,
        )
    )

    results.append(
        compute_silhouette_for_embedding(
            emb_np=final_np,
            name="Final",
            n_clusters=n_clusters,
            sample_size=sample_size,
            random_state=42,
            use_tsne_projection=False,
        )
    )

    return results


def print_results_table(results):
    print("\n" + "=" * 80)
    print(f"{'Name':<12} {'Space':<28} {'#Samples':<10} {'#Clusters':<10} {'Score':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<12} "
            f"{r['space_type']:<28} "
            f"{r['n_samples']:<10} "
            f"{r['n_clusters']:<10} "
            f"{r['score']:<10.6f}"
        )
    print("=" * 80)


def run_silhouette_evaluation(
    model_name,
    checkpoint_path,
    multimodal_data_dir='../data/data/baby_beit3_128token_add_title_brand_to_text/',
    n_clusters=10,
    sample_size=5000,
):
    config_dict = {
        'multimodal_data_dir': resolve_multimodal_data_dir(multimodal_data_dir),
        'save_model': False,
        'side_emb_div': 2,
        'valid_metric': 'Recall@20',
        'topk': [20],
        'use_gpu': torch.cuda.is_available(),
    }
    config = Config(model_name, 'baby', config_dict)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for key in [
        'knn_k',
        'seed',
        'sim_weight',
        'lambda_weight',
        'learning_rate',
        'n_layers',
        'aal_loss',
        'amp_loss',
        'tau_weight',
    ]:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][0]

    cache_dir = "cache_tsne"
    os.makedirs(cache_dir, exist_ok=True)

    emb_cache_path = os.path.join(
        cache_dir,
        f"{model_name}_baby_embeddings.pt"
    )

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
        state_dict = checkpoint.get('state_dict', checkpoint)
        load_info = model.load_state_dict(state_dict, strict=False)
        if load_info.missing_keys:
            print(f"[WARN] Missing keys while loading checkpoint: {load_info.missing_keys}")
        if load_info.unexpected_keys:
            print(f"[WARN] Unexpected keys ignored from checkpoint: {load_info.unexpected_keys}")
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

    results = compute_all_silhouette_scores(
        id_items=id_items,
        content_items=content_items,
        final_items=final_items,
        n_clusters=n_clusters,
        sample_size=sample_size,
    )

    print_results_table(results)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ANCHORREC")
    parser.add_argument("--checkpoint_path", type=str, default="saved/ANCHORREC_baby_best.pth")
    parser.add_argument(
        "--multimodal_data_dir",
        type=str,
        default="../data/data/baby_beit3_128token_add_title_brand_to_text/",
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=400,
        help='KMeans cluster count used for silhouette score calculation.',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=70000,
        help='Sample size used for silhouette score calculation.',
    )
    args = parser.parse_args()

    print(f"[INFO] n_clusters={args.n_clusters}, sample_size={args.sample_size}")
    run_silhouette_evaluation(
        model_name=args.model,
        checkpoint_path=args.checkpoint_path,
        multimodal_data_dir=args.multimodal_data_dir,
        n_clusters=args.n_clusters,
        sample_size=args.sample_size,
    )
