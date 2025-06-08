import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualize_alignment(content_embeds_items, id_embeds):
    """
    content_embeds_items: torch.Tensor of shape [n_items, d]
    id_embeds: torch.Tensor of shape [n_items, d]
    """
    content_np = content_embeds_items.detach().cpu().numpy()
    id_np = id_embeds.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    all_data = np.concatenate([content_np, id_np], axis=0)
    tsne_result = tsne.fit_transform(all_data)

    content_2d = tsne_result[:len(content_np)]
    id_2d = tsne_result[len(content_np):]

    plt.figure(figsize=(10, 5))
    plt.title("t-SNE of Content-ID Feature Alignment")
    plt.scatter(content_2d[:, 0], content_2d[:, 1], color='blue', label='Content Embedding', alpha=0.5)
    plt.scatter(id_2d[:, 0], id_2d[:, 1], color='red', label='ID Embedding', alpha=0.5)

    for i in range(len(content_2d)):
        plt.plot([content_2d[i, 0], id_2d[i, 0]], [content_2d[i, 1], id_2d[i, 1]], 'k--', linewidth=0.3)

    plt.legend()
    plt.tight_layout()
    plt.show()