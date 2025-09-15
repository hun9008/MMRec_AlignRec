import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colors = [
    (1.0, 0.0, 0.0),   # Red
    (1.0, 1.0, 0.0),   # Yellow
    (0.5, 1.0, 0.5),   # Light Green
    (0.5, 1.0, 1.0),   # Light Cyan
    (0.0, 0.0, 1.0),   # Blue
    (1.0, 0.0, 1.0),   # Magenta
    (1.0, 0.0, 0.0)    # Red again
]
custom_cmap = LinearSegmentedColormap.from_list("custom_sim_cmap", colors, N=256)

target = "final_emb"

# emb_path = "../saved_emb/0907_all_anchor/item_emb_align_mm.npy"
emb_path = "../saved_emb/0907_all_anchor/item_emb_final_alignrec.npy"
# emb_path = "./saved_emb/item_emb_alignrec_anchor_final.npy"
# emb_path = "./saved_emb/item_emb_alignrec_anchor_projection_aligned.npy"
# emb_path = "./saved_emb/item_emb_alignrec_final.npy"
item_emb = np.load(emb_path)
n_items = item_emb.shape[0]

np.random.seed(42)
sample_indices = np.random.choice(n_items, n_items // 2, replace=False)
sampled_emb = item_emb[sample_indices]

sim_matrix = cosine_similarity(sampled_emb)

mask = np.triu(np.ones_like(sim_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    sim_matrix,
    mask=mask,
    cmap=custom_cmap,
    square=True,
    cbar=True,
    xticklabels=200,
    yticklabels=200,
    vmin=-1, vmax=1,
    ax=ax
)

zoom_start_i = sim_matrix.shape[0] - 150 
zoom_end_i = sim_matrix.shape[0] - 100
zoom_start_j = 100
zoom_end_j = 150
zoom_data = sim_matrix[zoom_start_i:zoom_end_i, zoom_start_j:zoom_end_j]

axins = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=1.2)
sns.heatmap(
    zoom_data,
    cmap=custom_cmap,
    square=True,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
    vmin=-1, vmax=1,
    ax=axins
)

rect = plt.Rectangle((zoom_start_j, zoom_start_i),
                     zoom_end_j - zoom_start_j,
                     zoom_end_i - zoom_start_i,
                     edgecolor='red', facecolor='none', lw=2)
ax.add_patch(rect)

rect_center_x = (zoom_start_j + zoom_end_j) / 2
rect_center_y = (zoom_start_i + zoom_end_i) / 2

ax.annotate("", xy=(0.5, 0.5), xycoords=axins.transAxes,
            xytext=(rect_center_x, rect_center_y), textcoords='data',
            arrowprops=dict(arrowstyle="-", color='red', lw=1.5))

ax.set_title(f"[{target}] Item–Item Cosine Similarity Heatmap", fontsize=14)
# ax.set_title("Item–Item Cosine Similarity Heatmap [item_emb_alignrec_anchor_projection_aligned]", fontsize=14)
# ax.set_title("Item–Item Cosine Similarity Heatmap [item_emb_alignrec_final]", fontsize=14)
ax.set_xlabel("Item", fontsize=12)
ax.set_ylabel("Item", fontsize=12)
plt.tight_layout()
plt.savefig(f"item_emb_alignrec_{target}.png", dpi=100)
# plt.savefig("item_emb_alignrec_anchor_projection_aligned.png", dpi=300)
# plt.savefig("item_emb_alignrec_final.png", dpi=300)
# plt.show()
