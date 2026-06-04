from figure_5_top_k_baselines import generate_for_models
from numpy import dtype
from pathlib import Path


if __name__ == "__main__":
    generate_for_models(
        models=["LATTICE"],
        saved_emb_dir=Path("saved_emb"),
        out_prefix="figure_5",
        text_feat=Path("/home/hun/data/data/baby/text_feat.npy"),
        vision_feat=Path("/home/hun/data/data/baby/image_feat.npy"),
        interactions=Path("/home/hun/data/data/baby/baby.inter"),
        item_mapping=Path("/home/hun/data/data/baby/i_id_mapping.csv"),
        topk=3,
        block_size=1024,
        dtype=dtype("float32"),
    )
