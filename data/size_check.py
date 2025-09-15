import numpy as np

files = [
    "./baby/text_feat.npy",
    "./baby/image_feat.npy",
    "./beit3_128token_add_title_brand_to_text/image_feat.npy"
]

image_feat = [
    "./baby/image_feat.npy",
    "../saved_emb/0907_all_anchor/item_feat_raw_vision.npy",
]

for f in files:
    arr = np.load(f)
    print(f"{f}: shape={arr.shape}, dtype={arr.dtype}")

## same check
arr1 = np.load(image_feat[0]).astype(np.float32)
arr2 = np.load(image_feat[1]).astype(np.float32)
print(np.allclose(arr1, arr2))   # True에 가까울 것
print(f"arr1.shape={arr1.shape}, arr2.shape={arr2.shape}")
print(f"arr1.dtype={arr1.dtype}, arr2.dtype={arr2.dtype}")
print(f"arr1 == arr2: {np.all(arr1 == arr2)}")
print(f"arr1 - arr2: {np.abs(arr1 - arr2).sum()}")