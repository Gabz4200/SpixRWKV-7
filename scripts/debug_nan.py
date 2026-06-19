#!/usr/bin/env python3
"""Debug NaN in ADE20K forward pass — trace through each stage."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset

from spixrwkv7 import create_vision_rwkv7
from spixrwkv7.data.transforms import prepare_balanced_superpixel_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = 255
NUM_CLASSES = 150


def build_label_map(sample, img_size):
    H = W = img_size
    label = torch.full((H, W), IGNORE_INDEX, dtype=torch.long)
    segs = sample["segmentations"]
    objs = sample["objects"]
    for seg, obj in zip(segs, objs):
        if seg is not None and len(seg["counts"]) > 0:
            rle = {"size": seg["size"], "counts": seg["counts"]}
            try:
                from pycocotools import mask as mask_utils
                mask_np = mask_utils.decode(rle)
            except ImportError:
                continue
            h_seg, w_seg = mask_np.shape[:2]
            h_use, w_use = min(h_seg, H), min(w_seg, W)
            mask_t = torch.from_numpy(mask_np[:h_use, :w_use]).bool()
            label[:h_use, :w_use][mask_t] = obj["name_ndx"] % NUM_CLASSES
    return label


# --- Config (tiny) ---
img_size = 64
cfg = dict(
    img_size=img_size,
    embed_dims=128,
    num_heads=2,
    depth=2,
    init_values=1e-5,
    final_norm=True,
    out_indices=[-1],
    num_superpixels=36,
    scatter_output=True,
    diff_slic_iters=5,
    compactness=0.5,
)

torch.manual_seed(42)
device = torch.device("cpu")

# --- Load model ---
model = create_vision_rwkv7(**cfg).to(device)
model.eval()
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# --- Load one ADE20K sample ---
ds = load_dataset("1aurent/ADE20K", split="train", streaming=True)
sample = next(iter(ds))
print(f"\nImage size: {sample['image'].size}")
print(f"Objects: {len(sample['objects'])}")
print(f"Segmentations: {len(sample['segmentations'])}")

# Collect name_ndx values
name_ndxs = [obj["name_ndx"] for obj in sample["objects"]]
print(f"name_ndx range: {min(name_ndxs)}-{max(name_ndxs)}")
print(f"name_ndx unique: {len(set(name_ndxs))}")

# --- Build balanced features ---
pil_img = sample["image"].convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
arr = np.array(pil_img, dtype=np.float32) / 255.0
img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

balanced = prepare_balanced_superpixel_features(img_tensor, alpha=None, chroma_scale=2.5)
print(f"\nBalanced shape: {balanced.shape}")
print(f"Balanced range: [{balanced.min():.4f}, {balanced.max():.4f}]")
print(f"Balanced has NaN: {torch.isnan(balanced).any().item()}")
print(f"Balanced has Inf: {torch.isinf(balanced).any().item()}")

# --- Trace NaN through forward pass ---
print("\n--- Forward pass ---")
with torch.inference_mode():
    try:
        outs = model(balanced.to(device))
        for i, o in enumerate(outs):
            print(f"  out[{i}]: shape={o.shape}, "
                  f"has_nan={torch.isnan(o).any().item()}, "
                  f"has_inf={torch.isinf(o).any().item()}, "
                  f"range=[{o.min():.4f}, {o.max():.4f}]")
        print("\nNo NaN detected in forward pass.")
    except Exception as e:
        import traceback
        traceback.print_exc()

# --- Build label map ---
print("\n--- Label map ---")
label = build_label_map(sample, img_size)
unique_labels = label.unique().tolist()
print(f"Label map shape: {label.shape}")
print(f"Unique labels: {sorted(unique_labels)}")
print(f"Num classes present: {len(unique_labels) - (1 if IGNORE_INDEX in unique_labels else 0)}")

print("\nDebug complete.")
