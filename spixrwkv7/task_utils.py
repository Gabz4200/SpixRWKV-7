"""Shared task utilities for SpixRWKV-7 training, inference, and evaluation.

Eliminates copy-paste duplication across tasks/classification/humordb/,
tasks/segmentation/ade20k/, tasks/diagnostics/, and scripts/.
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# =====================================================================
# Image preprocessing
# =====================================================================


def pil_to_balanced(pil_image: Image.Image, img_size: int) -> torch.Tensor:
    """PIL RGB -> 6-channel balanced tensor for SpixRWKV-7 input.

    Resizes so that height matches ``img_size`` (proportional width).
    If ``img_size <= 0``, original resolution is preserved.
    """
    from spixrwkv7.data.image_utils import prepare_balanced_superpixel_features

    pil_image = pil_image.convert("RGB")
    if img_size > 0:
        orig_w, orig_h = pil_image.size
        aspect = orig_w / orig_h
        new_h = img_size
        new_w = int(round(new_h * aspect))
        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    balanced = prepare_balanced_superpixel_features(img_tensor, alpha=None, chroma_scale=2.5)
    return balanced.squeeze(0)


# =====================================================================
# Gradient norm
# =====================================================================


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return math.sqrt(total)


# =====================================================================
# Regression metrics
# =====================================================================


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Root mean squared error."""
    return math.sqrt(F.mse_loss(preds, targets).item())


def compute_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error."""
    return F.l1_loss(preds, targets).item()


def compute_r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """R-squared (coefficient of determination)."""
    mse = F.mse_loss(preds, targets)
    var = targets.var(unbiased=False)
    return (1.0 - mse / (var + 1e-8)).item()


def compute_pearson_r(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Pearson correlation coefficient."""
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    num = (preds_centered * targets_centered).sum()
    den = torch.sqrt(
        (preds_centered**2).sum() * (targets_centered**2).sum()
    )
    return (num / (den + 1e-8)).item()


# =====================================================================
# Segmentation metrics
# =====================================================================

_IGNORE_INDEX = -1


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Percent of non-ignore pixels correctly classified."""
    preds = logits.argmax(dim=1)
    mask = targets != _IGNORE_INDEX
    if not mask.any():
        return 0.0
    correct = (preds[mask] == targets[mask]).float().sum()
    total = mask.float().sum()
    return (correct / total).item()


def mean_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Mean intersection-over-union across classes (ignores _IGNORE_INDEX)."""
    preds = logits.argmax(dim=1)
    mask = targets != _IGNORE_INDEX
    if not mask.any():
        return 0.0
    ious = []
    for c in range(num_classes):
        pred_c = preds[mask] == c
        target_c = targets[mask] == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


# =====================================================================
# ADE20K dataset utilities
# =====================================================================


def discover_ade20k_classes(
    split: str = "train",
    max_samples: int = 128,
    shuffle_buffer: int = 100,
    seed: int = 42,
) -> dict[int, int]:
    """Scan the first max_samples of a split and build raw_ndx -> compressed index."""
    from datasets import load_dataset

    ds = load_dataset("1aurent/ADE20K", split=split, streaming=True)
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    class_set: set[int] = set()
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break
        for obj in sample["objects"]:
            class_set.add(obj["name_ndx"])
    sorted_classes = sorted(class_set)
    return {raw: comp for comp, raw in enumerate(sorted_classes)}


def build_label_map(
    sample: dict,
    height: int,
    width: int,
    class_map: dict,
) -> torch.Tensor:
    """Build label map: (H, W) long tensor with compressed indices or _IGNORE_INDEX."""
    from PIL import Image as _Image

    H, W = height, width
    label = torch.full((H, W), _IGNORE_INDEX, dtype=torch.long)
    for seg_pil, obj in zip(sample["segmentations"], sample["objects"]):
        raw_ndx = obj["name_ndx"]
        compressed = class_map.get(raw_ndx)
        if compressed is None:
            continue
        seg_resized = seg_pil.resize((W, H), _Image.Resampling.NEAREST)
        mask_arr = np.array(seg_resized, dtype=np.int64)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask = torch.from_numpy(mask_arr > 0)
        if mask.any():
            label = label.clone()
            label[mask] = compressed
    return label


# =====================================================================
# Checkpointing
# =====================================================================


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Save model checkpoint with metrics and config."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    if args is not None:
        state["args"] = vars(args)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict:
    """Load model checkpoint, returning the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
