#!/usr/bin/env python3
"""Sanity overfit test: SpixRWKV-7 on ADE20K semantic segmentation (subset).

Tests whether the backbone can overfit a small subset (128-512 images).
Uses streaming to avoid loading 5 GB into RAM.

Key details:
  - ADE20K raw class indices range 80-4000+; we discover the actual classes
    from the first N samples and build a compressed mapping.
  - num_classes is set dynamically = number of unique name_ndx found.

Dataset: https://huggingface.co/datasets/1aurent/ADE20K

Usage:
    uv run python tasks/dense_prediction/ade20k/sanity.py --num-train-images 128 --epochs 20
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_ROOT))

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from spixrwkv7 import create_vision_rwkv7
from spixrwkv7.data.transforms import prepare_balanced_superpixel_features

# ---------------------------------------------------------------------------
# ADE20K constants
# ---------------------------------------------------------------------------
_IGNORE_INDEX = 255  # void/unlabeled — used for unknown classes too


# ---------------------------------------------------------------------------
# Scale presets — embed_dims must be multiple of HEAD_SIZE=64
# ---------------------------------------------------------------------------
_SCALES = {
    "tiny": {
        "embed_dims": 128, "num_heads": 2, "depth": 4,
        "num_superpixels": 36, "img_size": 64,
    },
    "small": {
        "embed_dims": 384, "num_heads": 6, "depth": 8,
        "num_superpixels": 64, "img_size": 112,
    },
    "medium": {
        "embed_dims": 576, "num_heads": 9, "depth": 12,
        "num_superpixels": 128, "img_size": 112,
    },
    "100m": {
        "embed_dims": 768, "num_heads": 12, "depth": 12,
        "num_superpixels": 196, "img_size": 224,
    },
}


def resolve_scale(cfg: dict) -> dict:
    """Fill defaults for CLI-overridden scale config."""
    base = _SCALES[cfg.get("scale", "tiny")].copy()
    for k in ("embed_dims", "num_heads", "depth", "num_superpixels", "img_size"):
        if k in cfg and cfg[k] is not None:
            base[k] = cfg[k]
    return base


# =====================================================================
# Discover class mapping from ADE20K raw indices -> compressed 0..C-1
# =====================================================================


def discover_ade20k_classes(
    split: str = "train", max_samples: int = 128, shuffle_buffer: int = 100, seed: int = 42
) -> dict[int, int]:
    """Scan the first max_samples of a split and build raw_ndx -> compressed index."""
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


# =====================================================================
# Build semantic label map using compressed class indices
# =====================================================================


def build_label_map(sample: dict, img_size: int, class_map: dict) -> torch.Tensor:
    """Build label map: (H, W) long tensor with compressed indices or _IGNORE_INDEX."""
    H, W = img_size, img_size
    label = torch.full((H, W), _IGNORE_INDEX, dtype=torch.long)
    for seg_pil, obj in zip(sample["segmentations"], sample["objects"]):
        raw_ndx = obj["name_ndx"]
        compressed = class_map.get(raw_ndx)
        if compressed is None:
            continue
        seg_resized = seg_pil.resize((W, H), Image.Resampling.NEAREST)
        mask_arr = np.array(seg_resized, dtype=np.int64)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask = torch.from_numpy(mask_arr > 0)
        if mask.any():
            label = label.clone()
            label[mask] = compressed
    return label


# =====================================================================
# Image preprocessing
# =====================================================================


def pil_to_balanced(pil_image: Image.Image, img_size: int) -> torch.Tensor:
    """PIL RGB -> 6-channel balanced tensor for SpixRWKV-7 input."""
    pil_image = pil_image.convert("RGB").resize(
        (img_size, img_size), Image.Resampling.BILINEAR
    )
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    balanced = prepare_balanced_superpixel_features(img_tensor, alpha=None, chroma_scale=2.5)
    return balanced.squeeze(0)


# =====================================================================
# Streaming dataset
# =====================================================================


class ADE20KStreaming(IterableDataset):
    """Streaming ADE20K subset for sanity testing."""

    def __init__(
        self,
        split: str = "train",
        img_size: int = 64,
        max_samples: int = 128,
        shuffle_buffer: int = 100,
        seed: int = 42,
        class_map: dict | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.max_samples = max_samples
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.split = split
        self.class_map = class_map or {}

    def __iter__(self):
        ds = load_dataset("1aurent/ADE20K", split=self.split, streaming=True)
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        ds = ds.take(self.max_samples)

        for sample in ds:
            img_tensor = pil_to_balanced(sample["image"], self.img_size)
            label = build_label_map(sample, self.img_size, self.class_map)
            yield img_tensor, label


# =====================================================================
# Segmentation Head
# =====================================================================


class SegHead(nn.Module):
    """Norm + 1x1 conv segmentation head: (B, D, H, W) -> (B, C, H, W)."""

    def __init__(self, embed_dims: int, num_classes: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.head = nn.Conv2d(embed_dims, num_classes, kernel_size=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[-1]
        return self.head(self.norm(x))


# =====================================================================
# Full model: backbone + seg head
# =====================================================================


class ADE20KSegModel(nn.Module):
    """SpixRWKV-7 backbone + 1x1 conv segmentation head."""

    def __init__(self, config: dict, num_classes: int):
        super().__init__()
        self.backbone = create_vision_rwkv7(
            img_size=config["img_size"],
            embed_dims=config["embed_dims"],
            num_heads=config["num_heads"],
            depth=config["depth"],
            num_superpixels=config["num_superpixels"],
            drop_path_rate=config.get("drop_path_rate", 0.0),
            scatter_output=True,
            diff_slic_iters=config.get("diff_slic_iters", 1),
            compactness=config.get("compactness", 0.5),
            init_values=1e-5,
        )
        self.seg_head = SegHead(config["embed_dims"], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.seg_head(features)


# =====================================================================
# Metrics
# =====================================================================


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return math.sqrt(total)


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Percent of non-ignore pixels correctly classified."""
    preds = logits.argmax(dim=1)
    mask = targets != _IGNORE_INDEX
    if not mask.any():
        return 0.0
    correct = (preds[mask] == targets[mask]).float().sum()
    total = mask.float().sum()
    return (correct / total).item()


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADE20K sanity overfit test for SpixRWKV-7 segmentation"
    )
    parser.add_argument(
        "--scale", type=str, default="tiny", choices=list(_SCALES.keys()),
    )
    parser.add_argument("--embed-dims", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-superpixels", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--diff-slic-iters", type=int, default=1)
    parser.add_argument("--num-train-images", type=int, default=128)
    parser.add_argument("--num-val-images", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=100)
    parser.add_argument("--target-accuracy", type=float, default=0.0,
                        help="Stop early when pixel accuracy >= this")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = resolve_scale({
        "scale": args.scale, "embed_dims": args.embed_dims,
        "num_heads": args.num_heads, "depth": args.depth,
        "num_superpixels": args.num_superpixels, "img_size": args.img_size,
        "drop_path_rate": args.drop_path_rate, "diff_slic_iters": args.diff_slic_iters,
    })

    device = torch.device("cpu")
    print("=" * 72)
    print("ADE20K Sanity Overfit Test")
    print("=" * 72)
    print(f"  Model scale:     {args.scale}")
    print(f"  embed_dims:      {cfg['embed_dims']}  (num_heads={cfg['num_heads']})")
    print(f"  depth:           {cfg['depth']}")
    print(f"  num_superpixels: {cfg['num_superpixels']}")
    print(f"  img_size:        {cfg['img_size']}")
    print(f"  num_train:       {args.num_train_images}")
    print(f"  num_val:         {args.num_val_images}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  lr:              {args.lr}")
    print(f"  epochs:          {args.epochs}")
    print(f"  device:          {device}")
    print("=" * 72)

    # --- Discover ADE20K label classes ---
    print("Discovering ADE20K classes from train split...")
    class_map = discover_ade20k_classes(
        split="train",
        max_samples=args.num_train_images,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )
    NUM_CLASSES = len(class_map)
    unknown_count = 0
    print(f"  Found {NUM_CLASSES} unique classes in {args.num_train_images} train samples")
    val_ds_check = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
    val_ds_check = val_ds_check.take(args.num_val_images)
    for sample in val_ds_check:
        for obj in sample["objects"]:
            if obj["name_ndx"] not in class_map:
                unknown_count += 1
    print(f"  Unknown classes in val set: {unknown_count} ")
    print()

    # --- Datasets & DataLoaders ---
    train_ds = ADE20KStreaming(
        split="train", img_size=cfg["img_size"],
        max_samples=args.num_train_images,
        shuffle_buffer=args.shuffle_buffer, seed=args.seed,
        class_map=class_map,
    )
    val_ds = ADE20KStreaming(
        split="validation", img_size=cfg["img_size"],
        max_samples=args.num_val_images,
        shuffle_buffer=0, class_map=class_map,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False,
    )

    # --- Model ---
    model = ADE20KSegModel(cfg, NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} (trainable: {trainable:,})")
    head_params = sum(p.numel() for p in model.seg_head.parameters())
    print(f"  Seg head:      {head_params:,}  ({NUM_CLASSES} classes x {cfg['embed_dims']})")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=_IGNORE_INDEX)

    # --- Training ---
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)

            if torch.isnan(loss).item():
                print(f"  E{epoch:02d} B{batch_idx+1:03d} loss=NaN -- skipping batch")
                continue

            loss.backward()
            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            acc = pixel_accuracy(logits.detach(), targets)
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"  E{epoch:02d} B{batch_idx+1:03d} | "
                      f"loss={loss.item():.4f} | acc={acc*100:.1f}% | "
                      f"grad_norm={grad_norm:.2f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                vloss = criterion(logits, targets)
                if not torch.isnan(vloss).item():
                    val_loss += vloss.item()
                    val_acc += pixel_accuracy(logits, targets)
                    val_batches += 1

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches
        else:
            avg_loss = float("nan")
            avg_acc = 0.0

        if val_batches > 0:
            val_loss /= val_batches
            val_acc /= val_batches

        print(
            f"  E{epoch:02d}  | train_loss={avg_loss:.4f} train_acc={avg_acc*100:.1f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}% | "
            f"{elapsed:.0f}s"
        )

        if avg_loss < 0.1:
            print("  >> Loss < 0.1 -- model successfully overfitting!")

        if val_acc >= args.target_accuracy > 0:
            print(f"  >> Target accuracy {args.target_accuracy*100:.0f}% reached!")
            break

    print("=" * 72)
    print("Done.")
    avg_epoch = sum(epoch_times) / len(epoch_times)
    print(f"  Avg epoch time: {avg_epoch:.0f}s")
    print(f"  Loss trend: train={epoch_times[0] if epoch_times else 0:.1f} -> ...")
    print("  - Decreasing to ~0 -> model CAN overfit (architecture passes)")
    print("  - Stagnant / NaN    -> architecture or training issue")
    print("=" * 72)


if __name__ == "__main__":
    main()
