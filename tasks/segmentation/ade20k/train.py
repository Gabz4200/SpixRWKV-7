#!/usr/bin/env python3
"""Train SpixRWKV-7 on ADE20K semantic segmentation with streaming.

Dataset: https://huggingface.co/datasets/1aurent/ADE20K  (~5 GB, streaming)

Discovered classes: ADE20K raw name_ndx ranges 80-4000+; we scan the dataset
to discover unique classes and build a compressed mapping 0..C-1.

Usage:
    uv run python tasks/dense_prediction/ade20k/train.py --scale tiny --max-train-samples 128 --max-val-samples 32 --epochs 10
    uv run python tasks/dense_prediction/ade20k/train.py --scale medium --max-train-samples 256 --epochs 20
    uv run python tasks/dense_prediction/ade20k/train.py --scale 100m --epochs 50 --lr 3e-4
"""

import argparse
import json
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

from spixrwkv7.data.transforms import prepare_balanced_superpixel_features
from spixrwkv7.task_utils import (
    build_label_map,
    compute_grad_norm,
    discover_ade20k_classes,
    load_checkpoint as _load_checkpoint,
    mean_iou,
    pixel_accuracy,
    pil_to_balanced,
    save_checkpoint as _save_checkpoint,
)
from tasks.config_loader import load_model_config, build_backbone

# ---------------------------------------------------------------------------
# ADE20K constants
# ---------------------------------------------------------------------------
_IGNORE_INDEX = 255
_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints" / "ade20k"


# =====================================================================
# Discover class mapping
# =====================================================================


# =====================================================================
# Streaming dataset
# =====================================================================


class ADE20KStreaming(IterableDataset):
    """Streaming ADE20K dataset for segmentation training."""

    def __init__(
        self,
        split: str = "train",
        img_size: int = 64,
        max_samples: int | None = None,
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
        self._len: int | None = None

    def _get_len(self) -> int:
        if self._len is None:
            ds = load_dataset("1aurent/ADE20K", split=self.split, streaming=True)
            counted = 0
            for _ in ds:
                counted += 1
                if self.max_samples is not None and counted >= self.max_samples:
                    break
            self._len = counted
        return self._len

    def __len__(self) -> int:
        return self._get_len()

    def __iter__(self):
        ds = load_dataset("1aurent/ADE20K", split=self.split, streaming=True)
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        if self.max_samples is not None:
            ds = ds.take(self.max_samples)

        for sample in ds:
            img_tensor = pil_to_balanced(sample["image"], self.img_size)
            _, _, H, W = img_tensor.shape
            label = build_label_map(sample, H, W, self.class_map)
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

    def __init__(self, config: dict, num_classes: int, model_type: str = "spix"):
        super().__init__()
        self.model_type = model_type
        # Force scatter_output to True for dense prediction/segmentation task
        config = config.copy()
        config["scatter_output"] = True
        self.backbone = build_backbone(model_type, config)
        self.seg_head = SegHead(config["embed_dims"], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        return self.seg_head(features)


# =====================================================================
# Metrics
# =====================================================================


# =====================================================================
# Checkpointing
# =====================================================================


_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, metrics: dict,
) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "num_classes": model.seg_head.head.out_channels,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return state


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SpixRWKV-7 on ADE20K semantic segmentation"
    )
    parser.add_argument(
        "--scale", type=str, default="tiny", choices=["tiny", "small", "medium", "large"],
    )
    parser.add_argument(
        "--model-type", type=str, default="spix", choices=["spix", "conv", "vq", "gnn"],
        help="Model variant type"
    )
    parser.add_argument("--embed-dims", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-superpixels", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--diff-slic-iters", type=int, default=1)

    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training samples (None = all 25574)")
    parser.add_argument("--max-val-samples", type=int, default=200)
    parser.add_argument("--shuffle-buffer", type=int, default=100)
    parser.add_argument("--discovery-samples", type=int, default=500)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default=str(_CHECKPOINT_DIR))
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[CPU Tuning] For faster training, try:")
    print("  export OMP_NUM_THREADS=$(nproc)")
    print("  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so:$LD_PRELOAD")
    print()

    device = torch.device("cpu")

    cfg = load_model_config(args.model_type, args.scale)
    if args.embed_dims is not None:
        cfg["embed_dims"] = args.embed_dims
    if args.num_heads is not None:
        cfg["num_heads"] = args.num_heads
    if args.depth is not None:
        cfg["depth"] = args.depth
    if args.num_superpixels is not None:
        cfg["num_superpixels"] = args.num_superpixels
    if args.img_size is not None:
        cfg["img_size"] = args.img_size
    if args.drop_path_rate is not None:
        cfg["drop_path_rate"] = args.drop_path_rate
    if args.diff_slic_iters is not None:
        cfg["diff_slic_iters"] = args.diff_slic_iters

    print("=" * 72)
    print("ADE20K Segmentation Training")
    print("=" * 72)
    print(f"  Scale:            {args.scale}")
    print(f"  Model variant:    {args.model_type}")
    print(f"  Model:            embed_dims={cfg['embed_dims']}  "
          f"num_heads={cfg['num_heads']}  depth={cfg['depth']}")
    print(f"  Superpixels:      {cfg.get('num_superpixels', 'N/A')}  img={cfg['img_size']}")
    print(f"  Batch:            {args.batch_size}   LR: {args.lr}")
    print(f"  Max train:        {args.max_train_samples or 'all (25574)'}")
    print(f"  Max val:          {args.max_val_samples}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Discovery samples: {args.discovery_samples}")
    print(f"  Device:           {device}")
    print("=" * 72)

    # --- Discover ADE20K label classes ---
    print("Discovering ADE20K classes from train split...")
    discover_n = min(args.discovery_samples, args.max_train_samples or 25574)
    class_map = discover_ade20k_classes(
        split="train", max_samples=discover_n,
        shuffle_buffer=args.shuffle_buffer, seed=args.seed,
    )
    NUM_CLASSES = len(class_map)
    print(f"  Found {NUM_CLASSES} unique classes in {discover_n} train samples")

    val_check = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
    val_check = val_check.take(args.max_val_samples or 200)
    unknown_val = 0
    for sample in val_check:
        for obj in sample["objects"]:
            if obj["name_ndx"] not in class_map:
                unknown_val += 1
    print(f"  Unknown class instances in val set: {unknown_val}")
    print()

    # --- Datasets & DataLoaders ---
    print("Building datasets (streaming)...")
    train_ds = ADE20KStreaming(
        split="train", img_size=cfg["img_size"],
        max_samples=args.max_train_samples,
        shuffle_buffer=args.shuffle_buffer, seed=args.seed,
        class_map=class_map,
    )
    val_ds = ADE20KStreaming(
        split="validation", img_size=cfg["img_size"],
        max_samples=args.max_val_samples,
        shuffle_buffer=0, class_map=class_map,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True,
    )

    train_len = args.max_train_samples if args.max_train_samples is not None else len(train_ds)
    steps_per_epoch = math.ceil(train_len / args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    print(f"  Train samples:    {train_len}")
    print(f"  Steps per epoch:  {steps_per_epoch}")
    print(f"  Total steps:      {total_steps}")
    print("-" * 72)

    # --- Model ---
    model = ADE20KSegModel(cfg, NUM_CLASSES, model_type=args.model_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.seg_head.parameters())
    print(f"  Model params:     {total_params:,} total  ({trainable:,} trainable)")
    print(f"  Seg head:         {head_params:,}  ({NUM_CLASSES} classes x {model.backbone.embed_dims})")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(
            max(1, total_steps - args.warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=_IGNORE_INDEX)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    if args.resume:
        state = load_checkpoint(Path(args.resume), model, device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        print(f"  Resumed from epoch {state['epoch']}")
        print("-" * 72)

    history: dict = {"train_loss": [], "val_loss": [], "val_acc": [], "val_miou": []}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)

            if torch.isnan(loss).item():
                print(f"  E{epoch:02d} B{n_batches+1:04d} loss=NaN -- skip batch")
                continue

            loss.backward()
            grad_norm = compute_grad_norm(model)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            lr_now = scheduler.get_last_lr()[0]
            if n_batches % 20 == 0:
                print(
                    f"  E{epoch:02d} B{n_batches:04d}/{steps_per_epoch} | "
                    f"loss={loss.item():.4f} | grad_norm={grad_norm:.2f} | "
                    f"lr={lr_now:.2e}"
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"  -- E{epoch:02d} train_loss={avg_train_loss:.4f}  ({elapsed:.0f}s)")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_miou = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                vloss = criterion(logits, targets)
                if torch.isnan(vloss).item():
                    continue
                val_loss += vloss.item()
                val_acc += pixel_accuracy(logits, targets)
                val_miou += mean_iou(logits, targets, NUM_CLASSES)
                val_batches += 1

        val_loss /= max(val_batches, 1)
        val_acc /= max(val_batches, 1)
        val_miou /= max(val_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_miou"].append(val_miou)

        print(f"  -- E{epoch:02d} val_loss={val_loss:.4f}  "
              f"val_acc={val_acc*100:.2f}%  val_mIoU={val_miou*100:.2f}%")

        # --- Checkpoint ---
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_path = Path(args.checkpoint_dir) / "best_val_loss.pt"
            save_checkpoint(best_path, model, optimizer, epoch, {
                "train_loss": avg_train_loss, "val_loss": val_loss,
                "val_acc": val_acc, "val_miou": val_miou,
            })
            print(f"  >> Saved best ({best_path})")

        latest_path = Path(args.checkpoint_dir) / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch, {
            "train_loss": avg_train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "val_miou": val_miou,
        })

        history_path = Path(args.checkpoint_dir) / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print()

    print("=" * 72)
    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {args.checkpoint_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
