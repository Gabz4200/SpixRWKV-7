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
from typing import Optional

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
from tasks.config_loader import load_model_config, build_backbone

# ---------------------------------------------------------------------------
# ADE20K constants
# ---------------------------------------------------------------------------
_IGNORE_INDEX = 255  # void/unlabeled — used for unknown classes too


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


def build_label_map(sample: dict, height: int, width: int, class_map: dict) -> torch.Tensor:
    """Build label map: (H, W) long tensor with compressed indices or _IGNORE_INDEX."""
    H, W = height, width
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
    """PIL RGB -> 6-channel balanced tensor for SpixRWKV-7 input.

    Resizes so that height matches ``img_size`` (proportional width).
    If ``img_size <= 0``, original resolution is preserved.
    """
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
# Segmentation Head
# =====================================================================


class SegHead(nn.Module):
    """Norm + 1x1 conv segmentation head: (B, D, H, W) -> (B, C, H, W)."""

    def __init__(self, embed_dims: int, num_classes: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.head = nn.Conv2d(embed_dims, num_classes, kernel_size=1, bias=False)
        self._init_weights()

        # Attention Residuals parameters for segmentation head
        self.out_res_proj = nn.Linear(embed_dims, 1, bias=False)
        self.out_res_norm = nn.BatchNorm2d(embed_dims)
        self.out_res_bias = nn.Parameter(torch.tensor(10.0))
        nn.init.zeros_(self.out_res_proj.weight)

    def _init_weights(self):
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        attnres_history: Optional[list[torch.Tensor]] = None,
        project_fn = None,
    ) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[-1]

        if attnres_history is not None and len(attnres_history) > 0 and project_fn is not None:
            # Project all 3D tensors in the history to 4D (B, D, H, W)
            projected = [project_fn(h) for h in attnres_history]
            V = torch.stack(projected, dim=0)  # (L, B, D, H, W)

            L, B, D, H, W = V.shape
            K = self.out_res_norm(V.view(L * B, D, H, W)).view(L, B, D, H, W)

            query = self.out_res_proj.weight.view(-1)
            logits = torch.einsum("d, l b d h w -> l b h w", query, K)  # (L, B, H, W)
            logits[-1] = logits[-1] + self.out_res_bias

            weights = logits.softmax(dim=0)  # (L, B, H, W)
            x = torch.einsum("l b h w, l b d h w -> b d h w", weights, V)

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
        attnres_history = getattr(self.backbone, "last_attnres_history_patches", None)
        project_fn = getattr(self.backbone, "last_project_fn", None)
        return self.seg_head(features, attnres_history=attnres_history, project_fn=project_fn)


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
# Streaming dataset
# =====================================================================


class ADE20KStreaming(IterableDataset):
    """Streaming ADE20K dataset for segmentation sanity checks."""

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
            _, H, W = img_tensor.shape
            label = build_label_map(sample, H, W, self.class_map)
            yield img_tensor, label


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
    parser.add_argument("--model-type", choices=["spix", "vq", "conv", "gnn"], default="spix",
                        help="Backbone type (default: spix)")
    parser.add_argument("--codebook-size", type=int, default=1024,
                        help="VQ codebook size")
    parser.add_argument("--downsample-factor", type=int, default=16,
                        help="VQ downsample factor")
    parser.add_argument(
        "--downsample-factors",
        type=float,
        nargs="+",
        default=None,
        help="Downsample factors to sweep for spix backbone",
    )
    parser.add_argument(
        "--compare-variants",
        nargs="+",
        default=None,
        help="Run sanity overfit for spix/conv/vq and print a comparison table",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    if args.downsample_factor is not None:
        cfg["downsample_factor"] = args.downsample_factor
    if args.codebook_size is not None:
        cfg["codebook_size"] = args.codebook_size

    device = torch.device("cpu")
    print("=" * 72)
    print("ADE20K Sanity Overfit Test")
    print("=" * 72)
    print(f"  Model scale:     {args.scale}")
    print(f"  embed_dims:      {cfg['embed_dims']}  (num_heads={cfg['num_heads']})")
    print(f"  depth:           {cfg['depth']}")
    print(f"  num_superpixels: {cfg.get('num_superpixels', 'N/A')}")
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
    print(f"  Unknown classes in val set: {unknown_count} ")
    print()

    def make_model(variant: str, df: Optional[float] = None) -> nn.Module:
        cfg_variant = load_model_config(variant, args.scale)
        if args.embed_dims is not None:
            cfg_variant["embed_dims"] = args.embed_dims
        if args.num_heads is not None:
            cfg_variant["num_heads"] = args.num_heads
        if args.depth is not None:
            cfg_variant["depth"] = args.depth
        if args.num_superpixels is not None:
            cfg_variant["num_superpixels"] = args.num_superpixels
        if args.img_size is not None:
            cfg_variant["img_size"] = args.img_size
        if args.drop_path_rate is not None:
            cfg_variant["drop_path_rate"] = args.drop_path_rate
        if args.diff_slic_iters is not None:
            cfg_variant["diff_slic_iters"] = args.diff_slic_iters
        if df is not None:
            cfg_variant["downsample_factor"] = df
        elif args.downsample_factor is not None:
            cfg_variant["downsample_factor"] = args.downsample_factor
        if args.codebook_size is not None:
            cfg_variant["codebook_size"] = args.codebook_size
        return ADE20KSegModel(cfg_variant, NUM_CLASSES, model_type=variant).to(device)

    def train_variant(variant: str, df: Optional[float] = None) -> dict:
        print("\n" + "-" * 72)
        if df is not None:
            print(f"Training variant: {variant} (downsample_factor={df})")
        else:
            print(f"Training variant: {variant}")
        print("-" * 72)
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
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True,
        )

        model = make_model(variant, df)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params: {total_params:,} (trainable: {trainable:,})")
        head_params = sum(p.numel() for p in model.seg_head.parameters())
        print(f"  Seg head:      {head_params:,}  ({NUM_CLASSES} classes x {model.backbone.embed_dims})")
        print()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=_IGNORE_INDEX)

        epoch_times = []
        epoch_losses = []
        best_val_loss = float("inf")
        best_epoch = None
        target_epoch = None

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
                if variant == "vq":
                    q_loss = getattr(model.backbone, "_last_q_loss", None)
                    if q_loss is not None:
                        loss = loss + q_loss
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
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs)
                    vloss = criterion(logits, targets)
                    if torch.isfinite(vloss).item():
                        val_loss += vloss.item()
                        val_acc += pixel_accuracy(logits, targets)
                        val_batches += 1

            elapsed = time.time() - t0
            epoch_times.append(elapsed)

            avg_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")
            avg_acc = epoch_acc / n_batches if n_batches > 0 else 0.0
            val_loss /= val_batches if val_batches > 0 else 1
            val_acc /= val_batches if val_batches > 0 else 1
            epoch_losses.append(avg_loss)

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_epoch = epoch
            if avg_loss < 0.1 and target_epoch is None:
                target_epoch = epoch

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

        avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        return {
            "model_type": f"{variant} (df={df})" if df is not None else variant,
            "total_params": total_params,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "target_epoch": target_epoch,
            "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
            "avg_epoch_s": avg_epoch,
            "epochs_surfaced": len(epoch_losses),
        }

    variants = [args.model_type]
    results = []
    if args.compare_variants:
        candidates = [v for v in args.compare_variants if v in {"spix", "conv", "vq", "gnn"}]
        if candidates:
            variants = candidates

    for variant in variants:
        dfactors = args.downsample_factors if (args.downsample_factors is not None and variant in ("spix", "gnn")) else [None]
        for df in dfactors:
            results.append(train_variant(variant, df))

    if len(results) > 1:
        print("\n" + "=" * 72)
        print("VARIANT COMPARISON")
        print("=" * 72)
        print(f"{'Variant':<10} {'Params':<12} {'BestEpoch':<10} {'BestValLoss':<12} {'AvgEpoch(s)':<12} {'FinalLoss':<12} {'TargetEpoch':<12}")
        print("-" * 82)
        for r in results:
            print(f"{r['model_type']:<10} {r['total_params']:<12,} {str(r['best_epoch']):<10} {r['best_val_loss']:<12.4f} {r['avg_epoch_s']:<12.1f} {r['final_loss']:<12.4f} {str(r['target_epoch']):<12}")
        best = min(results, key=lambda r: r["best_val_loss"])
        print(f"\nBest validation loss: {best['model_type']} at epoch {best['best_epoch']} with loss {best['best_val_loss']:.4f}")
        print("  - Lower best_val_loss -> this variant fit the subset faster")
        print("  - Earlier best_epoch  -> faster convergence on this task/size")
        print("  - Earlier target_epoch -> quicker escape from high-loss plateau")

    print("=" * 72)
    print("Done.")
    if results:
        print(f"  Variants run: {', '.join(r['model_type'] for r in results)}")
    print("  - Decreasing to ~0 -> model CAN overfit (architecture passes)")
    print("  - Stagnant / NaN    -> architecture or training issue")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    has_downsample = False
    for i, arg in enumerate(sys.argv):
        if arg == "--downsample-factors":
            has_downsample = True
            break
    if has_downsample:
        import os
        from spixrwkv7.utils import redirect_stdout_tee
        os.makedirs("results", exist_ok=True)
        with redirect_stdout_tee("results/ade20k_sanity_downsample.txt"):
            main()
    else:
        main()
