#!/usr/bin/env python3
"""Run SpixRWKV-7 HumorDB checkpoint on test set and report metrics.

Usage:
    uv run python tasks/classification/humordb/infer.py
    uv run python tasks/classification/humordb/infer.py --checkpoint checkpoints/humordb/best_val_loss.pt
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_ROOT))

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from spixrwkv7 import create_vision_rwkv7
from spixrwkv7.data.transforms import prepare_balanced_superpixel_features


# ---------------------------------------------------------------------------
# Default checkpoint path
# ---------------------------------------------------------------------------
_DEFAULT_CKPT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "checkpoints" / "humordb" / "best_val_loss.pt"
)


# =====================================================================
# Image preprocessing (same as training)
# =====================================================================


def pil_to_balanced(pil_image: Image.Image, img_size: int) -> torch.Tensor:
    """Convert PIL RGB to 6-channel balanced tensor for SpixRWKV-7 input."""
    pil_image = pil_image.convert("RGB").resize(
        (img_size, img_size), Image.Resampling.BILINEAR
    )
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    img_tensor = (
        torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    )
    balanced = prepare_balanced_superpixel_features(
        img_tensor, alpha=None, chroma_scale=2.5
    )
    return balanced.squeeze(0)


# =====================================================================
# Streaming dataset (no shuffle for inference)
# =====================================================================


class HumorDBTestSet(IterableDataset):
    """Streaming test set for HumorDB inference."""

    def __init__(self, split: str, img_size: int):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self._dataset = load_dataset(
            "kreimanlab/HumorDB", split=split, streaming=True
        )

    def __iter__(self):
        for sample in self._dataset:
            img_tensor = pil_to_balanced(sample["image"], self.img_size)
            target = torch.tensor(
                sample["range_ratings_mean"], dtype=torch.float32
            )
            yield img_tensor, target

    def __len__(self):
        ds = load_dataset(
            "kreimanlab/HumorDB", split=self.split, streaming=True
        )
        return sum(1 for _ in ds)


# =====================================================================
# Regression head (same architecture as training)
# =====================================================================


class RegressionHead(nn.Module):
    """Regression head: GAP -> LayerNorm -> Linear(1)."""

    def __init__(self, embed_dims: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.head = nn.Linear(embed_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[-2, -1])
        x = self.norm(x)
        return self.head(x).squeeze(-1)


class HumorRegressor(nn.Module):
    """SpixRWKV-7 backbone + regression head for inference."""

    def __init__(self, backbone: nn.Module, embed_dims: int):
        super().__init__()
        self.backbone = backbone
        self.head = RegressionHead(embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.backbone(x)
        feat = outs[0]
        return self.head(feat)


# =====================================================================
# Metrics
# =====================================================================


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return math.sqrt(F.mse_loss(preds, targets).item())


def compute_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return F.l1_loss(preds, targets).item()


def compute_r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    mse = F.mse_loss(preds, targets)
    var = targets.var(unbiased=False)
    return (1.0 - mse / (var + 1e-8)).item()


def compute_pearson_r(
    preds: torch.Tensor, targets: torch.Tensor
) -> float:
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    num = (preds_centered * targets_centered).sum()
    den = torch.sqrt(
        (preds_centered**2).sum() * (targets_centered**2).sum()
    )
    return (num / (den + 1e-8)).item()


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpixRWKV-7 — HumorDB inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_DEFAULT_CKPT),
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Inference batch size",
    )
    parser.add_argument(
        "--img-size", type=int, default=64,
        help="Input image size (overrides checkpoint value if different)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load checkpoint metadata first to know embed_dims
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, dict):
        embed_dims = ckpt_args.get("embed_dims", 128)
        depth = ckpt_args.get("depth", 4)
        num_superpixels = ckpt_args.get("num_superpixels", 36)
        img_size = ckpt_args.get("img_size", args.img_size)
    else:
        embed_dims = 96
        depth = 4
        num_superpixels = 36
        img_size = args.img_size

    if args.img_size != 64:
        img_size = args.img_size

    print(f"  Using img_size={img_size}, embed_dims={embed_dims},"
          f" depth={depth}, num_superpixels={num_superpixels}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    backbone = create_vision_rwkv7(
        img_size=img_size,
        embed_dims=embed_dims,
        num_heads=max(2, embed_dims // 64),
        depth=depth,
        init_values=1e-5,
        final_norm=True,
        out_indices=[depth - 1],
        with_cls_token=False,
        output_cls_token=False,
        scatter_output=False,
        num_superpixels=num_superpixels,
        diff_slic_iters=1,
        compactness=0.5,
        drop_path_rate=0.0,
    ).to(device)

    model = HumorRegressor(backbone, embed_dims=embed_dims).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_metrics", {}).get("loss", "?")
    print(f"  Loaded checkpoint (epoch {epoch}, val_loss {val_loss})")
    print("-" * 72)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"  Loading {args.split} split...")
    ds = HumorDBTestSet(args.split, img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    preds_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    times: list[float] = []

    print(f"  Running inference on {args.split} set...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            t0 = time.perf_counter()
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            t1 = time.perf_counter()
            times.append(t1 - t0)

            preds_list.append(preds.cpu())
            targets_list.append(targets.cpu())

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"  Batch {batch_idx + 1:>4}/{len(ds) // args.batch_size}"
                    f" ({len(preds_list) * args.batch_size}/{len(ds)} samples)"
                )

    all_preds = torch.cat(preds_list)
    all_targets = torch.cat(targets_list)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"  Results — {args.split} set"
          f" ({all_targets.shape[0]} samples)")
    print("=" * 72)

    metrics = {
        "rmse": compute_rmse(all_preds, all_targets),
        "mae": compute_mae(all_preds, all_targets),
        "r2": compute_r2(all_preds, all_targets),
        "pearson_r": compute_pearson_r(all_preds, all_targets),
    }

    print(f"  RMSE            {metrics['rmse']:.4f}")
    print(f"  Results — {args.split} set ({all_targets.shape[0]} samples)")
    print(f"  R2              {metrics['r2']:.4f}")
    print(f"  Pearson r       {metrics['pearson_r']:.4f}")
    print()
    print("  Target stats:")
    print(f"    Mean          {all_targets.mean():.2f}")
    print(f"    Std           {all_targets.std():.2f}")
    print(f"    Min           {all_targets.min().item():.2f}")
    print(f"    Max           {all_targets.max().item():.2f}")
    print("  Prediction stats:")
    print(f"    Mean          {all_preds.mean():.2f}")
    print(f"    Std           {all_preds.std():.2f}")
    print(f"    Min           {all_preds.min().item():.2f}")
    print(f"    Max           {all_preds.max().item():.2f}")
    print()
    print("  Timing:")
    print(f"    Total samples {all_targets.shape[0]}")
    print(f"    Mean batch    {np.mean(times):.3f}s")
    print(f"    Per sample    {np.mean(times) / args.batch_size * 1000:.1f}ms")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    out_dir = ckpt_path.parent
    csv_path = out_dir / f"inference_{args.split}.csv"
    predictions_df = torch.stack([all_targets, all_preds], dim=1).numpy()
    np.savetxt(
        csv_path,
        predictions_df,
        delimiter=",",
        header="target,prediction",
        comments="",
        fmt="%.6f",
    )
    print(f"  Predictions saved to {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
