#!/usr/bin/env python3
"""Train SpixRWKV-7 on HumorDB for funniness rating regression (1-10 scale).

Dataset: https://huggingface.co/datasets/kreimanlab/HumorDB
Paper: https://arxiv.org/pdf/2406.13564

Usage:
    uv run python tasks/classification/humordb/train.py
    uv run python tasks/classification/humordb/train.py --embed-dims 128 --depth 6 --epochs 40
"""

import argparse
from typing import Optional
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Add project root for direct execution
if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_ROOT))

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7 as _create_model
from spixrwkv7.data.transforms import prepare_balanced_superpixel_features

# ---------------------------------------------------------------------------
# Checkpoint directory
# ---------------------------------------------------------------------------
_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints" / "humordb"

# ---------------------------------------------------------------------------
# Known HumorDB dataset sizes (from dataset metadata)
# ---------------------------------------------------------------------------
_DATASET_SIZES = {
    "train": 2136,
    "validation": 703,
    "test": 706,
}


# =====================================================================
# Image preprocessing
# =====================================================================


def pil_to_balanced(
    pil_image: Image.Image, img_size: int
) -> torch.Tensor:
    """Convert PIL RGB to 6-channel balanced tensor for SpixRWKV-7 input.

    Pipeline: RGB -> OkLAB -> Fixed Balancing (2*L-1, chroma_scale*a/b)
              -> alpha -> xy coords.
    """
    pil_image = pil_image.convert("RGB").resize(
        (img_size, img_size), Image.Resampling.BILINEAR
    )
    # (1, 3, H, W) float32 [0, 1]
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    img_tensor = (
        torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    )  # (1, 3, H, W)
    balanced = prepare_balanced_superpixel_features(
        img_tensor, alpha=None, chroma_scale=2.5
    )
    return balanced.squeeze(0)  # (6, H, W)


# =====================================================================
# Cached dataset — preprocesses once, then loads from disk
# =====================================================================


class HumorDBCached(Dataset):
    """HumorDB dataset with on-disk caching of preprocessed tensors.

    First run: iterates through the split, converts each PIL image to
    6-channel balanced tensor, and saves .pt files to `cache_dir/split/`.
    Subsequent runs: loads .pt files directly — no image processing.

    This eliminates the ~10 min/epoch preprocessing overhead after epoch 1.
    Total cache size: ~340 MB for all 3 splits at 64x64.
    """

    def __init__(
        self,
        split: str,
        img_size: int,
        cache_dir: str,
        rebuild: bool = False,
    ):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.cache_path = Path(cache_dir) / split
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._files: list[Path] = []

        expected = _DATASET_SIZES[split]
        existing = sorted(self.cache_path.glob("*.pt"))

        if len(existing) == expected and not rebuild:
            self._files = existing
        else:
            self._build_cache(expected)

    def _build_cache(self, expected: int) -> None:
        """Iterate the streaming dataset, preprocess, and save .pt files."""
        print(f"  Preprocessing {self.split} set ({expected} images)...")
        ds = load_dataset(
            "kreimanlab/HumorDB", split=self.split, streaming=True
        )
        for i, sample in enumerate(ds):
            if i >= expected:
                break
            tensor = pil_to_balanced(sample["image"], self.img_size)
            target = torch.tensor(
                sample["range_ratings_mean"], dtype=torch.float32
            )
            torch.save(
                {"img": tensor, "target": target},
                self.cache_path / f"{i:04d}.pt",
            )
            if (i + 1) % 200 == 0 or i == 0:
                print(f"    {i + 1}/{expected}")
        self._files = sorted(self.cache_path.glob("*.pt"))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(
            self._files[idx], map_location="cpu", weights_only=True
        )
        return data["img"], data["target"]

    def __len__(self):
        return len(self._files)


# =====================================================================
# DataLoader factory
# =====================================================================


def make_loader(
    split: str,
    img_size: int,
    batch_size: int,
    cache_dir: str,
    shuffle: bool,
    num_workers: int,
    rebuild_cache: bool = False,
):
    """Build a DataLoader for a HumorDB split with on-disk caching."""
    ds = HumorDBCached(
        split, img_size, cache_dir=cache_dir, rebuild=rebuild_cache
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# =====================================================================
# Regression head
# =====================================================================


class RegressionHead(nn.Module):
    """Regression head: GAP -> LayerNorm -> Linear(1)."""

    def __init__(self, embed_dims: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.head = nn.Linear(embed_dims, 1)

        # Attention Residuals parameters for regression head
        self.out_res_proj = nn.Linear(embed_dims, 1, bias=False)
        self.out_res_norm = nn.LayerNorm(embed_dims)
        self.out_res_bias = nn.Parameter(torch.tensor(10.0))
        nn.init.zeros_(self.out_res_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        attnres_history: Optional[list[torch.Tensor]] = None,
        project_fn = None,
    ) -> torch.Tensor:
        # x: (B, embed_dims, H_spatial, W_spatial)
        if attnres_history is not None and len(attnres_history) > 0 and project_fn is not None:
            # V: (L, B, SeqLen, D)
            V = torch.stack(attnres_history, dim=0)
            K = self.out_res_norm(V)
            query = self.out_res_proj.weight.view(-1)
            logits = torch.einsum("d, l b s d -> l b s", query, K)
            logits[-1] = logits[-1] + self.out_res_bias
            weights = logits.softmax(dim=0)
            h = torch.einsum("l b s, l b s d -> b s d", weights, V)
            feat = project_fn(h)
            x = feat.mean(dim=[-2, -1])
        else:
            x = x.mean(dim=[-2, -1])  # global average pool
        x = self.norm(x)
        return self.head(x).squeeze(-1)  # (B,)


class HumorRegressor(nn.Module):
    """SpixRWKV-7 backbone + regression head for HumorDB funniness rating.

    Unpacks the backbone's tuple output before passing to the regression head.
    """

    def __init__(self, backbone: nn.Module, embed_dims: int):
        super().__init__()
        self.backbone = backbone
        self.head = RegressionHead(embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.backbone(x)          # tuple of tensors, one per out_indices
        feat = outs[0]                   # (B, embed_dims, H_spatial, W_spatial)
        attnres_history = getattr(self.backbone, "last_attnres_history_patches", None)
        project_fn = getattr(self.backbone, "last_project_fn", None)
        return self.head(feat, attnres_history=attnres_history, project_fn=project_fn)


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
    """Pearson correlation coefficient."""
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    num = (preds_centered * targets_centered).sum()
    den = torch.sqrt((preds_centered**2).sum() * (targets_centered**2).sum())
    return (num / (den + 1e-8)).item()


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return math.sqrt(total)


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
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path, model: nn.Module, device: torch.device
) -> dict:
    """Load checkpoint and return metadata dict."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return state


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpixRWKV-7 — HumorDB funniness rating regression"
    )
    # Model hyper-parameters
    parser.add_argument(
        "--embed-dims", type=int, default=128,
        help="Embedding dimension (must be multiple of HEAD_SIZE=64)",
    )
    parser.add_argument(
        "--num-heads", type=int, default=2,
        help="Number of attention heads (embed_dims // HEAD_SIZE)"
    )
    parser.add_argument(
        "--depth", type=int, default=4,
        help="Number of Vision_RWKV7_Block layers"
    )
    parser.add_argument(
        "--img-size", type=int, default=64,
        help="Input image size in pixels (square)"
    )
    parser.add_argument(
        "--num-superpixels", type=int, default=36,
        help="Number of superpixel tokens (~6x6 grid for speed)"
    )
    parser.add_argument(
        "--diff-slic-iters", type=int, default=1,
        help="diffSLIC optimization iterations (1=faster, 5=better)"
    )
    parser.add_argument(
        "--compactness", type=float, default=0.5,
        help="diffSLIC compactness parameter"
    )
    parser.add_argument(
        "--drop-path-rate", type=float, default=0.1,
        help="Stochastic depth drop rate (regularization)"
    )
    parser.add_argument(
        "--init-values", type=float, default=1e-5,
        help="Weight initialization scale for RWKV params"
    )
    # Training hyper-parameters
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (CPU-friendly default)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Peak learning rate (AdamW)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log-interval", type=int, default=5,
        help="Log every N batches within each epoch"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(_CHECKPOINT_DIR),
        help="Directory for checkpoints (best_val_loss.pt + latest.pt + history.json)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (0=main process, 2-4 can speed disk I/O)",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Force rebuild of preprocessed cache (ignore existing .pt files)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  SpixRWKV-7 — HumorDB Funniness Regression")
    print("=" * 72)

    backbone = _create_model(
        img_size=args.img_size,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        depth=args.depth,
        init_values=args.init_values,
        final_norm=True,
        out_indices=[args.depth - 1],
        with_cls_token=False,
        output_cls_token=False,
        scatter_output=False,
        num_superpixels=args.num_superpixels,
        diff_slic_iters=args.diff_slic_iters,
        compactness=args.compactness,
        drop_path_rate=args.drop_path_rate,
        norm_layer="rmsnorm",
        act_layer="swiglu",
    ).to(device)

    backbone._init_weights()

    model = HumorRegressor(backbone, embed_dims=args.embed_dims).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("  Device          CPU")
    print(f"  Image size      {args.img_size}x{args.img_size}")
    print(f"  Embed dims      {args.embed_dims}")
    print(f"  Heads           {args.num_heads}")
    print(f"  Depth           {args.depth}")
    print(f"  Superpixels     {args.num_superpixels}")
    print(f"  Total params    {total_params:,}")
    print(f"  Train params    {train_params:,}")
    print(f"  Batch size      {args.batch_size}")
    print(f"  LR              {args.lr}")
    print(f"  Weight decay    {args.weight_decay}")
    print(f"  Drop path       {args.drop_path_rate}")
    print(f"  Epochs          {args.epochs}")
    print(f"  Checkpoints     {ckpt_dir.resolve()}")
    print("-" * 72)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("  Counting dataset splits...")
    train_size = _DATASET_SIZES["train"]
    val_size = _DATASET_SIZES["validation"]
    test_size = _DATASET_SIZES["test"]

    print(f"  Train: {train_size}  Val: {val_size}  Test: {test_size}")
    cache_dir = str(ckpt_dir)
    train_loader = make_loader(
        "train", args.img_size, args.batch_size, cache_dir,
        shuffle=True, num_workers=args.num_workers,
        rebuild_cache=args.rebuild_cache,
    )
    val_loader = make_loader(
        "validation", args.img_size, args.batch_size, cache_dir,
        shuffle=False, num_workers=args.num_workers,
        rebuild_cache=args.rebuild_cache,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    history: list[dict] = []
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        # -- Train --
        model.train()
        train_preds: list[torch.Tensor] = []
        train_targets: list[torch.Tensor] = []
        train_losses: list[float] = []
        grad_norms: list[float] = []

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)  # (B, 6, H, W)
            targets = targets.to(device)  # (B,)

            optimizer.zero_grad(set_to_none=True)
            preds = model(images)  # (B,)

            loss = F.mse_loss(preds, targets)
            loss.backward()

            gn = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(preds.detach().cpu())
            train_targets.append(targets.detach().cpu())
            grad_norms.append(gn)

            if (
                batch_idx == 0
                or (batch_idx + 1) % args.log_interval == 0
            ):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Train | Epoch {epoch:>3}/{args.epochs}"
                    f" Batch {batch_idx + 1:>4}/{max(train_size // args.batch_size, 1)}"
                    f" Loss {loss.item():.4f}"
                    f" LR {current_lr:.2e}"
                    f" GradNorm {gn:.2f}"
                )

        # Aggregate train metrics
        all_preds = torch.cat(train_preds)
        all_targets = torch.cat(train_targets)
        train_metrics = {
            "loss": float(np.mean(train_losses)),
            "rmse": compute_rmse(all_preds, all_targets),
            "mae": compute_mae(all_preds, all_targets),
            "r2": compute_r2(all_preds, all_targets),
            "pearson_r": compute_pearson_r(all_preds, all_targets),
            "grad_norm": float(np.mean(grad_norms)),
            "lr": optimizer.param_groups[0]["lr"],
        }

        # -- Validation --
        model.eval()
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        val_losses: list[float] = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)

                loss = F.mse_loss(preds, targets)
                val_losses.append(loss.item())
                val_preds.append(preds.cpu())
                val_targets.append(targets.cpu())

        all_val_preds = torch.cat(val_preds)
        all_val_targets = torch.cat(val_targets)
        val_metrics = {
            "loss": float(np.mean(val_losses)),
            "rmse": compute_rmse(all_val_preds, all_val_targets),
            "mae": compute_mae(all_val_preds, all_val_targets),
            "r2": compute_r2(all_val_preds, all_val_targets),
            "pearson_r": compute_pearson_r(
                all_val_preds, all_val_targets
            ),
        }

        # -- LR schedule --
        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start

        # -- Log --
        print("-" * 72)
        print(
            f"  TRAIN | Epoch {epoch:>3} | Loss={train_metrics['loss']:.4f} RMSE={train_metrics['rmse']:.4f} MAE={train_metrics['mae']:.4f} R2={train_metrics['r2']:.4f} r={train_metrics['pearson_r']:.4f} GradNorm={train_metrics['grad_norm']:.2f} LR={train_metrics['lr']:.2e}"
        )
        print(
            f"  VAL   | Epoch {epoch:>3} | Loss={val_metrics['loss']:.4f} RMSE={val_metrics['rmse']:.4f} MAE={val_metrics['mae']:.4f} R2={val_metrics['r2']:.4f} r={val_metrics['pearson_r']:.4f}"
        )
        print(f"  {'':>5} | Time={epoch_time:.1f}s")
        print("-" * 72)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "time_s": epoch_time,
            }
        )

        # -- Checkpointing --
        save_checkpoint(
            ckpt_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            train_metrics,
            val_metrics,
            args,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                ckpt_dir / "best_val_loss.pt",
                model,
                optimizer,
                epoch,
                train_metrics,
                val_metrics,
                args,
            )
            print(
                f"  >>> New best val loss: {best_val_loss:.4f}"
                f" (epoch {epoch})"
            )

        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.perf_counter() - total_start
    print(
        f"\n  Training complete in {total_time:.1f}s"
        f" ({total_time / 60:.1f}m)"
    )

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  Final Test Evaluation")
    print("=" * 72)

    best_ckpt = load_checkpoint(
        ckpt_dir / "best_val_loss.pt", model, device
    )
    print(f"  Loaded best checkpoint (epoch {best_ckpt['epoch']})")

    test_loader = make_loader(
        "test", args.img_size, args.batch_size, cache_dir,
        shuffle=False, num_workers=args.num_workers,
    )

    model.eval()
    test_preds: list[torch.Tensor] = []
    test_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            test_preds.append(preds.cpu())
            test_targets.append(targets.cpu())

    all_test_preds = torch.cat(test_preds)
    all_test_targets = torch.cat(test_targets)

    test_metrics = {
        "rmse": compute_rmse(all_test_preds, all_test_targets),
        "mae": compute_mae(all_test_preds, all_test_targets),
        "r2": compute_r2(all_test_preds, all_test_targets),
        "pearson_r": compute_pearson_r(
            all_test_preds, all_test_targets
        ),
    }

    print(f"  Test samples    {all_test_targets.shape[0]}")
    print(f"  RMSE            {test_metrics['rmse']:.4f}")
    print(f"  MAE             {test_metrics['mae']:.4f}")
    print(f"  R2              {test_metrics['r2']:.4f}")
    print(f"  Pearson r       {test_metrics['pearson_r']:.4f}")
    print(f"  Mean target     {all_test_targets.mean():.2f}")
    print(f"  Std target      {all_test_targets.std():.2f}")
    print(f"  Mean pred       {all_test_preds.mean():.2f}")
    print(f"  Std pred        {all_test_preds.std():.2f}")

    with open(ckpt_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    predictions_df = torch.stack(
        [all_test_targets, all_test_preds], dim=1
    ).numpy()
    np.savetxt(
        ckpt_dir / "test_predictions.csv",
        predictions_df,
        delimiter=",",
        header="target,prediction",
        comments="",
        fmt="%.6f",
    )
    print(
        f"  Predictions saved to {ckpt_dir / 'test_predictions.csv'}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
