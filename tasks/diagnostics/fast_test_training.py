"""Fast training convergence test for SpixRWKV-7.

Protocol (two-step ladder per wiki.imindlabs.com debugging guidance):
  1. Overfit a single batch — model should reach near 100% accuracy.
  2. (Future) Full CIFAR-10 classification benchmark.

Fails fast: a model that cannot overfit one batch likely has a bug or
bad optimization setup, not a weak architecture.
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Add project root to path for direct script execution
if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_ROOT))

from spixrwkv7 import ClassificationHead
from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7 as _create_model
from spixrwkv7.models.conv_spixrwkv7 import create_conv_vision_rwkv7
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7
from spixrwkv7.models.gnn_spixrwkv7 import create_gnn_vision

# ---------------------------------------------------------------------------
# Default hyper-parameters (tuned for single-batch overfit)
# ---------------------------------------------------------------------------
_BATCH_SIZE = 8
_NUM_CLASSES = 10
_IMG_SIZE = 512
_NUM_SUPERPIXELS = 128  # ~11x11 grid for speed
_BATCH_SIZE = 4
_NUM_SUPERPIXELS = 36    # ~6x6 grid — fewest tokens for speed
_EMBED_DIMS = 128
_NUM_HEADS = 2
_DEPTH = 2
_DROP_PATH = 0.0          # no regularization for overfit test
_INIT_VALUES = 1e-5
_LR = 5e-4
_WEIGHT_DECAY = 0.0       # no regularization for overfit test
_MAX_STEPS = 300
_LOG_INTERVAL = 5
_TARGET_ACCURACY = 0.95   # stop early when this is reached


def synth_batch(
    batch_size: int,
    num_classes: int,
    img_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Load a batch of real images from caltech101_classification."""
    from spixrwkv7.data.image_utils import load_random_caltech101_batch
    return load_random_caltech101_batch(
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        device=device,
        seed=42,
    )


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item() * 100.0


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return math.sqrt(total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpixRWKV-7 fast convergence test"
    )
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--num-classes", type=int, default=_NUM_CLASSES)
    parser.add_argument("--img-size", type=int, default=_IMG_SIZE)
    parser.add_argument("--num-superpixels", type=int, default=_NUM_SUPERPIXELS)
    parser.add_argument("--embed-dims", type=int, default=_EMBED_DIMS)
    parser.add_argument("--num-heads", type=int, default=_NUM_HEADS)
    parser.add_argument("--depth", type=int, default=_DEPTH)
    parser.add_argument("--lr", type=float, default=_LR)
    parser.add_argument("--weight-decay", type=float, default=_WEIGHT_DECAY)
    parser.add_argument("--max-steps", type=int, default=_MAX_STEPS)
    parser.add_argument("--log-interval", type=int, default=_LOG_INTERVAL)
    parser.add_argument(
        "--target-accuracy", type=float, default=_TARGET_ACCURACY
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-attnres", action="store_true", help="Enable Attention Residuals")
    parser.add_argument("--model-type", choices=["spix", "vq", "conv", "gnn"], default="spix",
                        help="Backbone type (default: spix)")
    parser.add_argument("--codebook-size", type=int, default=1024,
                        help="VQ codebook size")
    parser.add_argument("--downsample-factor", type=int, default=16,
                        help="VQ downsample factor")
    parser.add_argument("--downsample-factors", type=float, nargs="+", default=[1.0],
                        help="Downsample factors for spix sweep")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="VQ latent dimension (default: embed_dims)")
    parser.add_argument("--num-res-blocks", type=int, default=2,
                        help="Number of VQ residual blocks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data — one batch (shared across sweep runs)
    # ------------------------------------------------------------------
    x, y = synth_batch(
        args.batch_size, args.num_classes, args.img_size, device
    )

    factors = args.downsample_factors if args.model_type in ("spix", "gnn") else [1.0]

    for df in factors:
        if args.model_type in ("spix", "gnn"):
            print(f"\n=== Running {args.model_type} model with downsample_factor = {df} ===")

        # ---- Seed everything ----
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        if args.model_type == "vq":
            backbone = create_vq_rwkv7(
                img_size=args.img_size,
                embed_dims=args.embed_dims,
                num_heads=args.num_heads,
                depth=args.depth,
                init_values=_INIT_VALUES,
                final_norm=True,
                out_indices=[args.depth - 1],
                with_cls_token=False,
                output_cls_token=False,
                register_tokens=0,
                scatter_output=False,
                drop_path_rate=_DROP_PATH,
                codebook_size=args.codebook_size,
                downsample_factor=args.downsample_factor,
                latent_dim=args.latent_dim,
                num_res_blocks=args.num_res_blocks,
                use_ema=False,
                beta=0.25,
                norm_layer="rmsnorm",
                act_layer="swiglu",
                use_attnres=args.use_attnres,
                attnres_mode="block",
                attnres_gate_type="bias",
                attnres_num_blocks=8,
                attnres_recency_bias_init=10.0,
            ).to(device)
            backbone._init_weights()
        elif args.model_type == "conv":
            backbone = create_conv_vision_rwkv7(
                img_size=args.img_size,
                embed_dims=args.embed_dims,
                num_heads=args.num_heads,
                depth=args.depth,
                init_values=_INIT_VALUES,
                final_norm=True,
                out_indices=[args.depth - 1],
                num_superpixels=args.num_superpixels,
                scatter_output=False,
                diff_slic_iters=1,
                compactness=0.5,
                norm_layer="rmsnorm",
                act_layer="swiglu",
                spixel_backend="diff_slic",
                use_attnres=args.use_attnres,
                use_jit=False,
                conv_stem_channels=(32, 64, 128),
                conv_stem_kernel_sizes=(3, 5, 5),
                conv_stem_strides=(1, 2, 2),
                conv_stem_norm="batchnorm2d",
                conv_post_norm="layernorm",
            ).to(device)
            backbone._init_weights()
        elif args.model_type == "gnn":
            backbone = create_gnn_vision(
                img_size=args.img_size,
                embed_dims=args.embed_dims,
                num_heads=args.num_heads,
                depth=args.depth,
                init_values=_INIT_VALUES,
                final_norm=True,
                out_indices=[args.depth - 1],
                num_superpixels=args.num_superpixels,
                scatter_output=False,
                diff_slic_iters=1,
                compactness=0.5,
                drop_path_rate=_DROP_PATH,
                norm_layer="rmsnorm",
                act_layer="swiglu",
                downsample_factor=df,
                gnn_conv="gatv2",
                gnn_heads=4,
                gnn_aggr="mean",
            ).to(device)
            backbone._init_weights()
        else:
            backbone = _create_model(
                img_size=args.img_size,
                embed_dims=args.embed_dims,
                num_heads=args.num_heads,
                depth=args.depth,
                init_values=_INIT_VALUES,
                final_norm=True,
                out_indices=[args.depth - 1],  # only last layer
                with_cls_token=False,
                output_cls_token=False,
                scatter_output=False,
                num_superpixels=args.num_superpixels,
                diff_slic_iters=1,  # single iter — fast enough to check convergence
                compactness=0.5,
                drop_path_rate=_DROP_PATH,
                norm_layer="rmsnorm",
                act_layer="swiglu",
                use_attnres=args.use_attnres,
                downsample_factor=df,
            ).to(device)

            # Re-init weights for reproducibility
            backbone._init_weights()

        head = ClassificationHead(
            embed_dims=args.embed_dims, num_classes=args.num_classes
        ).to(device)

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------
        params = (
            list(backbone.parameters()) + list(head.parameters())
        )
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        d = device.type.upper()
        if device.type == "cuda":
            d += f" ({torch.cuda.get_device_name(0)})"
        total_params = sum(p.numel() for p in params)
        train_params = sum(p.numel() for p in params if p.requires_grad)

        print("=" * 72)
        print("  SpixRWKV-7 — Fast Convergence Test (single-batch overfit)")
        print("=" * 72)
        print(f"  Device         {d}")
        print(f"  Batch size     {args.batch_size}")
        print(f"  Image size     {args.img_size}x{args.img_size}")
        print(f"  Classes        {args.num_classes}")
        print(f"  Embed dims     {args.embed_dims}")
        print(f"  Heads          {args.num_heads}")
        print(f"  Depth          {args.depth}")
        print(f"  Superpixels    {args.num_superpixels}")
        print(f"  Downsample F   {df if args.model_type == 'spix' else args.downsample_factor}")
        print(f"  Total params   {total_params:,}")
        print(f"  Train params   {train_params:,}")
        print(f"  Optimizer      AdamW (lr={args.lr}, wd={args.weight_decay})")
        print(f"  Max steps      {args.max_steps}")
        print(f"  Target acc     {args.target_accuracy:.0%}")
        print("-" * 72)
        print(f"  {'Step':>5}  {'Loss':>8}  {'Acc':>6}  {'GradNorm':>8}"
              f"  {'Time/step':>9}")
        print("-" * 72)

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        best_acc = 0.0
        losses: list[float] = []
        accs: list[float] = []
        t0 = time.perf_counter()

        for step in range(1, args.max_steps + 1):
            step_t0 = time.perf_counter()

            backbone.train()
            head.train()
            optimizer.zero_grad(set_to_none=True)

            # Forward
            outs = backbone(x)          # tuple of tensors, one per out_indices
            feat = outs[0]              # (B, embed_dims, h_s, w_s)
            if getattr(backbone, "use_attnres", False):
                logits = head(
                    feat,
                    attnres_history=getattr(backbone, "last_attnres_history_patches", None),
                    project_fn=getattr(backbone, "last_project_fn", None)
                )
            else:
                logits = head(feat)         # (B, num_classes)
            loss = F.cross_entropy(logits, y)
            if args.model_type == "vq":
                q_loss = getattr(backbone, "_last_q_loss", None)
                if q_loss is not None:
                    loss = loss + q_loss

            # Backward
            loss.backward()
            grad_norm = compute_grad_norm(backbone) + compute_grad_norm(head)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            acc = accuracy(logits, y)
            losses.append(loss.item())
            accs.append(acc)
            best_acc = max(best_acc, acc)

            step_time = time.perf_counter() - step_t0

            if step % args.log_interval == 0 or step == 1:
                print(f"  {step:>5}  {loss.item():>8.4f}  {acc:>5.1f}%"
                      f"  {grad_norm:>8.2f}  {step_time*1e3:>8.1f}ms")

            if acc >= args.target_accuracy * 100.0:
                print(f"\n  >>> Target accuracy {args.target_accuracy:.0%} reached"
                      f" at step {step}.")
                break

        elapsed = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("=" * 72)
        print("  Summary")
        print("=" * 72)
        total_time_str = f"  Total time        {elapsed:.1f}s"
        print(total_time_str)
        print(f"  Steps run         {len(losses)}")
        if losses:
            print(f"  Final loss        {losses[-1]:.6f}")
        if accs:
            print(f"  Final accuracy    {accs[-1]:.1f}%")
        print(f"  Best accuracy     {best_acc:.1f}%")
        print("  Steps to 90%      ", end="")
        try:
            print(next(i + 1 for i, a in enumerate(accs) if a >= 90.0))
        except StopIteration:
            print("not reached")
        print("  Steps to 95%      ", end="")
        try:
            print(next(i + 1 for i, a in enumerate(accs) if a >= 95.0))
        except StopIteration:
            print("not reached")

        if best_acc >= args.target_accuracy * 100.0:
            print("\n  RESULT: PASS — model overfits the batch (convergence OK)")
        else:
            print(
                f"\n  RESULT: FAIL — best accuracy {best_acc:.1f}% < "
                f"{args.target_accuracy:.0%} target."
            )
            print("  Possible issues: bug in architecture, bad LR,"
                  " optimization setup.")

        print("=" * 72)


if __name__ == "__main__":
    from spixrwkv7.utils import redirect_stdout_tee
    os.makedirs('results', exist_ok=True)
    with redirect_stdout_tee('results/fast_test_training_downsample.txt'):
        main()
    print('Results saved to results/fast_test_training_downsample.txt')