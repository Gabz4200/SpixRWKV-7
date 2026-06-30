"""Diagnostic experiments for SpixRWKV-7 convergence behavior.

Runs multiple experiments in sequence, logging detailed gradient and
weight statistics for each. All output goes to stdout as structured
sections prefixed with [EXPERIMENT] for easy parsing.
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

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from spixrwkv7 import ClassificationHead  # noqa: E402
from spixrwkv7.kernels.optimized_vision import (  # noqa: E402
    create_optimized_vision_rwkv7 as _create_model,
)
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7  # noqa: E402


def synth_batch(batch_size, num_classes, img_size, device):
    x = torch.randn(batch_size, 6, img_size, img_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def build_model(depth, embed_dims, num_heads, num_superpixels, device,
                model_type="spix", codebook_size=1024, downsample_factor=16,
                latent_dim=None, num_res_blocks=2):
    if model_type == "vq":
        backbone = create_vq_rwkv7(
            img_size=512,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            init_values=1e-5,
            final_norm=True,
            out_indices=[depth - 1],
            with_cls_token=False,
            output_cls_token=False,
            scatter_output=False,
            drop_path_rate=0.0,
            codebook_size=codebook_size,
            downsample_factor=downsample_factor,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            norm_layer="rmsnorm",
            act_layer="swiglu",
        ).to(device)
        backbone._init_weights()
    else:
        backbone = _create_model(
            img_size=512,
            embed_dims=embed_dims,
            num_heads=num_heads,
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
            norm_layer="rmsnorm",
            act_layer="swiglu",
        ).to(device)
        backbone._init_weights()
    head = ClassificationHead(embed_dims, 10).to(device)
    return backbone, head


def log(msg: str):
    print(f"[EXPERIMENT] {msg}")


def run(
    label: str,
    depth: int = 2,
    embed_dims: int = 128,
    num_heads: int = 2,
    num_superpixels: int = 36,
    lr: float = 5e-4,
    max_steps: int = 2,
    seed: int = 42,
    model_type: str = "spix",
    codebook_size: int = 1024,
    downsample_factor: int = 16,
    latent_dim: int | None = None,
    num_res_blocks: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log(f"START label={label} seed={seed} depth={depth} "
        f"embed_dims={embed_dims} lr={lr}")

    backbone, head = build_model(depth, embed_dims, num_heads,
                                  num_superpixels, device,
                                  model_type=model_type,
                                  codebook_size=codebook_size,
                                  downsample_factor=downsample_factor,
                                  latent_dim=latent_dim,
                                  num_res_blocks=num_res_blocks)
    total = sum(p.numel() for p in backbone.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    log(f"PARAMS backbone={total} head={head_params}")

    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=0.0,
    )
    x, y = synth_batch(4, 10, 512, device)

    # Track metrics
    losses, accs, grad_norms = [], [], []
    # Layer-wise gradient stats
    block_grad_norms = {i: [] for i in range(depth)}

    step_times = []
    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        backbone.train()
        head.train()
        opt.zero_grad(set_to_none=True)
        outs = backbone(x)
        logits = head(outs[0])
        loss = F.cross_entropy(logits, y)
        if getattr(backbone, "_last_q_loss", None) is not None:
            q_loss = getattr(backbone, "_last_q_loss")
            loss = loss + q_loss
        loss.backward()

        # --- Gradient diagnostics ---
        total_grad = 0.0
        for i, block in enumerate(backbone.blocks):
            gn = 0.0
            for p in block.parameters():
                if p.grad is not None:
                    gn += p.grad.norm().item() ** 2
            block_grad_norms[i].append(math.sqrt(gn) if gn > 0 else 0.0)
            total_grad += gn

        # Head gradient
        for p in head.parameters():
            if p.grad is not None:
                total_grad += p.grad.norm().item() ** 2
        total_grad = math.sqrt(total_grad)

        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(head.parameters()),
            max_norm=10.0,
        )
        opt.step()

        acc = (logits.argmax(1) == y).float().mean().item() * 100
        losses.append(loss.item())
        accs.append(acc)
        grad_norms.append(total_grad)
        step_times.append(time.perf_counter() - t0)

    # --- Summary ---
    log(f"RESULT label={label} steps={len(losses)} "
        f"final_loss={losses[-1]:.6f} "
        f"final_acc={accs[-1]:.1f} "
        f"best_acc={max(accs):.1f} "
        f"mean_step_time={sum(step_times)/len(step_times)*1e3:.1f}ms")

    # Plateau detection: first step where acc >= 50%
    plateau_len = 0
    for a in accs:
        if a < 50.0:
            plateau_len += 1
        else:
            break
    log(f"PLATEAU steps_below_50pct={plateau_len}")

    # Gradient distribution across blocks (averaged over last 10 steps)
    tail = slice(-10, None) if len(accs) >= 10 else slice(None)
    for i in range(depth):
        avg_gn = sum(block_grad_norms[i][tail]) / max(len(block_grad_norms[i][tail]), 1)
        log(f"GRAD_BLOCK_{i} avg_last_steps={avg_gn:.4f}")

    # Gradient health: fraction of steps where grad norm is in [0.01, 100]
    healthy = sum(1 for g in grad_norms if 0.01 <= g <= 100.0)
    log(f"GRAD_HEALTH fraction_healthy={healthy}/{len(grad_norms)}")

    # Logit statistics at final step
    with torch.no_grad():
        outs = backbone(x)
        logits = head(outs[0])
        probs = F.softmax(logits, dim=1)
        log(f"LOGITS mean={logits.mean().item():.4f} std={logits.std().item():.4f} "
            f"max={logits.max().item():.4f} min={logits.min().item():.4f}")
        log(f"PROBS max={probs.max().item():.4f} "
            f"mean_conf={probs.max(dim=1).values.mean().item():.4f}")

    # Feature statistics at final step
    with torch.no_grad():
        feat = outs[0]
        log(f"FEAT mean={feat.mean().item():.4f} std={feat.std().item():.4f} "
            f"range=[{feat.min().item():.4f}, {feat.max().item():.4f}]")

    log(f"END label={label}")
    return losses, accs, grad_norms


# =====================================================================
# Experiment runners
# =====================================================================

def exp_lr_sweep():
    log("=== LR SWEEP ===")
    for lr in [1e-3, 5e-4, 1e-4, 5e-5]:
        run(label=f"lr={lr}", lr=lr)
        print()  # blank line between runs

def exp_depth():
    log("=== DEPTH SCALING ===")
    for depth in [1, 2, 4]:
        num_sup = 36  # keep constant for comparability
        run(label=f"depth={depth}", depth=depth, num_superpixels=num_sup)
        print()

def exp_seeds():
    log("=== SEED REPRODUCIBILITY ===")
    for seed in [42, 123, 456, 789]:
        run(label=f"seed={seed}", seed=seed)
        print()

def exp_gradient_deep():
    """Run a deeper model (depth=4) and track per-layer gradients."""
    log("=== GRADIENT DIAGNOSTIC (depth=4) ===")
    run(label="grad-diag-depth4", depth=4, max_steps=2)
    print()

def exp_no_head(model_type="spix"):
    """Verify backbone features alone are finite and have nonzero variance."""
    log("=== NO-HEAD FEATURE SANITY ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    x, _ = synth_batch(4, 10, 512, device)
    backbone, _ = build_model(2, 128, 2, 36, device, model_type=model_type)
    with torch.no_grad():
        outs = backbone(x)
        feat = outs[0]
    log(f"NOHEAD shape={tuple(feat.shape)}")
    log(f"NOHEAD mean={feat.mean().item():.6f} std={feat.std().item():.6f}")
    log(f"NOHEAD range=[{feat.min().item():.6f}, {feat.max().item():.6f}]")
    log(f"NOHEAD finite={feat.isfinite().all().item()}")
    # Check that features aren't all the same across spatial positions
    spatial_var = feat.var(dim=[-2, -1]).mean().item()
    log(f"NOHEAD spatial_variance={spatial_var:.6f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--lr-sweep", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--seeds", action="store_true")
    parser.add_argument("--grad-deep", action="store_true")
    parser.add_argument("--no-head", action="store_true")
    parser.add_argument("--model-type", choices=["spix", "vq"], default="spix",
                        help="Backbone type (default: spix)")
    parser.add_argument("--codebook-size", type=int, default=1024,
                        help="VQ codebook size")
    parser.add_argument("--downsample-factor", type=int, default=16,
                        help="VQ downsample factor")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="VQ latent dimension (default: embed_dims)")
    parser.add_argument("--num-res-blocks", type=int, default=2,
                        help="Number of VQ residual blocks")
    args = parser.parse_args()

    if not any(v for v in vars(args).values() if v is True):
        args.all = True

    from spixrwkv7.utils import redirect_stdout_tee
    os.makedirs('results', exist_ok=True)
    with redirect_stdout_tee('results/diagnose_training.txt'):
        if args.all or args.lr_sweep:
            exp_lr_sweep()
        if args.all or args.depth:
            exp_depth()
        if args.all or args.seeds:
            exp_seeds()
        if args.all or args.grad_deep:
            exp_gradient_deep()
        if args.all or args.no_head:
            exp_no_head(model_type=args.model_type)
    print('Results saved to results/diagnose_training.txt')
