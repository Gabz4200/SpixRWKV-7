"""Full benchmark suite: inference speed + training convergence for all models.

Compares spix, vq, conv, gnn at tiny/small configurations against ViT baseline.
Uses real images from data/caltech101_classification/.
Ensures optimized C++ kernels are enabled where applicable.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from spixrwkv7 import ClassificationHead
from spixrwkv7.data.image_utils import (
    load_random_caltech101_batch,
    load_random_caltech101_image,
    load_random_caltech101_rgb,
)
from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7
from spixrwkv7.models.conv_spixrwkv7 import create_conv_vision_rwkv7
from spixrwkv7.models.gnn_spixrwkv7 import create_gnn_vision
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7
from spixrwkv7.models.hybrid_spixrwkv7 import create_hybrid_vision


# =====================================================================
# Config & Helpers
# =====================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)["model"]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =====================================================================
# ViT Baseline
# =====================================================================

class SimpleViT(nn.Module):
    """ViT baseline with DINOv2-style register tokens for fair GNN comparison."""

    def __init__(self, img_size, patch_size, in_chans, num_classes,
                 embed_dim, depth, num_heads, register_tokens=0):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = register_tokens
        # pos_embed covers: register tokens + cls token + patch tokens
        seq_len = register_tokens + 1 + num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        if register_tokens > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, register_tokens, embed_dim))
            nn.init.xavier_uniform_(self.reg_token)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                       dim_feedforward=4 * embed_dim, batch_first=True)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.pos_embed)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.register_tokens > 0:
            reg = self.reg_token.expand(B, -1, -1)
            x = torch.cat([reg, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        # Skip register tokens (first R) and cls token (first after registers) for output
        if self.register_tokens > 0:
            x = x[:, self.register_tokens:]
        return self.head(x[:, 0])


VIT_CONFIGS = {
    "tiny":   {"embed_dim": 192, "depth": 12, "num_heads": 3, "register_tokens": 4},
    "small":  {"embed_dim": 384, "depth": 12, "num_heads": 6, "register_tokens": 4},
}


def get_vit_model(size, img_size, num_classes=3, in_chans=6):
    cfg = VIT_CONFIGS[size]
    model = SimpleViT(img_size=img_size, patch_size=16, in_chans=in_chans,
                      num_classes=num_classes, **cfg)
    model.eval()
    return model


# =====================================================================
# Model builders (with use_cpp=True enforced)
# =====================================================================

def build_rwkv_model(model_type, size, config, img_size):
    """Build an RWKV variant with C++ kernels enabled."""
    embed_dims = config["embed_dims"]
    depth = config["depth"]
    num_heads = config["num_heads"]
    num_superpixels = config.get("num_superpixels", 0)
    # Force C++ kernels on
    cfg = {**config, "use_cpp": True}

    if model_type == "vq":
        return create_vq_rwkv7(
            img_size=img_size, embed_dims=embed_dims, num_heads=num_heads,
            depth=depth, init_values=cfg.get("init_values", 1e-5),
            final_norm=cfg.get("final_norm", True),
            out_indices=cfg.get("out_indices", [-1]),
            scatter_output=cfg.get("scatter_output", False),
            codebook_size=cfg.get("codebook_size", 1024),
            downsample_factor=cfg.get("downsample_factor", 16),
            latent_dim=cfg.get("latent_dim", None),
            num_res_blocks=cfg.get("num_res_blocks", 2),
            use_ema=False, beta=0.25,
            norm_layer=cfg.get("norm_layer", "rmsnorm"),
            act_layer=cfg.get("act_layer", "swiglu"),
            use_cpp=True,
        )
    if model_type == "conv":
        return create_conv_vision_rwkv7(
            img_size=img_size, embed_dims=embed_dims, num_heads=num_heads,
            depth=depth, init_values=cfg.get("init_values", 1e-5),
            final_norm=cfg.get("final_norm", True),
            out_indices=cfg.get("out_indices", [-1]),
            num_superpixels=num_superpixels,
            scatter_output=cfg.get("scatter_output", True),
            diff_slic_iters=cfg.get("diff_slic_iters", 5),
            compactness=cfg.get("compactness", 0.5),
            norm_layer=cfg.get("norm_layer", "rmsnorm"),
            act_layer=cfg.get("act_layer", "swiglu"),
            spixel_backend=cfg.get("spixel_backend", "diff_slic"),
            use_cpp=True,
            conv_stem_channels=tuple(cfg.get("conv_stem_channels", [32, 64, 128])),
            conv_stem_kernel_sizes=tuple(cfg.get("conv_stem_kernel_sizes", [3, 5, 5])),
            conv_stem_strides=tuple(cfg.get("conv_stem_strides", [1, 2, 2])),
            conv_stem_norm=cfg.get("conv_stem_norm", "batchnorm2d"),
            conv_post_norm=cfg.get("conv_post_norm", "layernorm"),
        )
    if model_type == "gnn":
        return create_gnn_vision(
            img_size=img_size, embed_dims=embed_dims, num_heads=num_heads,
            depth=depth, init_values=cfg.get("init_values", 1e-5),
            final_norm=cfg.get("final_norm", True),
            out_indices=cfg.get("out_indices", [-1]),
            register_tokens=cfg.get("register_tokens", 0),
            num_superpixels=num_superpixels,
            scatter_output=cfg.get("scatter_output", True),
            diff_slic_iters=cfg.get("diff_slic_iters", 5),
            compactness=cfg.get("compactness", 0.5),
            norm_layer=cfg.get("norm_layer", "rmsnorm"),
            act_layer=cfg.get("act_layer", "swiglu"),
            spixel_backend=cfg.get("spixel_backend", "diff_slic"),
            downsample_factor=float(cfg.get("downsample_factor", 16)),
            gnn_conv=cfg.get("gnn_conv", "gatv2"),
            gnn_heads=cfg.get("gnn_heads", 4),
            gnn_aggr=cfg.get("gnn_aggr", "mean"),
            use_cpp=True,
        )
    if model_type == "hybrid":
        return create_hybrid_vision(
            img_size=img_size, embed_dims=embed_dims, num_heads=num_heads,
            depth=depth, init_values=cfg.get("init_values", 1e-5),
            final_norm=cfg.get("final_norm", True),
            out_indices=cfg.get("out_indices", [-1]),
            register_tokens=cfg.get("register_tokens", 4),
            num_superpixels=num_superpixels,
            scatter_output=cfg.get("scatter_output", True),
            diff_slic_iters=cfg.get("diff_slic_iters", 5),
            compactness=cfg.get("compactness", 0.5),
            norm_layer=cfg.get("norm_layer", "rmsnorm"),
            act_layer=cfg.get("act_layer", "swiglu"),
            spixel_backend=cfg.get("spixel_backend", "diff_slic"),
            downsample_factor=float(cfg.get("downsample_factor", 16)),
            num_rwkv_layers=cfg.get("num_rwkv_layers", 1),
            num_gnn_layers=cfg.get("num_gnn_layers", 3),
            knn_k=cfg.get("knn_k", 4),
            dedup_neighbors=cfg.get("dedup_neighbors", True),
            dedup_centroids=cfg.get("dedup_centroids", True),
            gnn_conv=cfg.get("gnn_conv", "gatv2"),
            gnn_heads=cfg.get("gnn_heads", 4),
            gnn_aggr=cfg.get("gnn_aggr", "mean"),
            use_cpp=True,
        )
    # spix (default)
    return create_optimized_vision_rwkv7(
        img_size=img_size, embed_dims=embed_dims, num_heads=num_heads,
        depth=depth, init_values=cfg.get("init_values", 1e-5),
        final_norm=cfg.get("final_norm", True),
        out_indices=cfg.get("out_indices", [-1]),
        num_superpixels=num_superpixels,
        scatter_output=cfg.get("scatter_output", True),
        diff_slic_iters=cfg.get("diff_slic_iters", 5),
        compactness=cfg.get("compactness", 0.5),
        norm_layer=cfg.get("norm_layer", "rmsnorm"),
        act_layer=cfg.get("act_layer", "swiglu"),
        use_cpp=True,
    )


# =====================================================================
# Inference Speed Benchmark
# =====================================================================

def benchmark_inference(model, input_tensor, warmup=5, runs=20, device="cpu"):
    model = model.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return {
        "avg_ms": t.mean().item(),
        "std_ms": t.std().item(),
        "min_ms": t.min().item(),
    }


def benchmark_rwkv_with_breakdown(model, input_tensor, warmup=5, runs=20, device="cpu"):
    """Benchmark RWKV model with tokenizer/backbone time breakdown."""
    model = model.to(device).eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()

    # Time tokenizer separately
    tokenizer = getattr(model, "tokenizer", None)
    conv_stem = getattr(model, "conv_stem", None)
    tok_times = []
    if tokenizer is not None:
        with torch.no_grad():
            x_feat = conv_stem(input_tensor) if conv_stem is not None else None
            for _ in range(runs):
                t0 = time.perf_counter()
                if conv_stem is not None:
                    _ = tokenizer(input_tensor, x_feat)
                else:
                    _ = tokenizer(input_tensor)
                if device == "cuda":
                    torch.cuda.synchronize()
                tok_times.append((time.perf_counter() - t0) * 1000)

    # Time full forward
    full_times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            full_times.append((time.perf_counter() - t0) * 1000)

    ft = torch.tensor(full_times)
    tt = torch.tensor(tok_times) if tok_times else torch.tensor([0.0])
    return {
        "avg_ms": ft.mean().item(),
        "std_ms": ft.std().item(),
        "tokenizer_ms": tt.mean().item(),
        "backbone_ms": ft.mean().item() - tt.mean().item(),
    }


# =====================================================================
# Training Convergence Benchmark
# =====================================================================

def train_convergence(model_type, size, config, img_size, device,
                      max_steps=300, lr=5e-4, batch_size=4, num_classes=3):
    """Train a model on real images and measure convergence speed."""
    backbone = build_rwkv_model(model_type, size, config, img_size)
    backbone = backbone.to(device)

    head = ClassificationHead(config["embed_dims"], num_classes).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    # Load real training data (re-used each step for single-batch overfit test)
    x_train, y_train = load_random_caltech101_batch(
        batch_size=batch_size, img_size=img_size, num_classes=num_classes,
        device=device, seed=42,
    )

    total_params = sum(p.numel() for p in params)
    step_times = []
    accs = []
    losses = []
    best_acc = 0.0

    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        step_t0 = time.perf_counter()

        backbone.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)

        outs = backbone(x_train)
        feat = outs[0]
        logits = head(feat)
        loss = F.cross_entropy(logits, y_train)

        # Add VQ quantization loss if present
        q_loss = getattr(backbone, "_last_q_loss", None)
        if q_loss is not None:
            loss = loss + q_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        acc = (logits.argmax(1) == y_train).float().mean().item() * 100
        step_times.append(time.perf_counter() - step_t0)
        accs.append(acc)
        losses.append(loss.item())
        best_acc = max(best_acc, acc)

        if acc >= 95.0:
            break

    elapsed = time.perf_counter() - t0

    # Find steps to thresholds
    steps_to_90 = next((i + 1 for i, a in enumerate(accs) if a >= 90.0), max_steps)
    steps_to_95 = next((i + 1 for i, a in enumerate(accs) if a >= 95.0), max_steps)

    del backbone, head
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "total_params": total_params,
        "steps_run": len(accs),
        "final_loss": losses[-1] if losses else float("nan"),
        "final_acc": accs[-1] if accs else 0.0,
        "best_acc": best_acc,
        "steps_to_90": steps_to_90,
        "steps_to_95": steps_to_95,
        "total_time_s": elapsed,
        "avg_step_ms": sum(step_times) / len(step_times) * 1000 if step_times else 0,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Full SpixRWKV-7 benchmark suite")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--sizes", nargs="+", default=["tiny", "small"])
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--train-batch", type=int, default=4)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_dir = Path("configs/model")
    model_types = ["spix", "vq", "conv", "gnn", "hybrid"]
    num_classes = 3  # caltech101_classification: butterfly, dalmatian, dolphin

    print("=" * 78)
    print("  SpixRWKV-7 FULL BENCHMARK SUITE")
    print("=" * 78)
    print(f"  Device:     {device}")
    print(f"  Sizes:      {args.sizes}")
    print(f"  Img size:   {args.img_size}")
    print(f"  Classes:    {num_classes} (caltech101: butterfly, dalmatian, dolphin)")
    print(f"  C++ kernels: ENABLED (where applicable)")
    print("=" * 78)

    all_inf_results = []
    all_train_results = []

    for size in args.sizes:
        print(f"\n{'=' * 78}")
        print(f"  SIZE: {size.upper()}")
        print(f"{'=' * 78}")

        # ── INFERENCE SPEED ──
        if not args.skip_inference:
            print(f"\n  {'─' * 74}")
            print(f"  INFERENCE SPEED BENCHMARK")
            print(f"  {'─' * 74}")

            # Build RWKV models
            for mt in model_types:
                config_file = f"{'conv_' if mt == 'conv' else 'gnn_' if mt == 'gnn' else 'vq_' if mt == 'vq' else 'hybrid_' if mt == 'hybrid' else ''}{size}.yaml"
                config_path = config_dir / config_file
                if not config_path.exists():
                    print(f"  [{mt}] Config not found: {config_path}, skipping")
                    continue
                config = load_config(str(config_path))

                print(f"\n  [{mt.upper()}] Building model...")
                try:
                    model = build_rwkv_model(mt, size, config, args.img_size)
                    params = count_parameters(model)

                    # Load real image
                    rwkv_input, _, _ = load_random_caltech101_image(
                        img_size=args.img_size, seed=42,
                    )

                    print(f"  [{mt.upper()}] Params: {params / 1e6:.2f}M")
                    metrics = benchmark_rwkv_with_breakdown(
                        model, rwkv_input, args.warmup, args.runs, device,
                    )
                    tok_pct = 100 * metrics["tokenizer_ms"] / metrics["avg_ms"] if metrics["avg_ms"] > 0 else 0

                    print(f"  [{mt.upper()}] Total: {metrics['avg_ms']:.2f}ms "
                          f"(±{metrics['std_ms']:.2f})")
                    print(f"    Tokenizer: {metrics['tokenizer_ms']:.2f}ms ({tok_pct:.0f}%)")
                    print(f"    Backbone:  {metrics['backbone_ms']:.2f}ms")

                    all_inf_results.append({
                        "size": size, "model_type": mt, "params": params,
                        **metrics, "tok_pct": tok_pct,
                    })
                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  [{mt.upper()}] ERROR: {e}")

            # ViT baseline
            print(f"\n  [ViT] Building baseline...")
            vit = get_vit_model(size, args.img_size, num_classes=num_classes, in_chans=6)
            vit_params = count_parameters(vit)
            # Use 6-channel input (same as RWKV models)
            vit_input, _, _ = load_random_caltech101_image(
                img_size=args.img_size, seed=42,
            )
            print(f"  [ViT] Params: {vit_params / 1e6:.2f}M")
            vit_metrics = benchmark_inference(
                vit, vit_input, args.warmup, args.runs, device,
            )
            print(f"  [ViT] Total: {vit_metrics['avg_ms']:.2f}ms "
                  f"(±{vit_metrics['std_ms']:.2f})")
            all_inf_results.append({
                "size": size, "model_type": "vit", "params": vit_params,
                "avg_ms": vit_metrics["avg_ms"], "std_ms": vit_metrics["std_ms"],
                "tokenizer_ms": 0, "backbone_ms": vit_metrics["avg_ms"],
                "tok_pct": 0,
            })
            del vit
            if device == "cuda":
                torch.cuda.empty_cache()

            # Speedup summary
            vit_time = vit_metrics["avg_ms"]
            print(f"\n  INFERENCE SPEEDUP vs ViT ({size.upper()}):")
            for r in all_inf_results:
                if r["size"] == size and r["model_type"] != "vit":
                    speedup = vit_time / r["avg_ms"] if r["avg_ms"] > 0 else 0
                    print(f"    {r['model_type']:>6}: {speedup:.2f}x")

        # ── TRAINING CONVERGENCE ──
        if not args.skip_training:
            print(f"\n  {'─' * 74}")
            print(f"  TRAINING CONVERGENCE BENCHMARK (single-batch overfit)")
            print(f"  {'─' * 74}")

            for mt in model_types:
                config_file = f"{'conv_' if mt == 'conv' else 'gnn_' if mt == 'gnn' else 'vq_' if mt == 'vq' else 'hybrid_' if mt == 'hybrid' else ''}{size}.yaml"
                config_path = config_dir / config_file
                if not config_path.exists():
                    print(f"  [{mt}] Config not found, skipping")
                    continue
                config = load_config(str(config_path))

                print(f"\n  [{mt.upper()}] Training for up to {args.train_steps} steps...")
                try:
                    result = train_convergence(
                        mt, size, config, args.img_size, device,
                        max_steps=args.train_steps,
                        batch_size=args.train_batch,
                        num_classes=num_classes,
                    )
                    print(f"  [{mt.upper()}] Steps: {result['steps_run']}, "
                          f"Best acc: {result['best_acc']:.1f}%, "
                          f"Final loss: {result['final_loss']:.4f}")
                    print(f"    Steps to 90%: {result['steps_to_90']}, "
                          f"Steps to 95%: {result['steps_to_95']}")
                    print(f"    Total time: {result['total_time_s']:.1f}s, "
                          f"Step time: {result['avg_step_ms']:.0f}ms")
                    all_train_results.append({"size": size, "model_type": mt, **result})
                except Exception as e:
                    print(f"  [{mt.upper()}] ERROR: {e}")

            # ViT training
            print(f"\n  [ViT] Training baseline...")
            try:
                vit_model = get_vit_model(size, args.img_size, num_classes=num_classes, in_chans=6)
                vit_model = vit_model.to(device)
                opt = torch.optim.AdamW(vit_model.parameters(), lr=5e-4)
                # Use 6-channel input (same as RWKV models)
                x_vit, y_vit = load_random_caltech101_batch(
                    batch_size=args.train_batch, img_size=args.img_size,
                    device=device, seed=42,
                )

                step_times_v = []
                accs_v = []
                best_acc_v = 0.0
                t0 = time.perf_counter()
                for step in range(1, args.train_steps + 1):
                    st0 = time.perf_counter()
                    vit_model.train()
                    opt.zero_grad(set_to_none=True)
                    logits = vit_model(x_vit)
                    loss = F.cross_entropy(logits, y_vit)
                    loss.backward()
                    opt.step()
                    acc = (logits.argmax(1) == y_vit).float().mean().item() * 100
                    step_times_v.append(time.perf_counter() - st0)
                    accs_v.append(acc)
                    best_acc_v = max(best_acc_v, acc)
                    if acc >= 95.0:
                        break
                elapsed_v = time.perf_counter() - t0
                steps_to_90_v = next((i + 1 for i, a in enumerate(accs_v) if a >= 90.0), args.train_steps)
                steps_to_95_v = next((i + 1 for i, a in enumerate(accs_v) if a >= 95.0), args.train_steps)
                vit_result = {
                    "size": size, "model_type": "vit",
                    "total_params": count_parameters(vit_model),
                    "steps_run": len(accs_v),
                    "final_loss": loss.item(),
                    "final_acc": accs_v[-1],
                    "best_acc": best_acc_v,
                    "steps_to_90": steps_to_90_v,
                    "steps_to_95": steps_to_95_v,
                    "total_time_s": elapsed_v,
                    "avg_step_ms": sum(step_times_v) / len(step_times_v) * 1000,
                }
                print(f"  [ViT] Steps: {vit_result['steps_run']}, "
                      f"Best acc: {vit_result['best_acc']:.1f}%")
                all_train_results.append(vit_result)
                del vit_model
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ViT] ERROR: {e}")

    # ── FINAL SUMMARY ──
    print(f"\n{'=' * 78}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 78}")

    if all_inf_results:
        print(f"\n  INFERENCE SPEED (img_size={args.img_size})")
        print(f"  {'Size':<8} {'Model':<8} {'Params':>10} {'Total(ms)':>12} {'Tok(ms)':>10} {'Bkb(ms)':>10}")
        print(f"  {'-' * 62}")
        for r in all_inf_results:
            print(f"  {r['size']:<8} {r['model_type']:<8} "
                  f"{r['params']/1e6:>9.2f}M {r['avg_ms']:>11.2f} "
                  f"{r.get('tokenizer_ms', 0):>9.2f} {r.get('backbone_ms', 0):>9.2f}")

        # Speedup table
        print(f"\n  SPEEDUP vs ViT:")
        print(f"  {'Size':<8} {'spix':>8} {'vq':>8} {'conv':>8} {'gnn':>8} {'hybrid':>8}")
        print(f"  {'-' * 48}")
        for size in args.sizes:
            vit_time = next((r["avg_ms"] for r in all_inf_results
                           if r["size"] == size and r["model_type"] == "vit"), None)
            if vit_time is None:
                continue
            row = f"  {size:<8}"
            for mt in ["spix", "vq", "conv", "gnn", "hybrid"]:
                mt_time = next((r["avg_ms"] for r in all_inf_results
                               if r["size"] == size and r["model_type"] == mt), None)
                if mt_time is not None and mt_time > 0:
                    row += f" {vit_time / mt_time:>7.2f}x"
                else:
                    row += f" {'N/A':>7}"
            print(row)

    if all_train_results:
        print(f"\n  TRAINING CONVERGENCE (single-batch overfit)")
        print(f"  {'Size':<8} {'Model':<8} {'Steps':>7} {'Best%':>7} {'90%@':>7} {'95%@':>7} {'Time':>8}")
        print(f"  {'-' * 54}")
        for r in all_train_results:
            print(f"  {r['size']:<8} {r['model_type']:<8} "
                  f"{r['steps_run']:>6} {r['best_acc']:>6.1f}% "
                  f"{r['steps_to_90']:>6} {r['steps_to_95']:>6} "
                  f"{r['total_time_s']:>6.1f}s")

    print(f"\n{'=' * 78}")
    print("  Benchmark complete. Results saved above.")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    from spixrwkv7.utils import redirect_stdout_tee
    os.makedirs("results", exist_ok=True)
    with redirect_stdout_tee("results/full_benchmark.txt"):
        main()
    print("Results saved to results/full_benchmark.txt")
