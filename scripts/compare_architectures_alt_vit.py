"""Compare Vision RWKV-7 vs ViT inference speed across model sizes.

This script benchmarks both architectures at equivalent parameter counts,
using the same image resolution for each size config. Optimized for CPU
execution with optional GPU support and minimal memory footprint.

Uses alternative ViT implementation with einops-based patch embedding
and sincos positional encoding (no class token, mean pooling).

Key insight: diffSLIC tokenization is the primary bottleneck in Vision RWKV-7.
When ported to optimized C++ kernel, the recurrent backbone becomes significantly
faster relative to ViT's quadratic attention.
"""

import time
import argparse
import tracemalloc
import os
from typing import Dict, Any, List

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from einops import rearrange
from einops.layers.torch import Rearrange

from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7 as _create_model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)["model"]


# ── ViT implementation with einops and sincos positional encoding ──

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    """Simple ViT using einops patch embedding and sincos positional encoding.

    No class token - uses mean pooling over patches.
    """

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


def get_vit_model(size: str, img_size: int, num_classes: int = 1000):
    """Get ViT model matching the size config.

    Maps:
    - tiny → ViT-T (embed_dim=192, depth=12, heads=3)
    - small → ViT-S (embed_dim=384, depth=12, heads=6)
    - medium → ViT-B (embed_dim=768, depth=12, heads=12)
    - large → ViT-L (embed_dim=1024, depth=24, heads=16)
    """
    vit_configs = {
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "medium": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    }

    cfg = vit_configs[size]
    model = SimpleViT(
        image_size=img_size,
        patch_size=16,
        channels=3,
        num_classes=num_classes,
        dim=cfg["embed_dim"],
        depth=cfg["depth"],
        heads=cfg["num_heads"],
        mlp_dim=4 * cfg["embed_dim"],
    )
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_memory_usage(model: nn.Module, input_tensor: torch.Tensor) -> float:
    """Get peak memory usage in MB during forward pass."""
    tracemalloc.start()
    with torch.no_grad():
        _ = model(input_tensor)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)  # Convert to MB


def benchmark_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = 5,
    timed_runs: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmark model inference speed."""
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    times_tensor = torch.tensor(times)
    return {
        "avg_time_ms": times_tensor.mean().item(),
        "std_time_ms": times_tensor.std().item(),
        "min_time_ms": times_tensor.min().item(),
        "max_time_ms": times_tensor.max().item(),
    }


def benchmark_rwkv_components(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = 5,
    timed_runs: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmark Vision RWKV-7 with diffSLIC timing breakdown.

    Returns total time plus tokenizer-only time for diffSLIC analysis.
    """
    model = model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()

    # Time tokenizer (diffSLIC) separately
    tokenizer_times = []
    with torch.no_grad():
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model.tokenizer(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            tokenizer_times.append((end - start) * 1000)

    # Time full forward pass
    full_times = []
    with torch.no_grad():
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            full_times.append((end - start) * 1000)

    tokenizer_tensor = torch.tensor(tokenizer_times)
    full_tensor = torch.tensor(full_times)

    return {
        "avg_time_ms": full_tensor.mean().item(),
        "std_time_ms": full_tensor.std().item(),
        "tokenizer_time_ms": tokenizer_tensor.mean().item(),
        "backbone_time_ms": full_tensor.mean().item() - tokenizer_tensor.mean().item(),
    }


def create_dummy_input(img_size: int, channels: int = 3, batch_size: int = 1) -> torch.Tensor:
    """Create dummy input tensor."""
    return torch.randn(batch_size, channels, img_size, img_size)


def run_size_comparison(
    sizes: List[str],
    config_dir: Path,
    device: str,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Run comparison across model sizes at their configured image resolutions."""
    results = []

    for size in sizes:
        config_path = config_dir / f"{size}.yaml"
        config = load_config(str(config_path))

        img_size = 512
        embed_dims = config["embed_dims"]
        depth = config["depth"]
        num_heads = config["num_heads"]
        num_superpixels = config["num_superpixels"]

        print(f"\n--- {size.upper()} Config ---")
        print(f"  img_size: {img_size}")
        print(f"  embed_dims: {embed_dims}")
        print(f"  depth: {depth}")
        print(f"  num_heads: {num_heads}")
        print(f"  num_superpixels: {num_superpixels}")

        # Create Vision RWKV-7 model
        rwkv_model = _create_model(
            img_size=img_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            init_values=config.get("init_values", 1e-5),
            final_norm=config.get("final_norm", True),
            out_indices=config.get("out_indices", [-1]),
            num_superpixels=num_superpixels,
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            norm_layer=config.get("norm_layer", "layernorm"),
            act_layer=config.get("act_layer", "relu2"),
        )

        rwkv_params = count_parameters(rwkv_model)
        print(f"  Vision RWKV-7 params: {rwkv_params / 1e6:.2f}M")

        # Create ViT model
        vit_model = get_vit_model(size, img_size)
        vit_params = count_parameters(vit_model)
        print(f"  ViT params: {vit_params / 1e6:.2f}M")

        # Create input tensors
        rwkv_input = create_dummy_input(img_size, channels=6, batch_size=batch_size)
        vit_input = create_dummy_input(img_size, channels=3, batch_size=batch_size)

        # Memory usage
        rwkv_mem = get_memory_usage(rwkv_model, rwkv_input)
        vit_mem = get_memory_usage(vit_model, vit_input)
        print(f"  Vision RWKV-7 peak memory: {rwkv_mem:.2f}MB")
        print(f"  ViT peak memory: {vit_mem:.2f}MB")

        # Benchmark Vision RWKV-7 with component breakdown
        print(f"\n  Benchmarking Vision RWKV-7...")
        rwkv_metrics = benchmark_rwkv_components(
            rwkv_model, rwkv_input, warmup_runs, timed_runs, device
        )
        print(f"    Total: {rwkv_metrics['avg_time_ms']:.2f}ms")
        print(f"    diffSLIC (tokenizer): {rwkv_metrics['tokenizer_time_ms']:.2f}ms")
        print(f"    Backbone (recurrent): {rwkv_metrics['backbone_time_ms']:.2f}ms")

        # Benchmark ViT
        print(f"  Benchmarking ViT...")
        vit_metrics = benchmark_model(
            vit_model, vit_input, warmup_runs, timed_runs, device
        )
        print(f"    Total: {vit_metrics['avg_time_ms']:.2f}ms")

        speedup = vit_metrics["avg_time_ms"] / rwkv_metrics["avg_time_ms"]
        print(f"\n  Speed ratio (ViT/RWKV-7): {speedup:.2f}x")
        if speedup > 1:
            print(f"    Vision RWKV-7 is {speedup:.2f}x FASTER than ViT")
        else:
            print(f"    ViT is {1/speedup:.2f}x FASTER than Vision RWKV-7")

        results.append({
            "size": size,
            "img_size": img_size,
            "rwkv_params": rwkv_params,
            "vit_params": vit_params,
            "rwkv_time": rwkv_metrics["avg_time_ms"],
            "rwkv_tokenizer": rwkv_metrics["tokenizer_time_ms"],
            "rwkv_backbone": rwkv_metrics["backbone_time_ms"],
            "vit_time": vit_metrics["avg_time_ms"],
            "speedup": speedup,
            "rwkv_mem_mb": rwkv_mem,
            "vit_mem_mb": vit_mem,
        })

        del rwkv_model, vit_model, rwkv_input, vit_input
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def run_resolution_sweep(
    size: str,
    img_sizes: List[int],
    device: str,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
):
    """Run comparison for a single model size across increasing image resolutions."""
    print(f"\n{'=' * 70}")
    print(f"RESOLUTION SWEEP FOR {size.upper()} MODEL")
    print(f"{'=' * 70}")

    config_dir = Path("configs/model")
    config = load_config(str(config_dir / f"{size}.yaml"))

    embed_dims = config["embed_dims"]
    depth = config["depth"]
    num_heads = config["num_heads"]
    num_superpixels = config["num_superpixels"]

    print(f"Model config: embed_dims={embed_dims}, depth={depth}, num_heads={num_heads}")
    print(f"Testing image sizes: {img_sizes}")

    results = []

    for img_size in img_sizes:
        print(f"\n  --- Image size: {img_size}x{img_size} ---")

        # Create Vision RWKV-7 model
        rwkv_model = _create_model(
            img_size=img_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            init_values=config.get("init_values", 1e-5),
            final_norm=config.get("final_norm", True),
            out_indices=config.get("out_indices", [-1]),
            num_superpixels=num_superpixels,
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            norm_layer=config.get("norm_layer", "layernorm"),
            act_layer=config.get("act_layer", "relu2"),
        )

        rwkv_params = count_parameters(rwkv_model)
        print(f"    Vision RWKV-7 params: {rwkv_params / 1e6:.2f}M")

        # Create ViT model
        vit_model = get_vit_model(size, img_size)
        vit_params = count_parameters(vit_model)
        print(f"    ViT params: {vit_params / 1e6:.2f}M")

        # Create input tensors
        rwkv_input = create_dummy_input(img_size, channels=6, batch_size=batch_size)
        vit_input = create_dummy_input(img_size, channels=3, batch_size=batch_size)

        # Memory usage
        rwkv_mem = get_memory_usage(rwkv_model, rwkv_input)
        vit_mem = get_memory_usage(vit_model, vit_input)
        print(f"    RWKV-7 peak memory: {rwkv_mem:.2f}MB")
        print(f"    ViT peak memory: {vit_mem:.2f}MB")

        # Benchmark Vision RWKV-7 with component breakdown
        rwkv_metrics = benchmark_rwkv_components(
            rwkv_model, rwkv_input, warmup_runs, timed_runs, device
        )
        print(f"    RWKV-7 Total: {rwkv_metrics['avg_time_ms']:.2f}ms")
        print(f"    RWKV-7 diffSLIC: {rwkv_metrics['tokenizer_time_ms']:.2f}ms")
        print(f"    RWKV-7 Backbone: {rwkv_metrics['backbone_time_ms']:.2f}ms")

        # Benchmark ViT
        vit_metrics = benchmark_model(
            vit_model, vit_input, warmup_runs, timed_runs, device
        )
        print(f"    ViT Total: {vit_metrics['avg_time_ms']:.2f}ms")

        speedup = vit_metrics["avg_time_ms"] / rwkv_metrics["avg_time_ms"]
        print(f"    Speed ratio (ViT/RWKV-7): {speedup:.2f}x")

        results.append({
            "img_size": img_size,
            "rwkv_time": rwkv_metrics["avg_time_ms"],
            "rwkv_tokenizer": rwkv_metrics["tokenizer_time_ms"],
            "rwkv_backbone": rwkv_metrics["backbone_time_ms"],
            "vit_time": vit_metrics["avg_time_ms"],
            "speedup": speedup,
            "rwkv_mem_mb": rwkv_mem,
            "vit_mem_mb": vit_mem,
        })

        del rwkv_model, vit_model, rwkv_input, vit_input
        if device == "cuda":
            torch.cuda.empty_cache()

    # Print resolution sweep table
    print(f"\n  Resolution sweep results:")
    print(f"  {'Img':<8} {'RWKV-7':<12} {'ViT':<12} {'Speedup':<10} {'Tokenizer %':<12} {'Mem RWKV':<10} {'Mem ViT':<10}")
    print(f"  {'-' * 74}")
    for r in results:
        tok_pct = 100 * r["rwkv_tokenizer"] / r["rwkv_time"]
        print(f"  {r['img_size']:<8} {r['rwkv_time']:<12.2f} {r['vit_time']:<12.2f} "
              f"{r['speedup']:<10.2f}x {tok_pct:<12.1f} {r['rwkv_mem_mb']:<10.2f} {r['vit_mem_mb']:<10.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Vision RWKV-7 vs ViT speed (alt ViT)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup runs (default: 5)"
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="Number of timed runs (default: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--skip-large", action="store_true", help="Skip large config (slow on CPU)"
    )
    parser.add_argument(
        "--resolution-sweep",
        nargs="+",
        type=int,
        default=[64, 128, 512],
        help="Image sizes for resolution sweep (default: 64 128 512)",
    )
    parser.add_argument(
        "--sweep-size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Model size for resolution sweep (default: small)",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {device}")
    print(f"Warmup runs: {args.warmup}, Timed runs: {args.runs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    config_dir = Path("configs/model")
    sizes = ["tiny", "small", "medium", "large"]
    if args.skip_large:
        sizes = ["tiny", "small", "medium"]

    # Part 1: Model size comparison at configured resolutions
    print("\n" + "=" * 70)
    print("PART 1: MODEL SIZE COMPARISON")
    print("=" * 70)
    size_results = run_size_comparison(
        sizes, config_dir, device, args.warmup, args.runs, args.batch_size
    )

    # Part 2: Resolution sweep for selected model size
    print("\n" + "=" * 70)
    print("PART 2: RESOLUTION SWEEP")
    print("=" * 70)
    run_resolution_sweep(
        args.sweep_size, args.resolution_sweep, device,
        args.warmup, args.runs, args.batch_size
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Size':<8} {'Img':<6} {'RWKV-7':<10} {'ViT':<10} {'Speedup':<8}")
    print("-" * 70)
    for r in size_results:
        print(f"{r['size']:<8} {r['img_size']:<6} {r['rwkv_time']:<10.2f} {r['vit_time']:<10.2f} "
              f"{r['speedup']:<8.2f}x")

    # diffSLIC breakdown analysis
    print("\n" + "=" * 70)
    print("DIFFSLIC BREAKDOWN ANALYSIS")
    print("=" * 70)
    print("This shows what fraction of Vision RWKV-7 time is spent in diffSLIC.")
    print("When diffSLIC is ported to optimized C++, the backbone speedup")
    print("relative to ViT becomes more significant.\n")
    print(f"{'Size':<8} {'Tokenizer %':<12} {'Backbone %':<12} {'Backbone/Tokenizer':<18}")
    print("-" * 70)
    for r in size_results:
        tok_pct = 100 * r["rwkv_tokenizer"] / r["rwkv_time"]
        back_pct = 100 * r["rwkv_backbone"] / r["rwkv_time"]
        ratio = r["rwkv_backbone"] / r["rwkv_tokenizer"] if r["rwkv_tokenizer"] > 0 else 0
        print(f"{r['size']:<8} {tok_pct:<12.1f} {back_pct:<12.1f} {ratio:<18.2f}x")

    # Memory comparison
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON")
    print("=" * 70)
    print(f"{'Size':<8} {'RWKV-7 (MB)':<14} {'ViT (MB)':<14} {'Ratio':<10}")
    print("-" * 70)
    for r in size_results:
        ratio = r["vit_mem_mb"] / r["rwkv_mem_mb"] if r["rwkv_mem_mb"] > 0 else 0
        print(f"{r['size']:<8} {r['rwkv_mem_mb']:<14.2f} {r['vit_mem_mb']:<14.2f} {ratio:<10.2f}x")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    fastest = min(size_results, key=lambda r: r["rwkv_time"])
    print(f"Fastest Vision RWKV-7: {fastest['size']} ({fastest['rwkv_time']:.2f}ms)")

    print("\nParameter efficiency (time per M params):")
    for r in size_results:
        rwkv_eff = r["rwkv_time"] / (r["rwkv_params"] / 1e6)
        vit_eff = r["vit_time"] / (r["vit_params"] / 1e6)
        print(f"  {r['size']:<8}: RWKV-7={rwkv_eff:.4f}ms/M, ViT={vit_eff:.4f}ms/M")

    print("\nKey findings:")
    for r in size_results:
        if r["rwkv_tokenizer"] > r["rwkv_backbone"]:
            print(f"  {r['size']}: diffSLIC dominates ({100*r['rwkv_tokenizer']/r['rwkv_time']:.0f}% of time)")
            print(f"    -> C++ diffSLIC kernel would significantly improve RWKV-7 speed")
        else:
            print(f"  {r['size']}: Backbone dominates ({100*r['rwkv_backbone']/r['rwkv_time']:.0f}% of time)")


if __name__ == "__main__":
    from spixrwkv7.utils import redirect_stdout_tee
    os.makedirs('results', exist_ok=True)
    with redirect_stdout_tee('results/compare_architectures_alt_vit.txt'):
        main()
    print('Results saved to results/compare_architectures_alt_vit.txt')