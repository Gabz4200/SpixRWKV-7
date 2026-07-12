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

import argparse
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from einops import rearrange
from einops.layers.torch import Rearrange

from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7 as _create_model
from spixrwkv7.models.conv_spixrwkv7 import create_conv_vision_rwkv7
from spixrwkv7.models.gnn_spixrwkv7 import create_gnn_vision_rwkv7
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7


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

    # Time tokenizer separately
    tokenizer_times = []
    tokenizer = getattr(model, "tokenizer", None)
    conv_stem = getattr(model, "conv_stem", None)
    if tokenizer is not None:
        with torch.no_grad():
            x_feat = conv_stem(input_tensor) if conv_stem is not None else None
            for _ in range(timed_runs):
                start = time.perf_counter()
                if conv_stem is not None:
                    _ = tokenizer(input_tensor, x_feat)
                else:
                    _ = tokenizer(input_tensor)
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


def build_rwkv(
    model_type: str,
    size: str,
    config: Dict[str, Any],
    img_size: int,
    use_parallel: bool = False,
    hard_mode: bool = False,
    downsample_factor: Optional[float] = None,
):
    embed_dims = config["embed_dims"]
    depth = config["depth"]
    num_heads = config["num_heads"]
    num_superpixels = config.get("num_superpixels", 0)

    if model_type == "vq":
        return create_vq_rwkv7(
            img_size=img_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            init_values=config.get("init_values", 1e-5),
            final_norm=config.get("final_norm", True),
            out_indices=config.get("out_indices", [-1]),
            with_cls_token=config.get("with_cls_token", False),
            output_cls_token=config.get("output_cls_token", False),
            register_tokens=config.get("register_tokens", 0),
            scatter_output=config.get("scatter_output", False),
            drop_path_rate=config.get("drop_path_rate", 0.0),
            codebook_size=config.get("codebook_size", 1024),
            downsample_factor=config.get("downsample_factor", 16),
            latent_dim=config.get("latent_dim", None),
            num_res_blocks=config.get("num_res_blocks", 2),
            use_ema=config.get("use_ema", False),
            beta=config.get("beta", 0.25),
            norm_layer=config.get("norm_layer", "rmsnorm"),
            act_layer=config.get("act_layer", "swiglu"),
            use_attnres=config.get("use_attnres", False),
            attnres_mode=config.get("attnres_mode", "block"),
            attnres_gate_type=config.get("attnres_gate_type", "bias"),
            attnres_num_blocks=config.get("attnres_num_blocks", 8),
            attnres_recency_bias_init=config.get("attnres_recency_bias_init", 10.0),
        )
    if model_type == "conv":
        return create_conv_vision_rwkv7(
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
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            use_jit=config.get("use_jit", False),
            conv_stem_channels=tuple(config.get("conv_stem_channels", [32, 64, 128])),
            conv_stem_kernel_sizes=tuple(config.get("conv_stem_kernel_sizes", [3, 5, 5])),
            conv_stem_strides=tuple(config.get("conv_stem_strides", [1, 2, 2])),
            conv_stem_norm=config.get("conv_stem_norm", "layernorm"),
            conv_post_norm=config.get("conv_post_norm", "layernorm"),
            with_cls_token=config.get("with_cls_token", False),
            output_cls_token=config.get("output_cls_token", False),
            register_tokens=config.get("register_tokens", 0),
            spixel_size=config.get("spixel_size", None),
            use_attnres=config.get("use_attnres", False),
            attnres_mode=config.get("attnres_mode", "block"),
            attnres_gate_type=config.get("attnres_gate_type", "bias"),
            attnres_num_blocks=config.get("attnres_num_blocks", 8),
            attnres_recency_bias_init=config.get("attnres_recency_bias_init", 10.0),
            use_cpp=config.get("use_cpp", False),
        )
    if model_type == "gnn":
        return create_gnn_vision_rwkv7(
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
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            downsample_factor=float(downsample_factor) if downsample_factor is not None else config.get("downsample_factor", 16),
            gnn_conv=config.get("gnn_conv", "gatv2"),
            gnn_heads=config.get("gnn_heads", 4),
            gnn_aggr=config.get("gnn_aggr", "mean"),
            with_cls_token=config.get("with_cls_token", False),
            output_cls_token=config.get("output_cls_token", False),
            register_tokens=config.get("register_tokens", 0),
            spixel_size=config.get("spixel_size", None),
            use_cpp=config.get("use_cpp", False),
            use_jit=config.get("use_jit", False),
        )
    kwargs = {
        "img_size": img_size,
        "embed_dims": embed_dims,
        "num_heads": num_heads,
        "depth": depth,
        "init_values": config.get("init_values", 1e-5),
        "final_norm": config.get("final_norm", True),
        "out_indices": config.get("out_indices", [-1]),
        "num_superpixels": num_superpixels,
        "scatter_output": config.get("scatter_output", True),
        "diff_slic_iters": config.get("diff_slic_iters", 5),
        "compactness": config.get("compactness", 0.5),
        "norm_layer": config.get("norm_layer", "layernorm"),
        "act_layer": config.get("act_layer", "relu2"),
        "with_cls_token": config.get("with_cls_token", False),
        "output_cls_token": config.get("output_cls_token", False),
        "register_tokens": config.get("register_tokens", 0),
        "spixel_size": config.get("spixel_size", None),
        "use_attnres": config.get("use_attnres", False),
        "attnres_mode": config.get("attnres_mode", "block"),
        "attnres_gate_type": config.get("attnres_gate_type", "bias"),
        "attnres_num_blocks": config.get("attnres_num_blocks", 8),
        "attnres_recency_bias_init": config.get("attnres_recency_bias_init", 10.0),
        "use_jit": config.get("use_jit", False),
        "use_parallel": use_parallel,
    }
    if downsample_factor is not None:
        kwargs["downsample_factor"] = float(downsample_factor)
    return _create_model(**kwargs)


def benchmark_variant(
    model_type: str,
    size: str,
    config: Dict[str, Any],
    img_size: int,
    device: str,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
    use_parallel: bool = False,
    hard_mode: bool = False,
    downsample_factor: Optional[float] = None,
) -> Dict[str, Any]:
    model = build_rwkv(
        model_type, size, config, img_size,
        use_parallel=use_parallel, hard_mode=hard_mode,
        downsample_factor=downsample_factor,
    )
    if hard_mode and model_type not in ("vq", "conv"):
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            setattr(tokenizer, "mode", "hard")
            diff_slic = getattr(tokenizer, "diff_slic", None)
            if diff_slic is not None:
                setattr(diff_slic, "hard_mode", True)

    rwkv_params = count_parameters(model)
    vit_model = get_vit_model(size, img_size)
    vit_params = count_parameters(vit_model)
    rwkv_input = create_dummy_input(img_size, channels=6, batch_size=batch_size)
    vit_input = create_dummy_input(img_size, channels=3, batch_size=batch_size)
    rwkv_mem = get_memory_usage(model, rwkv_input)
    vit_mem = get_memory_usage(vit_model, vit_input)
    rwkv_metrics = benchmark_rwkv_components(model, rwkv_input, warmup_runs, timed_runs, device)
    vit_metrics = benchmark_model(vit_model, vit_input, warmup_runs, timed_runs, device)
    speedup = vit_metrics["avg_time_ms"] / rwkv_metrics["avg_time_ms"]
    result = {
        "size": size,
        "model_type": model_type,
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
    }
    del model, vit_model, rwkv_input, vit_input
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def run_size_comparison(
    sizes: List[str],
    config_dir: Path,
    device: str,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
    model_type: str = "spix",
    img_size: int = 512,
    compare_variants: Optional[List[str]] = None,
    downsample_factors: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Run comparison across model sizes at a given image height."""
    results: List[Dict[str, Any]] = []
    variant_results: List[Dict[str, Any]] = []
    variants = [model_type]
    if compare_variants:
        variants = [v for v in compare_variants if v in {"spix", "conv", "vq", "gnn"}]
        if not variants:
            variants = [model_type]

    if downsample_factors is None:
        downsample_factors = [1.0]
    for size in sizes:
        print(f"\n--- {size.upper()} ---")
        for variant in variants:
            config_path = config_dir / (f"conv_{size}.yaml" if variant == "conv" else f"gnn_{size}.yaml" if variant == "gnn" else f"vq_{size}.yaml" if variant == "vq" else f"{size}.yaml")
            config = load_config(str(config_path))
            dfactors = [1.0] if variant not in ("spix", "gnn") else downsample_factors
            for downsample_factor in dfactors:
                print(f"  running {variant} downsample_factor={downsample_factor}")
                entry = benchmark_variant(
                    variant, size, config, img_size, device, warmup_runs, timed_runs,
                    batch_size,
                    downsample_factor=downsample_factor,
                )
                variant_results.append(entry)
                if variant == model_type:
                    results.append(entry)

    if compare_variants and variant_results:
        print("\n" + "=" * 70)
        print("VARIANT COMPARISON")
        print("=" * 70)
        for size in sizes:
            print(f"\n--- {size.upper()} ---")
            print(f"{'Variant':<10} {'Params(M)':<12} {'Total(ms)':<12} {'Tokenizer(ms)':<16} {'Backbone(ms)':<14} {'Speedup':<10}")
            print("-" * 74)
            size_variants = [r for r in variant_results if r["size"] == size]
            for r in size_variants:
                print(f"{r['model_type']:<10} {r['rwkv_params']/1e6:<12.2f} {r['rwkv_time']:<12.2f} "
                      f"{r['rwkv_tokenizer']:<16.2f} {r['rwkv_backbone']:<14.2f} {r['speedup']:<10.2f}x")
            best = min(size_variants, key=lambda r: r["rwkv_time"]) if size_variants else None
            if best:
                print(f"  -> Fastest variant: {best['model_type']} ({best['rwkv_time']:.2f}ms)")

    return results + variant_results if compare_variants else results


def run_resolution_sweep(
    size: str,
    img_sizes: List[int],
    device: str,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
    model_type: str = "spix",
    compare_variants: Optional[List[str]] = None,
    downsample_factor: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run comparison for a single model size across increasing image resolutions."""
    print(f"\n{'=' * 70}")
    print(f"RESOLUTION SWEEP FOR {size.upper()} MODEL")
    print(f"{'=' * 70}")

    config_dir = Path("configs/model")
    config_path = config_dir / (f"gnn_{size}.yaml" if model_type == "gnn" else f"conv_{size}.yaml" if model_type == "conv" else f"vq_{size}.yaml" if model_type == "vq" else f"{size}.yaml")
    if not config_path.exists():
        print(f"  Skipping size={size}: config not found at {config_path}")
        return []
    config = load_config(str(config_path))

    variants = [variant for variant in (compare_variants or [model_type]) if variant in {"spix", "conv", "vq", "gnn"}]
    if not variants:
        variants = [model_type]

    embed_dims = config["embed_dims"]
    depth = config["depth"]
    num_heads = config["num_heads"]

    print(f"Model config: embed_dims={embed_dims}, depth={depth}, num_heads={num_heads}")
    print(f"Testing image sizes: {img_sizes}")

    results: List[Dict[str, Any]] = []

    for img_size in img_sizes:
        print(f"\n  --- Image size: {img_size}x{img_size} ---")
        for variant in variants:
            # Build through the fully-wired builder (single source of truth for
            # variant kwargs + hard_mode/parallel handling).
            rwkv_model = build_rwkv(
                variant, size, config, img_size,
            )

            rwkv_input = create_dummy_input(img_size, channels=6, batch_size=batch_size)
            rwkv_mem = get_memory_usage(rwkv_model, rwkv_input)
            rwkv_metrics = benchmark_rwkv_components(rwkv_model, rwkv_input, warmup_runs, timed_runs, device)

            vit_model = get_vit_model(size, img_size)
            vit_input = create_dummy_input(img_size, channels=3, batch_size=batch_size)
            vit_mem = get_memory_usage(vit_model, vit_input)
            vit_metrics = benchmark_model(vit_model, vit_input, warmup_runs, timed_runs, device)

            speedup = vit_metrics["avg_time_ms"] / rwkv_metrics["avg_time_ms"]
            results.append({
                "size": size,
                "model_type": variant,
                "img_size": img_size,
                "rwkv_params": count_parameters(rwkv_model),
                "vit_params": count_parameters(vit_model),
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
    parser.add_argument(
        "--model-type",
        choices=["spix", "vq", "conv", "gnn"],
        default="spix",
        help="Backbone type",
    )
    parser.add_argument(
        "--img-size", type=int, default=512,
        help="Input image height in pixels (proportional width; default: 512)",
    )
    parser.add_argument(
        "--compare-variants",
        nargs="+",
        default=None,
        help="If set, compare these variants head-to-head per size, e.g. spix conv vq",
    )
    parser.add_argument(
        "--downsample-factors",
        nargs="+",
        type=float,
        default=[1.0],
        help="Downsample factors for spix/base model builder (default: 1.0)",
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
        sizes, config_dir, device, args.warmup, args.runs, args.batch_size,
        model_type=args.model_type, img_size=args.img_size,
        compare_variants=args.compare_variants,
        downsample_factors=args.downsample_factors,
    )

    # Part 2: Resolution sweep for selected model size+sizes
    print("\n" + "=" * 70)
    print("PART 2: RESOLUTION SWEEP")
    print("=" * 70)
    sweep_sizes = [args.sweep_size]
    if args.compare_variants:
        sweep_sizes.extend([v for v in args.compare_variants if v in {"spix", "conv", "vq"}])
    seen: set[str] = set()
    resolution_results = []
    for sweep_size in sweep_sizes:
        if sweep_size in seen:
            continue
        seen.add(sweep_size)
        resolution_results.extend(
            run_resolution_sweep(
                sweep_size, args.resolution_sweep, device,
                args.warmup, args.runs, args.batch_size,
                model_type=sweep_size,
                compare_variants=args.compare_variants,
                downsample_factor=args.downsample_factors[0] if args.downsample_factors else None,
            )
        )

    combined_results = size_results + resolution_results

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Size':<8} {'Variant':<8} {'Img':<6} {'RWKV-7':<10} {'ViT':<10} {'Speedup':<8}")
    print("-" * 70)
    for r in combined_results:
        print(f"{r['size']:<8} {r['model_type']:<8} {r['img_size']:<6} {r['rwkv_time']:<10.2f} {r['vit_time']:<10.2f} "
              f"{r['speedup']:<8.2f}x")

    # diffSLIC breakdown analysis
    print("\n" + "=" * 70)
    print("DIFFSLIC BREAKDOWN ANALYSIS")
    print("=" * 70)
    print("This shows what fraction of Vision RWKV-7 time is spent in diffSLIC.")
    print("When diffSLIC is ported to optimized C++, the backbone speedup")
    print("relative to ViT becomes more significant.\n")
    print(f"{'Variant':<10} {'Tokenizer %':<14} {'Backbone %':<14} {'Backbone/Tokenizer':<18}")
    print("-" * 70)
    for r in combined_results:
        tok_pct = 100 * r["rwkv_tokenizer"] / r["rwkv_time"]
        back_pct = 100 * r["rwkv_backbone"] / r["rwkv_time"]
        ratio = r["rwkv_backbone"] / r["rwkv_tokenizer"] if r["rwkv_tokenizer"] > 0 else 0
        print(f"{r['model_type']:<10} {tok_pct:<14.1f} {back_pct:<14.1f} {ratio:<18.2f}x")

    # Memory comparison
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON")
    print("=" * 70)
    print(f"{'Variant':<10} {'RWKV-7 (MB)':<14} {'ViT (MB)':<14} {'Ratio':<10}")
    print("-" * 70)
    for r in combined_results:
        ratio = r["vit_mem_mb"] / r["rwkv_mem_mb"] if r["rwkv_mem_mb"] > 0 else 0
        print(f"{r['model_type']:<10} {r['rwkv_mem_mb']:<14.2f} {r['vit_mem_mb']:<14.2f} {ratio:<10.2f}x")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    fastest = min(combined_results, key=lambda r: r["rwkv_time"]) if combined_results else None
    if fastest:
        print(f"Fastest Vision RWKV-7 run: {fastest['model_type']} {fastest['size']} @ {fastest['img_size']}px = {fastest['rwkv_time']:.2f}ms")

    print("\nParameter efficiency (time per M params):")
    printed_sizes: set[str] = set()
    for r in combined_results:
        if r["size"] in printed_sizes:
            continue
        printed_sizes.add(r["size"])
        subset = [x for x in combined_results if x["size"] == r["size"]]
        print(f"  {r['size']:<8}:")
        for x in subset:
            eff = x["rwkv_time"] / (x["rwkv_params"] / 1e6)
            print(f"    {x['model_type']:<10} {eff:.4f} ms/M params")

    print("\nKey findings:")
    for r in size_results:
        if r["rwkv_tokenizer"] > r["rwkv_backbone"]:
            print(f"  {r['size']} baseline: diffSLIC dominates ({100*r['rwkv_tokenizer']/r['rwkv_time']:.0f}% of time)")
            print("    -> C++ diffSLIC kernel would significantly improve RWKV-7 speed")
        else:
            print(f"  {r['size']} baseline: Backbone dominates ({100*r['rwkv_backbone']/r['rwkv_time']:.0f}% of time)")
    alt_seen: set[str] = set()
    for r in combined_results:
        if r["model_type"] in alt_seen or r["model_type"] == args.model_type:
            continue
        alt_seen.add(r["model_type"])
        subset = [x for x in combined_results if x["model_type"] == r["model_type"]]
        fastest_alt = min(subset, key=lambda x: x["rwkv_time"]) if subset else None
        if fastest_alt:
            print(f"  {r['model_type']} alt variant: fastest {fastest_alt['size']} @ {fastest_alt['img_size']}px = {fastest_alt['rwkv_time']:.2f}ms")


if __name__ == "__main__":
    import sys
    from spixrwkv7.utils import redirect_stdout_tee
    os.makedirs('results', exist_ok=True)
    
    # Check sys.argv for downsample-factors that are not 1.0 or 1
    has_sweep = False
    for i, arg in enumerate(sys.argv):
        if arg == "--downsample-factors":
            vals = []
            for val in sys.argv[i+1:]:
                if val.startswith("-"):
                    break
                vals.append(val)
            if vals and vals != ["1.0"] and vals != ["1"]:
                has_sweep = True
            break
            
    out = 'results/compare_architectures_alt_vit_downsample.txt' if has_sweep else 'results/compare_architectures_alt_vit.txt'
    with redirect_stdout_tee(out):
        main()
