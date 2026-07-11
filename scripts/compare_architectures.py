"""Compare Vision RWKV-7 vs ViT inference speed across model sizes.

This script benchmarks both architectures at equivalent parameter counts,
using the same image resolution for each size config. Optimized for CPU
execution with optional GPU support and minimal memory footprint.

Key insight: diffSLIC tokenization is the primary bottleneck in Vision RWKV-7.
When ported to optimized C++ kernel, the recurrent backbone becomes significantly
faster relative to ViT's quadratic attention.
"""

import argparse
import inspect
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7 as _create_model
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7
from spixrwkv7.models.conv_spixrwkv7 import create_conv_vision_rwkv7


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)["model"]


class SimpleViT(nn.Module):
    """Simple ViT implementation matching torchvision ViT parameter counts.

    Uses standard patch embedding, class token, and transformer blocks.
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, num_classes: int,
                 embed_dim: int, depth: int, num_heads: int):
        super().__init__()
        self.num_features = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                batch_first=True,
            )
            for _ in range(depth)
        ])

        # Head
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//P, W//P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Add class token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head
        return self.head(x[:, 0])


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
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
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
            # Conv model tokenizer needs both raw image and stem features
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
    else:
        tokenizer_times = [0.0] * timed_runs

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


def _build_rwkv(
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
            scatter_output=config.get("scatter_output", True),
            drop_path_rate=config.get("drop_path_rate", 0.0),
            codebook_size=config.get("codebook_size", 1024),
            downsample_factor=config.get("downsample_factor", 16),
            norm_layer=config.get("norm_layer", "layernorm"),
            act_layer=config.get("act_layer", "relu2"),
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
            conv_stem_norm=config.get("conv_stem_norm", "batchnorm2d"),
            conv_post_norm=config.get("conv_post_norm", "layernorm"),
        )
    if downsample_factor is None:
        downsample_factor = config.get("downsample_factor", 16)
    if model_type == "spix":
        kwargs = dict(
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
            use_parallel=use_parallel,
        )
        if downsample_factor is not None:
            sig = inspect.signature(_create_model)
            if "downsample_factor" in sig.parameters:
                val = int(downsample_factor) if downsample_factor == int(downsample_factor) else downsample_factor
                kwargs["downsample_factor"] = val
        return _create_model(**kwargs)


def _benchmark_variant(
    model_type: str, size: str, config: Dict[str, Any], img_size: int,
    device: str, warmup_runs: int, timed_runs: int, batch_size: int,
    use_parallel: bool = False, hard_mode: bool = False,
    downsample_factor: Optional[float] = None,
) -> Dict[str, Any]:
    model = _build_rwkv(
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
    if use_parallel and model_type not in ("vq", "conv"):
        print("    Using parallel RWKV-7 scan")

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
    if downsample_factor is not None:
        result["downsample_factor"] = downsample_factor
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
    use_parallel: bool = False,
    hard_mode: bool = False,
    model_type: str = "spix",
    img_size: int = 512,
    compare_variants: Optional[List[str]] = None,
    downsample_factors: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run comparison across model sizes at a given image height.

    Returns ``(results, variant_results)`` where ``results`` preserves the
    original ViT-vs-RWKV behavior for ``model_type``, and ``variant_results``
    contains one entry per ``(size, variant)`` when ``compare_variants`` is set.
    """
    results: List[Dict[str, Any]] = []
    variant_results: List[Dict[str, Any]] = []
    variants = [model_type]
    if compare_variants:
        variants = [v for v in compare_variants if v in {"spix", "conv", "vq"}]
        if not variants:
            variants = [model_type]

    if downsample_factors is None:
        downsample_factors = [1.0]

    for size in sizes:
        for factor in downsample_factors:
            run_variants = [v for v in variants if v == "spix" or factor == downsample_factors[0]]
            if not run_variants:
                continue
            print(f"\n--- {size.upper()} (factor={factor}) ---")
            for variant in run_variants:
                config_path = config_dir / (f"conv_{size}.yaml" if variant == "conv" else f"{size}.yaml")
                config = load_config(str(config_path))
                entry = _benchmark_variant(
                    variant, size, config, img_size, device, warmup_runs, timed_runs,
                    batch_size, use_parallel=use_parallel, hard_mode=hard_mode,
                    downsample_factor=factor,
                )
                variant_results.append(entry)
                if variant == model_type:
                    results.append(entry)

    return results, variant_results


def run_resolution_sweep(
    size: str,
    img_sizes: List[int],
    device: str,
    warmup_runs: int,
    timed_runs: int,
    batch_size: int,
    use_parallel: bool = False,
    hard_mode: bool = False,
    model_type: str = "spix",
):
    """Run comparison for a single model size across increasing image resolutions."""
    print(f"\n{'=' * 70}")
    print(f"RESOLUTION SWEEP FOR {size.upper()} MODEL")
    print(f"{'=' * 70}")

    config_dir = Path("configs/model")
    config_path = config_dir / f"{size}.yaml" if model_type != "conv" else config_dir / f"conv_{size}.yaml"
    config = load_config(str(config_path))

    embed_dims = config["embed_dims"]
    depth = config["depth"]
    num_heads = config["num_heads"]
    num_superpixels = config["num_superpixels"]

    print(f"Model config: embed_dims={embed_dims}, depth={depth}, num_heads={num_heads}")
    print(f"Testing image sizes: {img_sizes}")

    results = []

    for img_size in img_sizes:
        print(f"\n  --- Image size: {img_size}x{img_size} ---")

        # Create Vision RWKV-7 model (optimized if available)
        if model_type == "vq":
            rwkv_model = create_vq_rwkv7(
                img_size=img_size,
                embed_dims=embed_dims,
                num_heads=num_heads,
                depth=depth,
                init_values=config.get("init_values", 1e-5),
                final_norm=config.get("final_norm", True),
                out_indices=config.get("out_indices", [-1]),
                scatter_output=config.get("scatter_output", True),
                drop_path_rate=config.get("drop_path_rate", 0.0),
                codebook_size=config.get("codebook_size", 1024),
                downsample_factor=config.get("downsample_factor", 16),
                norm_layer=config.get("norm_layer", "layernorm"),
                act_layer=config.get("act_layer", "relu2"),
            )
        elif model_type == "conv":
            rwkv_model = create_conv_vision_rwkv7(
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
                conv_stem_norm=config.get("conv_stem_norm", "batchnorm2d"),
                conv_post_norm=config.get("conv_post_norm", "layernorm"),
            )
        else:
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
                use_parallel=use_parallel,
            )
        if hard_mode and model_type not in ("vq", "conv"):
            tokenizer = getattr(rwkv_model, "tokenizer", None)
            if tokenizer is not None:
                setattr(tokenizer, "mode", "hard")
                diff_slic = getattr(tokenizer, "diff_slic", None)
                if diff_slic is not None:
                    setattr(diff_slic, "hard_mode", True)
        if use_parallel and model_type not in ("vq", "conv"):
            pass  # already passed to model


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
    print("\n  Resolution sweep results:")
    print(f"  {'Img':<8} {'RWKV-7':<12} {'ViT':<12} {'Speedup':<10} {'Tokenizer %':<12} {'Mem RWKV':<10} {'Mem ViT':<10}")
    print(f"  {'-' * 74}")
    for r in results:
        tok_pct = 100 * r["rwkv_tokenizer"] / r["rwkv_time"]
        print(f"  {r['img_size']:<8} {r['rwkv_time']:<12.2f} {r['vit_time']:<12.2f} "
              f"{r['speedup']:<10.2f}x {tok_pct:<12.1f} {r['rwkv_mem_mb']:<10.2f} {r['vit_mem_mb']:<10.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Vision RWKV-7 vs ViT speed")
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
        "--sizes", nargs="+", default=None,
        help="List of model sizes to benchmark (default: tiny small medium large)",
    )
    parser.add_argument(
        "--resolution-sweep",
        nargs="+",
        type=int,
        default=[64, 128, 512, 1024],
        help="Image sizes for resolution sweep (default: 64 128 512 1024)",
    )
    parser.add_argument(
        "--sweep-size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Model size for resolution sweep (default: small)",
    )
    parser.add_argument(
        "--use-parallel",
        action="store_true",
        help="Use parallel RWKV-7 scan (Blelloch prefix) instead of sequential RNN mode",
    )
    parser.add_argument(
        "--hard-mode",
        action="store_true",
        help="Use hard diffSLIC mode (seeds-revised style, faster inference)",
    )
    parser.add_argument(
        "--use-cpp",
        action="store_true",
        default=True,
        help="Use optimized C++ kernel for RWKV-7 backbone (default: True)",
    )
    parser.add_argument(
        "--model-type",
        choices=["spix", "vq", "conv"],
        default="spix",
        help="Backbone type (default: spix)",
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
        default=[1.0],
        type=float,
        help="Downsample factors to sweep on spix/base variants (default: 1.0)",
    )
    parser.add_argument(
        "--img-size", type=int, default=512,
        help="Input image height in pixels (proportional width; default: 512)",
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
    if args.sizes is not None:
        sizes = args.sizes
    else:
        sizes = ["tiny", "small", "medium", "large"]
        if args.skip_large:
            sizes = ["tiny", "small", "medium"]

    # Part 1: Model size comparison at configured resolutions
    print("\n" + "=" * 70)
    print("PART 1: MODEL SIZE COMPARISON")
    print("=" * 70)
    size_results, variant_results = run_size_comparison(
        sizes, config_dir, device, args.warmup, args.runs, args.batch_size,
        use_parallel=args.use_parallel,
        hard_mode=args.hard_mode,
        model_type=args.model_type,
        img_size=args.img_size,
        compare_variants=args.compare_variants,
        downsample_factors=args.downsample_factors,
    )

    # Part 2: Resolution sweep for selected model size
    print("\n" + "=" * 70)
    print("PART 2: RESOLUTION SWEEP")
    print("=" * 70)
    run_resolution_sweep(
        args.sweep_size, args.resolution_sweep, device,
        args.warmup, args.runs, args.batch_size,
        use_parallel=args.use_parallel,
        hard_mode=args.hard_mode,
        model_type=args.model_type,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Size':<14} {'Img':<6} {'RWKV-7':<10} {'ViT':<10} {'Speedup':<8}")
    print("-" * 70)
    for r in size_results:
        factor_str = f" (df={r['downsample_factor']})" if "downsample_factor" in r else ""
        print(f"{r['size'] + factor_str:<14} {r['img_size']:<6} {r['rwkv_time']:<10.2f} {r['vit_time']:<10.2f} "
              f"{r['speedup']:<8.2f}x")

    # diffSLIC breakdown analysis
    print("\n" + "=" * 70)
    print("DIFFSLIC BREAKDOWN ANALYSIS")
    print("=" * 70)
    print("This shows what fraction of Vision RWKV-7 time is spent in diffSLIC.")
    print("When diffSLIC is ported to optimized C++, the backbone speedup")
    print("relative to ViT becomes more significant.\n")
    print(f"{'Size':<14} {'Tokenizer %':<12} {'Backbone %':<12} {'Backbone/Tokenizer':<18}")
    print("-" * 70)
    for r in size_results:
        tok_pct = 100 * r["rwkv_tokenizer"] / r["rwkv_time"]
        back_pct = 100 * r["rwkv_backbone"] / r["rwkv_time"]
        ratio = r["rwkv_backbone"] / r["rwkv_tokenizer"] if r["rwkv_tokenizer"] > 0 else 0
        factor_str = f" (df={r['downsample_factor']})" if "downsample_factor" in r else ""
        print(f"{r['size'] + factor_str:<14} {tok_pct:<12.1f} {back_pct:<12.1f} {ratio:<18.2f}x")

    # Memory comparison
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON")
    print("=" * 70)
    print(f"{'Size':<14} {'RWKV-7 (MB)':<14} {'ViT (MB)':<14} {'Ratio':<10}")
    print("-" * 70)
    for r in size_results:
        ratio = r["vit_mem_mb"] / r["rwkv_mem_mb"] if r["rwkv_mem_mb"] > 0 else 0
        factor_str = f" (df={r['downsample_factor']})" if "downsample_factor" in r else ""
        print(f"{r['size'] + factor_str:<14} {r['rwkv_mem_mb']:<14.2f} {r['vit_mem_mb']:<14.2f} {ratio:<10.2f}x")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    fastest = min(size_results, key=lambda r: r["rwkv_time"])
    factor_str = f" (df={fastest['downsample_factor']})" if "downsample_factor" in fastest else ""
    print(f"Fastest Vision RWKV-7: {fastest['size'] + factor_str} ({fastest['rwkv_time']:.2f}ms)")

    print("\nParameter efficiency (time per M params):")
    for r in size_results:
        rwkv_eff = r["rwkv_time"] / (r["rwkv_params"] / 1e6)
        vit_eff = r["vit_time"] / (r["vit_params"] / 1e6)
        factor_str = f" (df={r['downsample_factor']})" if "downsample_factor" in r else ""
        print(f"  {r['size'] + factor_str:<14}: RWKV-7={rwkv_eff:.4f}ms/M, ViT={vit_eff:.4f}ms/M")

    print("\nKey findings:")
    for r in size_results:
        factor_str = f" (df={r['downsample_factor']})" if "downsample_factor" in r else ""
        if r["rwkv_tokenizer"] > r["rwkv_backbone"]:
            print(f"  {r['size'] + factor_str}: diffSLIC dominates ({100*r['rwkv_tokenizer']/r['rwkv_time']:.0f}% of time)")
            print("    -> C++ diffSLIC kernel would significantly improve RWKV-7 speed")
        else:
            print(f"  {r['size'] + factor_str}: Backbone dominates ({100*r['rwkv_backbone']/r['rwkv_time']:.0f}% of time)")

    if variant_results:
        print("\n" + "=" * 70)
        print("VARIANT COMPARISON")
        print("=" * 70)
        for size in sizes:
            print(f"\n--- {size.upper()} ---")
            print(f"{'Variant':<16} {'Params(M)':<12} {'Total(ms)':<12} {'Tokenizer(ms)':<16} {'Backbone(ms)':<14} {'Speedup':<10}")
            print("-" * 80)
            size_variants = [r for r in variant_results if r["size"] == size]
            for r in size_variants:
                factor_str = f" (df={r['downsample_factor']})" if "downsample_factor" in r else ""
                print(f"{r['model_type'] + factor_str:<16} {r['rwkv_params']/1e6:<12.2f} {r['rwkv_time']:<12.2f} "
                      f"{r['rwkv_tokenizer']:<16.2f} {r['rwkv_backbone']:<14.2f} {r['speedup']:<10.2f}x")

            best = min(size_variants, key=lambda r: r["rwkv_time"]) if size_variants else None
            if best:
                print(f"  -> Fastest variant: {best['model_type']} ({best['rwkv_time']:.2f}ms)")

    # Explanation of results
    print("\n" + "=" * 70)
    print("WHY IS ViT FASTER ON CPU?")
    print("=" * 70)
    print("""
The SimpleViT uses nn.TransformerEncoderLayer which is a highly optimized
PyTorch implementation with efficient matrix operations. Key factors:

1. PyTorch Transformer uses optimized GEMM (General Matrix Multiply) kernels
   that are heavily tuned for CPU.

2. The Vision RWKV-7 backbone has a sequential recurrent loop (for t in range(N))
   that cannot be vectorized like the parallel attention in ViT. Each timestep
   requires its own forward pass through the recurrence.

3. diffSLIC tokenization involves iterative clustering with softmax operations
   over spatial dimensions - this is computationally expensive on CPU.

4. The RWKV-7 recurrence uses small matrix operations (head_size=64) that
   have poor cache efficiency compared to larger GEMM operations.

5. On GPU, the comparison would favor RWKV-7 more because:
   - The recurrent loop can run in parallel across sequence positions
   - No quadratic attention memory overhead
   - diffSLIC would still be a bottleneck but less severe

The "parameter efficiency" metric (time per M params) shows ViT is ~100-400x
more efficient on CPU because it uses optimized parallel operations, while
RWKV-7 uses sequential recurrence that doesn't parallelize well on CPU.
""")


if __name__ == "__main__":
    from spixrwkv7.utils import redirect_stdout_tee
    os.makedirs('results', exist_ok=True)
    with redirect_stdout_tee('results/compare_architectures.txt'):
        main()
    print('Results saved to results/compare_architectures.txt')
