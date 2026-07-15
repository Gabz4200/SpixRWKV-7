"""Benchmark: C++ vs PyTorch kernels across all architectures.

Compares inference speed of each model variant with and without C++ kernels,
plus ViT baseline. Uses real images from data/caltech101_classification/.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from spixrwkv7 import ClassificationHead
from spixrwkv7.data.image_utils import load_random_caltech101_image
from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7
from spixrwkv7.models.conv_spixrwkv7 import create_conv_vision_rwkv7
from spixrwkv7.models.gnn_spixrwkv7 import create_gnn_vision
from spixrwkv7.models.hybrid_spixrwkv7 import create_hybrid_vision
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7


class SimpleViT(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, num_classes,
                 embed_dim, depth, num_heads, register_tokens=0):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = register_tokens
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
        if self.register_tokens > 0:
            x = x[:, self.register_tokens:]
        return self.head(x[:, 0])


def benchmark(model, x, warmup=3, runs=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def build_model(model_type, size, img_size, use_cpp):
    cfgs = {
        "tiny": {"embed_dims": 192, "depth": 2, "num_heads": 3, "num_superpixels": 36},
        "small": {"embed_dims": 384, "depth": 6, "num_heads": 6, "num_superpixels": 144},
    }
    cfg = cfgs[size]

    common = dict(
        img_size=img_size, embed_dims=cfg["embed_dims"], num_heads=cfg["num_heads"],
        depth=cfg["depth"], init_values=1e-5, final_norm=True, out_indices=[-1],
        num_superpixels=cfg["num_superpixels"], scatter_output=True,
        diff_slic_iters=5, compactness=0.5, norm_layer="rmsnorm", act_layer="swiglu",
    )

    if model_type == "spix":
        return create_optimized_vision_rwkv7(**common, use_cpp=use_cpp)
    elif model_type == "conv":
        return create_conv_vision_rwkv7(**common, use_cpp=use_cpp,
            conv_stem_channels=(32, 64, 128), conv_stem_kernel_sizes=(3, 5, 5),
            conv_stem_strides=(1, 2, 2), conv_stem_norm="batchnorm2d", conv_post_norm="layernorm")
    elif model_type == "gnn":
        return create_gnn_vision(**common, use_cpp=use_cpp, gnn_conv="gatv2", gnn_heads=4)
    elif model_type == "hybrid":
        return create_hybrid_vision(**common, use_cpp=use_cpp,
            num_rwkv_layers=1, num_gnn_layers=3, knn_k=4)
    elif model_type == "vq":
        return create_vq_rwkv7(**common, use_cpp=use_cpp, codebook_size=1024, downsample_factor=16)
    elif model_type == "vit":
        return SimpleViT(img_size=img_size, patch_size=16, in_chans=6, num_classes=3,
                         embed_dim=cfg["embed_dims"], depth=cfg["depth"],
                         num_heads=cfg["num_heads"], register_tokens=4)
    raise ValueError(f"Unknown: {model_type}")


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    img_size = 256
    sizes = ["tiny", "small"]
    model_types = ["spix", "conv", "gnn", "hybrid", "vq", "vit"]

    x, _, _ = load_random_caltech101_image(img_size=img_size, seed=42)

    print("=" * 90)
    print("  C++ KERNEL BENCHMARK: With vs Without C++ Kernels")
    print("=" * 90)
    print(f"  Device: CPU | Image: {img_size}px | Input: {x.shape}")
    print("=" * 90)

    rows = []

    for size in sizes:
        print(f"\n--- {size.upper()} ---")
        for mt in model_types:
            for use_cpp in ([True, False] if mt != "vit" else [False]):
                label = f"{mt}" + ("+cpp" if use_cpp else "+py")
                try:
                    model = build_model(mt, size, img_size, use_cpp=use_cpp)
                    params = count_params(model)
                    avg, std = benchmark(model, x)
                    rows.append({
                        "size": size, "model": mt, "cpp": use_cpp,
                        "params": params, "avg_ms": avg, "std_ms": std,
                        "label": label,
                    })
                    print(f"  {label:12s} {params/1e6:7.2f}M  {avg:8.2f}ms ± {std:.2f}")
                    del model
                except Exception as e:
                    print(f"  {label:12s} ERROR: {e}")
                    rows.append({
                        "size": size, "model": mt, "cpp": use_cpp,
                        "params": 0, "avg_ms": float("nan"), "std_ms": float("nan"),
                        "label": label,
                    })

    # Print comparison table
    print("\n" + "=" * 90)
    print("  RESULTS TABLE")
    print("=" * 90)

    for size in sizes:
        print(f"\n  {size.upper()}:")
        print(f"  {'Model':<14} {'Params':>8} {'PyTorch (ms)':>14} {'C++ (ms)':>12} {'Speedup':>10}")
        print(f"  {'-'*60}")

        size_rows = [r for r in rows if r["size"] == size]
        # Group by model type
        by_model = {}
        for r in size_rows:
            by_model.setdefault(r["model"], []).append(r)

        for mt in model_types:
            if mt not in by_model:
                continue
            mrs = by_model[mt]
            py_row = next((r for r in mrs if not r["cpp"]), None)
            cpp_row = next((r for r in mrs if r["cpp"]), None)

            params = (cpp_row or py_row)["params"]
            py_ms = py_row["avg_ms"] if py_row else float("nan")
            cpp_ms = cpp_row["avg_ms"] if cpp_row else float("nan")

            if py_row and cpp_row and py_ms > 0 and cpp_ms > 0:
                speedup = f"{py_ms/cpp_ms:.2f}x"
            else:
                speedup = "—"

            py_str = f"{py_ms:.2f}" if py_row and py_ms == py_ms else "—"
            cpp_str = f"{cpp_ms:.2f}" if cpp_row and cpp_ms == cpp_ms else "—"

            print(f"  {mt:<14} {params/1e6:7.2f}M {py_str:>14} {cpp_str:>12} {speedup:>10}")

    return rows


if __name__ == "__main__":
    rows = main()
