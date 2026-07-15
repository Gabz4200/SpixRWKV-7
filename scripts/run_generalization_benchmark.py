"""Generalization benchmark: train/val evaluation on tiny-imagenet.

Trains ONE model variant per invocation on tiny-imagenet (200 classes, 100K train,
10K val, 64x64). Reports train/val accuracy per epoch and overfitting gap.

Usage:
    uv run python scripts/run_generalization_benchmark.py --model-type gnn --size tiny
    uv run python scripts/run_generalization_benchmark.py --model-type spix --size tiny --max-epochs 5
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from spixrwkv7 import ClassificationHead
from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7
from spixrwkv7.models.conv_spixrwkv7 import create_conv_vision_rwkv7
from spixrwkv7.models.gnn_spixrwkv7 import create_gnn_vision
from spixrwkv7.models.hybrid_spixrwkv7 import create_hybrid_vision
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7


# =====================================================================
# ViT Baseline (from run_full_benchmark.py)
# =====================================================================

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


VIT_CONFIGS = {
    "tiny":  {"embed_dim": 192, "depth": 12, "num_heads": 3, "register_tokens": 4},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "register_tokens": 4},
}


# =====================================================================
# Dataset: tiny-imagenet via HuggingFace
# =====================================================================

def load_tiny_imagenet():
    """Load tiny-imagenet dataset. Returns (train_dataset, val_dataset, num_classes)."""
    from datasets import load_dataset
    from PIL import Image
    import numpy as np

    ds = load_dataset("zh-plus/tiny-imagenet")

    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, hf_split, img_size):
            self.examples = hf_split
            self.img_size = img_size

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            item = self.examples[idx]
            img = item["image"]
            label = item["label"]

            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))

            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_np = torch.tensor(np.array(img), dtype=torch.uint8)

            img_float = img_np.permute(2, 0, 1).float() / 255.0

            srgb = img_float.unsqueeze(0)
            linear_rgb = srgb.clamp(0).pow(2.2)
            from spixrwkv7.data.colors import from_linear_rgb_to_oklab
            oklab = from_linear_rgb_to_oklab(linear_rgb).squeeze(0)

            alpha = torch.ones(1, self.img_size, self.img_size)
            H, W = self.img_size, self.img_size
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H),
                torch.linspace(-1, 1, W),
                indexing="ij",
            )
            xy = torch.stack([xx, yy], dim=0)

            x = torch.cat([oklab, alpha, xy], dim=0)
            return x, label

    return (
        TensorDataset(ds["train"], 64),
        TensorDataset(ds["valid"], 64),
        200,
    )


# =====================================================================
# Model builders
# =====================================================================

def build_model(model_type: str, size: str, num_classes: int, config: dict) -> tuple[nn.Module, nn.Module]:
    """Build backbone + head. Returns (backbone, head)."""
    img_size = 64

    if model_type == "vit":
        cfg = VIT_CONFIGS[size]
        vit = SimpleViT(
            img_size=img_size, patch_size=16, in_chans=6,
            num_classes=num_classes, **cfg,
        )
        return vit, None

    embed_dims = config["embed_dims"]
    depth = config["depth"]
    num_heads = config["num_heads"]

    builder_kwargs = {
        "img_size": img_size,
        "embed_dims": embed_dims,
        "num_heads": num_heads,
        "depth": depth,
        "init_values": config.get("init_values", 1e-5),
        "final_norm": config.get("final_norm", True),
        "out_indices": config.get("out_indices", [-1]),
        "norm_layer": config.get("norm_layer", "rmsnorm"),
        "act_layer": config.get("act_layer", "swiglu"),
        "use_cpp": True,
    }

    if model_type == "spix":
        backbone = create_optimized_vision_rwkv7(
            **builder_kwargs,
            num_superpixels=config.get("num_superpixels", 36),
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            register_tokens=config.get("register_tokens", 0),
            use_attnres=config.get("use_attnres", False),
        )
    elif model_type == "conv":
        backbone = create_conv_vision_rwkv7(
            **builder_kwargs,
            num_superpixels=config.get("num_superpixels", 36),
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            register_tokens=config.get("register_tokens", 0),
            use_attnres=config.get("use_attnres", False),
            conv_stem_channels=tuple(config.get("conv_stem_channels", [32, 64, 128])),
            conv_stem_kernel_sizes=tuple(config.get("conv_stem_kernel_sizes", [3, 5, 5])),
            conv_stem_strides=tuple(config.get("conv_stem_strides", [1, 2, 2])),
            conv_stem_norm=config.get("conv_stem_norm", "layernorm"),
            conv_post_norm=config.get("conv_post_norm", "layernorm"),
        )
    elif model_type == "vq":
        backbone = create_vq_rwkv7(
            **builder_kwargs,
            scatter_output=config.get("scatter_output", False),
            codebook_size=config.get("codebook_size", 128),
            downsample_factor=config.get("downsample_factor", 16),
            latent_dim=config.get("latent_dim", None),
            num_res_blocks=config.get("num_res_blocks", 2),
            use_ema=False,
            beta=0.25,
            register_tokens=config.get("register_tokens", 0),
            use_attnres=config.get("use_attnres", False),
        )
    elif model_type == "gnn":
        backbone = create_gnn_vision(
            **builder_kwargs,
            num_superpixels=config.get("num_superpixels", 36),
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            register_tokens=config.get("register_tokens", 0),
            downsample_factor=float(config.get("downsample_factor", 16)),
            gnn_conv=config.get("gnn_conv", "gatv2"),
            gnn_heads=config.get("gnn_heads", 4),
            gnn_aggr=config.get("gnn_aggr", "mean"),
            use_attnres=config.get("use_attnres", False),
        )
    elif model_type == "hybrid":
        backbone = create_hybrid_vision(
            **builder_kwargs,
            num_superpixels=config.get("num_superpixels", 36),
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            register_tokens=config.get("register_tokens", 4),
            downsample_factor=float(config.get("downsample_factor", 16)),
            num_rwkv_layers=config.get("num_rwkv_layers", 1),
            num_gnn_layers=config.get("num_gnn_layers", 3),
            knn_k=config.get("knn_k", 4),
            dedup_neighbors=config.get("dedup_neighbors", True),
            dedup_centroids=config.get("dedup_centroids", True),
            gnn_conv=config.get("gnn_conv", "gatv2"),
            gnn_heads=config.get("gnn_heads", 4),
            gnn_aggr=config.get("gnn_aggr", "mean"),
            use_attnres=config.get("use_attnres", False),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    head = ClassificationHead(embed_dims, num_classes)
    return backbone, head


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =====================================================================
# Training loop
# =====================================================================

def train_epoch(backbone, head, dataloader, optimizer, device, scaler, use_amp):
    backbone.train()
    if head is not None:
        head.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outs = backbone(x)
            feat = outs[0] if isinstance(outs, tuple) else outs

            if head is not None:
                logits = head(feat)
            else:
                logits = feat

            loss = F.cross_entropy(logits, y)

            q_loss = getattr(backbone, "_last_q_loss", None)
            if q_loss is not None:
                loss = loss + q_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + (list(head.parameters()) if head is not None else []),
            max_norm=10.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(backbone, head, dataloader, device, use_amp):
    backbone.eval()
    if head is not None:
        head.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outs = backbone(x)
            feat = outs[0] if isinstance(outs, tuple) else outs

            if head is not None:
                logits = head(feat)
            else:
                logits = feat

            loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * correct / total


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generalization benchmark on tiny-imagenet")
    parser.add_argument("--model-type", choices=["spix", "vq", "conv", "gnn", "hybrid", "vit"], required=True)
    parser.add_argument("--size", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"  GENERALIZATION BENCHMARK: {args.model_type.upper()} ({args.size})")
    print("=" * 78)
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.max_epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 78)

    print("\nLoading tiny-imagenet...")
    train_ds, val_ds, num_classes = load_tiny_imagenet()
    print(f"  Train: {len(train_ds)} images, Val: {len(val_ds)} images, Classes: {num_classes}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    config_path = _ROOT / "configs" / "model" / f"{'conv_' if args.model_type == 'conv' else 'gnn_' if args.model_type == 'gnn' else 'vq_' if args.model_type == 'vq' else 'hybrid_' if args.model_type == 'hybrid' else ''}{args.size}.yaml"
    if args.model_type == "vit":
        config = {}
    elif config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)["model"]
    else:
        print(f"  Config not found: {config_path}, using defaults")
        config = {}

    print(f"\nBuilding {args.model_type.upper()} model...")
    backbone, head = build_model(args.model_type, args.size, num_classes, config)
    backbone = backbone.to(device)
    if head is not None:
        head = head.to(device)

    backbone_params = count_parameters(backbone)
    head_params = count_parameters(head) if head is not None else 0
    total_params = backbone_params + head_params
    print(f"  Backbone params: {backbone_params / 1e6:.2f}M")
    print(f"  Head params:     {head_params / 1e6:.2f}M")
    print(f"  Total params:    {total_params / 1e6:.2f}M")

    params = list(backbone.parameters())
    if head is not None:
        params += list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\nTraining for {args.max_epochs} epochs...")
    print(f"  {'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Gap':>7} {'Time':>8}")
    print(f"  {'-' * 65}")

    best_val_acc = 0.0
    best_epoch = 0
    results = []

    t0 = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_t0 = time.perf_counter()

        train_loss, train_acc = train_epoch(backbone, head, train_loader, optimizer, device, scaler, use_amp)
        val_loss, val_acc = evaluate(backbone, head, val_loader, device, use_amp)
        gap = train_acc - val_acc

        epoch_time = time.perf_counter() - epoch_t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "gap": gap,
            "time": epoch_time,
        })

        print(f"  {epoch:5d} {train_loss:11.4f} {train_acc:9.2f}% {val_loss:10.4f} {val_acc:8.2f}% {gap:6.2f}% {epoch_time:7.1f}s")

    total_time = time.perf_counter() - t0

    print(f"\n{'=' * 78}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Model:          {args.model_type.upper()} ({args.size})")
    print(f"  Total params:   {total_params / 1e6:.2f}M")
    print(f"  Best val acc:   {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Final train:    {results[-1]['train_acc']:.2f}%")
    print(f"  Final val:      {results[-1]['val_acc']:.2f}%")
    print(f"  Final gap:      {results[-1]['gap']:.2f}%")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Avg epoch time: {total_time / args.max_epochs:.1f}s")
    print(f"{'=' * 78}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/generalization_{args.model_type}_{args.size}.txt"
    with open(out_path, "w") as f:
        f.write(f"Generalization Benchmark: {args.model_type.upper()} ({args.size})\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Config:\n")
        f.write(f"  Device:     {device}\n")
        f.write(f"  Epochs:     {args.max_epochs}\n")
        f.write(f"  LR:         {args.lr}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Params:     {total_params / 1e6:.2f}M\n\n")
        f.write(f"  {'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Gap':>7}\n")
        f.write(f"  {'-' * 55}\n")
        for r in results:
            f.write(f"  {r['epoch']:5d} {r['train_loss']:11.4f} {r['train_acc']:9.2f}% {r['val_loss']:10.4f} {r['val_acc']:8.2f}% {r['gap']:6.2f}%\n")
        f.write(f"\nBest val acc: {best_val_acc:.2f}% (epoch {best_epoch})\n")
        f.write(f"Total time: {total_time:.1f}s\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    from spixrwkv7.utils import redirect_stdout_tee
    os.makedirs("results", exist_ok=True)
    main()
