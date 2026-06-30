"""SpixRWKV-7 demo: verify backbone with dummy image input."""

import argparse
import os
import random
import sys
from pathlib import Path

# Add project root for direct execution
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from spixrwkv7.data.transforms import preprocess_image_for_rwkv7  # noqa: E402
from spixrwkv7.kernels.optimized_vision import (  # noqa: E402
    create_optimized_vision_rwkv7 as _create_model,
)
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7  # noqa: E402

# Inspired by: https://arxiv.org/abs/2109.08203
# Recommended: run 5-10 seeds, report mean ± std
seeds = [3407, 42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000]

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="SpixRWKV-7 demo")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--embed-dims", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-superpixels", type=int, default=196)
    parser.add_argument("--diff-slic-iters", type=int, default=5)
    parser.add_argument("--norm-layer", type=str, default="rmsnorm")
    parser.add_argument("--act-layer", type=str, default="swiglu")
    parser.add_argument("--spixel-backend", type=str, default="diff_slic", choices=["diff_slic", "grid", "slic", "slico", "lnsnet"])
    parser.add_argument("--use-attnres", action="store_true", help="Enable Attention Residuals")
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
    parser.add_argument("--output", type=str, default="results/demo.txt")
    args = parser.parse_args()

    from spixrwkv7.utils import redirect_stdout_tee
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with redirect_stdout_tee(args.output):
        # Initialize the model (Superpixel or VQ)
        if args.model_type == "vq":
            model = create_vq_rwkv7(
                img_size=args.img_size,
                embed_dims=args.embed_dims,
                num_heads=args.num_heads,
                depth=args.depth,
                init_values=1e-5,
                final_norm=True,
                out_indices=[args.depth - 1],
                scatter_output=True,
                codebook_size=args.codebook_size,
                downsample_factor=args.downsample_factor,
                latent_dim=args.latent_dim,
                num_res_blocks=args.num_res_blocks,
                norm_layer=args.norm_layer,
                act_layer=args.act_layer,
                use_attnres=args.use_attnres,
            )
        else:
            model = _create_model(
                img_size=args.img_size,
                embed_dims=args.embed_dims,
                num_heads=args.num_heads,
                depth=args.depth,
                init_values=1e-5,
                final_norm=True,
                out_indices=[args.depth - 1],
                num_superpixels=args.num_superpixels,
                scatter_output=True,
                diff_slic_iters=args.diff_slic_iters,
                norm_layer=args.norm_layer,
                act_layer=args.act_layer,
                spixel_backend=args.spixel_backend,
                use_attnres=args.use_attnres,
            )

        # Model created on CPU, then moved to device (standard PyTorch pattern)
        callable_model = model.eval().to(TORCH_DEVICE)

        if TORCH_DEVICE == "cuda":
            callable_model = torch.compile(
                callable_model
            )  # On CPU this actually makes it slower, but on CUDA it can be much faster after the initial warmup.

        with torch.no_grad():
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                print(f"\n=== Testing seed {seed} ===")

                # Reset standard PyTorch layers (LayerNorm, Linear, etc.)
                model.apply(lambda m: getattr(m, "reset_parameters", lambda: None)())

                # Reset your custom RWKV parameters
                model._init_weights()

                # Dummy image input
                x_raw = preprocess_image_for_rwkv7(
                    "test_image_from_slirack_pinterest.jpg",
                    target_size=(args.img_size, args.img_size),
                    include_alpha=True,
                )
                # x_raw is already (1, 6, H, W) from preprocess_image_for_rwkv7
                x = x_raw.to(TORCH_DEVICE)

                # Forward pass
                outs = callable_model(x)

                print(f"Input:  {tuple(x.shape)}")
                print(f"Output levels:  {len(outs)}")
                for i, o in enumerate(outs):
                    print(f"  level {i}: {tuple(o.shape)}")

                total = sum(p.numel() for p in model.parameters())
                print(f"\nTotal params: {total / 1e6:.2f}M")

                # Verify no NaN/Inf
                all_finite = all(o.isfinite().all() for o in outs)
                print(f"All outputs finite: {all_finite}")

                # Verify deterministic: same input -> same output
                outs2 = callable_model(x)
                deterministic = all(
                    (o1 - o2).abs().max().item() < 1e-5 for o1, o2 in zip(outs, outs2)
                )
                print(f"Deterministic: {deterministic}")

                # Cleanup Memory
                del x, outs, outs2
                if TORCH_DEVICE == "cuda":
                    torch.cuda.empty_cache()
        del model, callable_model
        if TORCH_DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
