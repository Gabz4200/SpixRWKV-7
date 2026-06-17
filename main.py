"""Vision-RWKV-7 demo: verify backbone with dummy image input."""

import torch
from VisualRWKV7.model import Vision_RWKV7
from VisualRWKV7.utils.data import load_image_to_tensor

# Inspired by: https://arxiv.org/abs/2109.08203
# Recommended: run 5-10 seeds, report mean ± std
seeds = [3407, 42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000]

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    torch.set_default_device(TORCH_DEVICE)

    # Initialize the new Vision-RWKV-7 with Superpixel Tokenization (diffSLIC)
    model = Vision_RWKV7(
        img_size=64,
        in_chans=3,
        embed_dims=192,
        num_heads=3,
        depth=12,
        init_values=1e-5,
        final_norm=True,
        out_indices=[3, 5, 7, 11],
        num_superpixels=196,  # Target number of superpixels (approx 14x14 grid)
        diff_slic_iters=5,  # Number of iterations for diffSLIC optimization
    )

    callable_model = model.eval().to(TORCH_DEVICE)

    if TORCH_DEVICE == "cuda":
        callable_model = torch.compile(
            callable_model
        )  # On CPU this actually makes it slower, but on CUDA it can be much faster after the initial warmup.

    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed)
            print(f"\n=== Testing seed {seed} ===")

            # Reset standard PyTorch layers (LayerNorm, Linear, etc.)
            model.apply(lambda m: getattr(m, "reset_parameters", lambda: None)())

            # Reset your custom RWKV parameters
            model._init_weights()

            # Dummy image input
            x = load_image_to_tensor(
                "test_image_from_slirack_pinterest.jpg",
                color_space="oklab",
                target_size=(64, 64),
                normalize=True,
            )

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


if __name__ == "__main__":
    main()
