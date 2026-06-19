#!/usr/bin/env python3
"""Visualize Superpixel Tokenization (diffSLIC) for SpixRWKV-7."""

import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from spixrwkv7 import create_vision_rwkv7
from spixrwkv7.data.transforms import preprocess_image_for_rwkv7
from spixrwkv7.data.diff_slic import spixel_upsampling
from skimage.segmentation import mark_boundaries


def main():
    parser = argparse.ArgumentParser(description="Visualize SpixRWKV-7 Superpixel Tokenization")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--n_count", type=int, default=196, help="Target number of superpixels")
    parser.add_argument("--spixel_size", type=int, default=None, help="Target superpixel size")
    parser.add_argument("--compactness", type=float, default=0.5, help="Compactness factor")
    parser.add_argument("--iters", type=int, default=5, help="Number of diffSLIC iterations")
    parser.add_argument("--size", type=int, default=512, help="Image size for visualization")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Preprocess image for the model
    x = preprocess_image_for_rwkv7(
        args.image_path,
        target_size=(args.size, args.size),
        include_alpha=True,
    ).to(device)

    # 2. Initialize Model (Backbone)
    model = create_vision_rwkv7(
        img_size=args.size,
        embed_dims=64,
        depth=1,
        num_superpixels=args.n_count,
        spixel_size=args.spixel_size,
        diff_slic_iters=args.iters,
        compactness=args.compactness,
    ).to(device).eval()

    # 3. Run Forward pass to get superpixel labels
    with torch.no_grad():
        B, _, H, W = x.shape

        n_sp = args.n_count
        if args.spixel_size is not None:
            n_sp = int(round((H * W) / (args.spixel_size**2)))

        x_for_slic = torch.cat([x[:, :-2], x[:, -2:] * model.compactness], dim=1)

        clst_feats, p2s_assign, _ = model.diff_slic(x_for_slic, n_spixels=n_sp)
        h_s, w_s = clst_feats.shape[-2:]
        K = h_s * w_s
        radius = model.diff_slic.candidate_radius

        hard_assign_idx = p2s_assign.argmax(1)
        neighbor_range = 2 * radius + 1
        hard_assign_mask = torch.nn.functional.one_hot(
            hard_assign_idx, neighbor_range**2
        ).permute(0, 3, 1, 2).float()

        label_grid = (
            torch.arange(K, dtype=torch.float, device=x.device)
            .reshape(1, 1, h_s, w_s)
            .expand(B, -1, -1, -1)
        )

        global_labels = (
            spixel_upsampling(label_grid, hard_assign_mask, candidate_radius=radius)
            .squeeze(1)
            .long()
        )

    # 4. Prepare image for display
    display_img = Image.open(args.image_path).convert("RGB").resize((args.size, args.size))
    display_np = np.array(display_img)

    labels_np = global_labels[0].cpu().numpy()

    # 5. Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    out = mark_boundaries(display_np, labels_np, color=(1, 1, 0))

    ax.imshow(out)
    ax.set_title(
        f"SpixRWKV-7 Superpixels (n_count={args.n_count}, compactness={args.compactness})"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
