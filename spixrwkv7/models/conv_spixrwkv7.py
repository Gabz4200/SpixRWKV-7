# Convolutional-Stem Vision-RWKV-7
#
# Adds a configurable strided-convolution stem before the standard
# SuperpixelTokenizer → RWKV-7 backbone.  The stem learns to redistribute
# information across channels while reducing spatial resolution, then a
# stream-split design keeps the raw semantic signal for superpixel
# generation and the deep feature map for token pooling.
#
# Data flow:
#   Raw (B, 6, H, W)
#   ├── ConvStem → deep features (B, C_feat, H/R, W/R)
#   └── F.interpolate → downsampled raw (B, 6, H/R, W/R) → diffSLIC → masks
#        └── ConvSuperpixelEmbedding(features, masks) → tokens → RWKV-7 blocks

import math
import warnings
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spixrwkv7.data.diff_slic import DiffSLIC, spixel_upsampling
from spixrwkv7.data.lnsnet import LNSNet, download_lnsnet_weights, lnsnet_assignment
from spixrwkv7.layers.graph import HEAD_SIZE, build_knn_graph
from spixrwkv7.models.spixrwkv7 import (
    Vision_RWKV7_Block,
    get_norm_layer,
    hilbert_sort_batched,
    remap_neighbors,
)

# =====================================================================
# ConvStem — Configurable strided-convolution feature extractor
# =====================================================================


class ConvStem(nn.Module):
    """Reduce spatial resolution with a sequence of strided convolutions.

    Each block: Conv2d → (optional BatchNorm / GroupNorm) → SiLU.
    Spatial reduction accumulates as the product of all ``stem_strides``.

    Anti-aliasing guard:  when ``stride > 1`` the kernel must be at least
    ``stride + 2`` so every input position is touched by at least one
    receptive-field centre, preventing the "blind-spot" aliasing that
    occurs with small kernels and large strides.

    Default config (3 blocks, strides ``(1, 2, 2)``) gives **4× reduction**
    on the middle block only, while the first block extracts features at
    full resolution and the final block refines features at ¼ resolution.

    Parameters
    ----------
    in_chans:
        Number of input channels (typically 6 for OkLAB + alpha + xy).
    stem_channels:
        Output channels for each block.  Last entry = output channel count.
    stem_kernel_sizes:
        Kernel size for each block (must be odd).  Strided layers need
        kernel >= stride + 2.
    stem_strides:
        Stride for each block.  Total reduction = product of strides.
    use_bias:
        Whether to add bias to Conv2d layers.
    use_norm:
        Whether to insert a normalisation layer after each convolution.
    norm_layer:
        ``"batchnorm2d"`` or ``"layernorm"`` (GroupNorm(1, C)).
    """

    def __init__(
        self,
        in_chans: int = 6,
        stem_channels: Tuple[int, ...] = (32, 64, 128),
        stem_kernel_sizes: Tuple[int, ...] = (3, 5, 5),
        stem_strides: Tuple[int, ...] = (1, 2, 2),
        use_bias: bool = True,
        use_norm: bool = True,
        norm_layer: str = "batchnorm2d",
    ):
        super().__init__()
        n = len(stem_channels)
        if not (len(stem_kernel_sizes) == len(stem_strides) == n):
            raise ValueError(
                f"stem_channels ({n}), stem_kernel_sizes ({len(stem_kernel_sizes)}), "
                f"and stem_strides ({len(stem_strides)}) must have the same length"
            )
        if norm_layer == "batchnorm2d":
            warnings.warn(
                "BatchNorm2d in ConvStem uses batch statistics during training "
                "but running statistics during eval. This changes the feature "
                "distribution between train/eval, which can affect superpixel "
                "segmentation boundaries. Consider 'layernorm' (GroupNorm) instead.",
                UserWarning,
                stacklevel=2,
            )

        self.total_reduction = 1
        for s in stem_strides:
            self.total_reduction *= s

        layers = []
        in_c = in_chans
        for i, (out_c, k, s) in enumerate(
            zip(stem_channels, stem_kernel_sizes, stem_strides)
        ):
            if k % 2 == 0:
                raise ValueError(f"Kernel sizes must be odd, got kernel={k}")
            if s > 1 and k < s + 2:
                raise ValueError(
                    f"Block {i}: kernel={k} with stride={s} causes aliasing "
                    f"(at least {s + 2} required for stride={s})"
                )

            pad = k // 2
            # Bias=False when following a norm layer (norm provides the affine shift)
            conv = nn.Conv2d(
                in_c,
                out_c,
                kernel_size=k,
                stride=s,
                padding=pad,
                bias=use_bias and not use_norm,
            )
            layers.append(conv)

            if use_norm:
                if norm_layer == "batchnorm2d":
                    layers.append(nn.BatchNorm2d(out_c))
                elif norm_layer == "layernorm":
                    layers.append(nn.GroupNorm(1, out_c))
                else:
                    raise ValueError(f"Unknown norm_layer: {norm_layer}")

            layers.append(nn.SiLU(inplace=True))
            in_c = out_c

        self.stem = nn.Sequential(*layers)
        self.out_chans = stem_channels[-1]

        # Residual connection when first block preserves spatial dims and channels
        self._use_residual = (stem_strides[0] == 1 and stem_channels[0] == in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deep features at reduced spatial resolution.

        Args:
            x: (B, in_chans, H, W) raw input.

        Returns:
            (B, out_chans, H // total_reduction, W // total_reduction)
            feature map.
        """
        if self._use_residual:
            return x + self.stem(x)
        return self.stem(x)


# =====================================================================
# RMSNorm2d — RMSNorm for 2D feature maps (B, C, H, W)
# =====================================================================


class RMSNorm2d(nn.Module):
    """RMSNorm applied to 2D feature maps via reshape-apply-reshape.

    Reshapes ``(B, C, H, W) -> (B*H*W, C)``, applies 1D RMSNorm, then
    reshapes back.  This provides the same per-channel normalisation
    semantics as the 1D RMSNorm used in the RWKV-7 backbone, but
    operates spatially over the conv-feature grid.

    Using RMSNorm (or LayerNorm) on the conv-stem output normalises the
    deep features before superpixel pooling, which stabilises training
    by preventing outlier channels from dominating the token embeddings.
    BatchNorm is also available via ``Batchnorm2d`` for comparison.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        variance = x_flat.pow(2).mean(-1, keepdim=True)
        x_norm = x_flat * torch.rsqrt(variance + self.eps) * self.scale
        return x_norm.reshape(B, H, W, C).permute(0, 3, 1, 2)


# =====================================================================
# ConvSuperpixelEmbedding — Pool from ConvStem features
# =====================================================================


class ConvSuperpixelEmbedding(nn.Module):
    """Convert ConvStem feature maps to superpixel tokens.

    Unlike the base ``SuperpixelEmbedding``, there is **no internal
    convolution** — the ConvStem has already performed feature extraction.
    This module:

    1. Pools features from the ConvStem output using superpixel masks.
    2. Injects centroid coordinates, normalised area, and Fourier
       positional embeddings.
    3. Projects ``(C_feat + 3) → embed_dims`` with skip-connected
       Fourier positional MLP.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dims: int,
        num_superpixels: int,
        mode: str = "soft",
        norm_layer: str = "layernorm",
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.num_superpixels = num_superpixels
        self.mode = mode

        # No extra conv — ConvStem does the heavy lifting.
        # Concatenated vector: centroids(2) + area(1) + pooled(C_feat).
        self.proj = nn.Linear(in_chans + 3, embed_dims)
        self.norm = get_norm_layer(norm_layer)(embed_dims)
        self.num_freqs = 8
        self.pos_mlp = nn.Sequential(
            nn.Linear(4 * self.num_freqs, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self._coord_cache: dict = {}

    def forward(
        self,
        x: torch.Tensor,
        sp_map: torch.Tensor,
        K: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C_in, H, W = x.shape
        assert C_in == self.in_chans, f"Expected {self.in_chans} channels, got {C_in}"

        # ---- mask → pooling weights ----
        if self.mode == "hard":
            if K is None:
                K = self.num_superpixels
            mask = F.one_hot(sp_map.long(), num_classes=K).permute(0, 3, 1, 2).float()
        else:
            mask = sp_map
            K = mask.shape[1]

        weights = mask / (mask.sum(dim=(2, 3), keepdim=True) + 1e-6)

        # ---- pool features ----
        pooled_raw = torch.einsum("bkhw,bchw->bkc", weights, x)

        # ---- centroids + area ----
        areas = mask.sum(dim=(2, 3)) / (H * W)
        areas_norm = 2.0 * areas - 1.0

        cache_key = (H, W, str(x.device), str(x.dtype))
        if cache_key not in self._coord_cache:
            grid_y = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
            grid_x = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            self._coord_cache[cache_key] = torch.stack([gx, gy], dim=-1)
        coords = self._coord_cache[cache_key]

        centroids = torch.einsum("bkhw,hwc->bkc", weights, coords)

        # ---- assemble token ----
        final_tokens = torch.cat(
            [centroids, areas_norm.unsqueeze(-1), pooled_raw], dim=-1
        )

        freqs = 2.0 ** torch.arange(
            self.num_freqs, device=centroids.device, dtype=centroids.dtype
        )
        f = centroids.unsqueeze(-1) * freqs
        fourier = torch.cat([f.sin(), f.cos()], dim=-1).flatten(2)
        pos = self.pos_mlp(fourier)

        return self.norm(self.proj(final_tokens) + pos), centroids


# =====================================================================
# ConvolutionalSuperpixelTokenizer — Stream-split tokenizer
# =====================================================================


class ConvolutionalSuperpixelTokenizer(nn.Module):
    """Two-stream tokeniser: semantic (raw→masks) + feature (conv→pooling).

    The raw 6-channel input is spatially downsampled via
    ``F.interpolate`` to match the ConvStem output resolution, preserving
    the semantic meaning required by diffSLIC / LNSNet.  The ConvStem
    feature map is pooled using the resulting superpixel masks, so the
    clustering criteria (colour + proximity) are physically meaningful
    while the token content comes from learned features.
    """

    def __init__(
        self,
        in_chans: int,
        feat_chans: int,
        embed_dims: int,
        num_superpixels: int,
        compactness: float,
        diff_slic_iters: int = 5,
        mode: str = "soft",
        use_cpp: bool = False,
        norm_layer: str = "layernorm",
        spixel_backend: str = "diff_slic",
    ):
        super().__init__()
        self.in_chans = in_chans
        self.feat_chans = feat_chans
        self.embed_dims = embed_dims
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.mode = mode
        self.spixel_backend = spixel_backend
        self.diff_slic_iters = diff_slic_iters

        # Superpixel backends (same as parent)
        if spixel_backend == "diff_slic":
            self.diff_slic = DiffSLIC(
                n_spixels=num_superpixels,
                n_iter=diff_slic_iters,
                tau=0.01,
                candidate_radius=1,
                stable=True,
                use_cpp=use_cpp,
            )
        else:
            self.diff_slic = None

        if spixel_backend == "lnsnet":
            import os

            self.lnsnet_model = LNSNet(n_spix=num_superpixels)
            cache_dir = os.path.expanduser("~/.cache/spixrwkv7")
            check_path = os.path.join(cache_dir, "lnsnet_BSDS_checkpoint.pth")
            download_lnsnet_weights(check_path)
            if not os.path.exists(check_path):
                raise FileNotFoundError(
                    f"LNSNet checkpoint not found at {check_path}. "
                    "Provide the weights or choose a different spixel_backend."
                )
            state_dict = torch.load(check_path, map_location="cpu")
            self.lnsnet_model.load_state_dict(state_dict)
            print(f"Loaded LNSNet BSDS checkpoint from {check_path}.")
        else:
            self.lnsnet_model = None

        # Simplified embedding (no internal conv)
        self.patch_embed = ConvSuperpixelEmbedding(
            feat_chans,
            embed_dims,
            num_superpixels,
            mode=mode,
            norm_layer=norm_layer,
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        x_feat: torch.Tensor,
        num_superpixels: Optional[int] = None,
        spixel_size: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Tokenize and return ordered tokens, graph, and spatial metadata.

        Args:
            x_raw: (B, in_chans, H, W) — raw semantic input (e.g. OkLAB+alpha+xy).
            x_feat: (B, feat_chans, Hf, Wf) — ConvStem feature map.
            num_superpixels: optional override for target superpixel count.
            spixel_size: optional override to derive count from resolution.
            mask: optional (B, N) token mask.

        Returns:
            Dict with keys:
              tokens, neighbors, neighbor_dists, inv_order, batch_idx,
              global_soft_mask, global_labels, h_s, w_s, mask
        """
        B, _, Hf, Wf = x_feat.shape

        # ---- Downsample raw to match feature resolution ----
        x_raw_down = F.interpolate(
            x_raw, size=(Hf, Wf), mode="bilinear", align_corners=False
        )

        # ---- Determine superpixel count ----
        if num_superpixels is None:
            if spixel_size is not None:
                n_sp = round((Hf * Wf) / (spixel_size**2))
            else:
                n_sp = self.num_superpixels
        else:
            n_sp = num_superpixels

        height_s = max(1, int(math.sqrt(n_sp * Hf / Wf)))
        width_s = max(1, int(math.sqrt(n_sp * Wf / Hf)))
        h_s, w_s = height_s, width_s
        K = h_s * w_s

        # ---- Tokenization (masks from downsampled raw) ----
        global_soft_mask: Optional[torch.Tensor] = None
        global_labels: Optional[torch.Tensor] = None

        if self.spixel_backend == "diff_slic":
            x_for_slic = torch.cat(
                [x_raw_down[:, :-2], x_raw_down[:, -2:] * self.compactness], dim=1
            )
            assert self.diff_slic is not None
            clst_feats, p2s_assign, _ = self.diff_slic(x_for_slic, n_spixels=n_sp)
            h_s, w_s = clst_feats.shape[-2:]
            K = h_s * w_s
            radius = self.diff_slic.candidate_radius

            if self.mode == "hard":
                neighbor_range = 2 * radius + 1
                hard_assign = (
                    F.one_hot(p2s_assign.argmax(1), neighbor_range**2)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .float()
                )
                label_grid = (
                    torch.arange(K, dtype=torch.float, device=x_raw.device)
                    .reshape(1, 1, h_s, w_s)
                    .expand(B, -1, -1, -1)
                )
                global_labels = (
                    spixel_upsampling(label_grid, hard_assign, candidate_radius=radius)
                    .squeeze(1)
                    .long()
                )
                tokens, centroids = self.patch_embed(x_feat, global_labels, K=K)
            else:
                spixel_ids = (
                    torch.arange(K, device=x_raw.device)
                    .reshape(1, K, 1, 1)
                    .expand(B, -1, h_s, w_s)
                    .float()
                )
                global_soft_mask = spixel_upsampling(
                    spixel_ids, p2s_assign, candidate_radius=radius
                )
                tokens, centroids = self.patch_embed(x_feat, global_soft_mask)

        elif self.spixel_backend == "lnsnet":
            x_lnsnet = torch.cat([x_raw_down[:, :3], x_raw_down[:, 4:6]], dim=1)
            x_lnsnet = (x_lnsnet - x_lnsnet.mean(dim=(2, 3), keepdim=True)) / (
                x_lnsnet.std(dim=(2, 3), keepdim=True) + 1e-6
            )
            assert self.lnsnet_model is not None
            cx, cy, f, probs = self.lnsnet_model(x_lnsnet)

            S = Hf * Wf / n_sp
            sp_h = max(1, int(math.floor(math.sqrt(S) / (Wf / float(Hf)))))
            sp_w = max(1, int(math.floor(S / math.floor(sp_h))))
            h_s = int(math.ceil(Hf / sp_h))
            w_s = int(math.ceil(Wf / sp_w))
            K = h_s * w_s

            p2s_assign = lnsnet_assignment(f, x_lnsnet, cx, cy)

            if self.mode == "hard":
                global_labels = p2s_assign.argmax(dim=1)
                tokens, centroids = self.patch_embed(x_feat, global_labels, K=K)
            else:
                global_soft_mask = p2s_assign
                tokens, centroids = self.patch_embed(x_feat, global_soft_mask)

        elif self.spixel_backend == "grid":
            grid_y = torch.arange(Hf, device=x_raw.device) * h_s // Hf
            grid_x = torch.arange(Wf, device=x_raw.device) * w_s // Wf
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            global_labels = (gy * w_s + gx).unsqueeze(0).expand(B, -1, -1).long()

            if self.mode == "hard":
                tokens, centroids = self.patch_embed(x_feat, global_labels, K=K)
            else:
                global_soft_mask = (
                    F.one_hot(global_labels, num_classes=K).permute(0, 3, 1, 2).float()
                )
                tokens, centroids = self.patch_embed(x_feat, global_soft_mask)

        elif self.spixel_backend in ("slic", "slico"):
            import numpy as np
            import skimage.segmentation as seg

            labels_list = []
            comp = self.compactness * 20.0
            slic_zero = self.spixel_backend == "slico"
            for i in range(B):
                img_np = x_raw_down[i, :3].permute(1, 2, 0).detach().cpu().numpy()
                lbls = seg.slic(
                    img_np,
                    n_segments=K,
                    compactness=comp,
                    max_num_iter=self.diff_slic_iters,
                    slic_zero=slic_zero,
                    start_label=0,
                    enforce_connectivity=True,
                )
                lbls = np.clip(lbls, 0, K - 1)
                labels_list.append(
                    torch.from_numpy(lbls).to(device=x_raw.device, dtype=torch.long)
                )
            global_labels = torch.stack(labels_list, dim=0)

            if self.mode == "hard":
                tokens, centroids = self.patch_embed(x_feat, global_labels, K=K)
            else:
                global_soft_mask = (
                    F.one_hot(global_labels, num_classes=K).permute(0, 3, 1, 2).float()
                )
                tokens, centroids = self.patch_embed(x_feat, global_soft_mask)

        else:
            raise ValueError(f"Unknown spixel_backend: {self.spixel_backend}")

        # ---- KNN graph + Hilbert reorder ----
        neighbors, neighbor_dists = build_knn_graph(centroids.detach(), k=4)
        coords_int = ((centroids + 1.0) * 4096).long().clamp(0, 8191)
        order = hilbert_sort_batched(coords_int)
        inv_order = torch.argsort(order, dim=1)
        batch_idx = torch.arange(B, device=x_raw.device)
        tokens = tokens[batch_idx[:, None], order]
        neighbors = remap_neighbors(neighbors, order, inv_order)
        neighbor_dists = neighbor_dists[batch_idx[:, None], order]
        sorted_mask = mask[batch_idx[:, None], order] if mask is not None else None

        return {
            "tokens": tokens,
            "neighbors": neighbors,
            "neighbor_dists": neighbor_dists,
            "inv_order": inv_order,
            "batch_idx": batch_idx,
            "global_soft_mask": global_soft_mask,
            "global_labels": global_labels,
            "h_s": h_s,
            "w_s": w_s,
            "mask": sorted_mask,
        }


# =====================================================================
# ConvolutionalVision_RWKV7 (Backbone)
# =====================================================================


class ConvolutionalVision_RWKV7(nn.Module):
    """Vision-RWKV-7 with a convolutional stem for learned downsampling.

    Composes::

        ConvStem (feature extraction + 4× downsample)
          → ConvolutionalSuperpixelTokenizer (masks from raw, pool from features)
            → Hilbert sort + KNN graph
              → RWKV-7 blocks
                → output projection

    The convolutional stem allows the model to learn a semantically-aware
    redistribution of the 6-channel input before the superpixel and
    recurrent stages.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 6,
        embed_dims: int = 192,
        num_heads: Optional[int] = None,
        depth: int = 12,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 0.0,
        final_norm: bool = True,
        out_indices: Sequence[int] = (-1,),
        with_cls_token: bool = False,
        output_cls_token: bool = False,
        register_tokens: int = 0,
        scatter_output: bool = False,
        num_superpixels: int = 256,
        spixel_size: Optional[int] = None,
        diff_slic_iters: int = 5,
        compactness: float = 0.5,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        spixel_backend: str = "diff_slic",
        use_attnres: bool = False,
        attnres_mode: str = "block",
        attnres_gate_type: str = "bias",
        attnres_num_blocks: int = 8,
        attnres_recency_bias_init: float = 10.0,
        use_cpp: bool = False,
        # ConvStem configuration
        conv_stem_channels: Tuple[int, ...] = (32, 64, 128),
        conv_stem_kernel_sizes: Tuple[int, ...] = (3, 5, 5),
        conv_stem_strides: Tuple[int, ...] = (1, 2, 2),
        conv_stem_norm: str = "batchnorm2d",
        conv_post_norm: str = "layernorm",
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.num_layers = depth
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.scatter_output = scatter_output
        self.compactness = compactness
        self.in_chans = in_chans
        self.num_superpixels = num_superpixels
        self.spixel_size = spixel_size
        self.spixel_backend = spixel_backend

        # Attention Residuals configuration
        self.use_attnres = use_attnres
        self.attnres_mode = attnres_mode
        self.attnres_gate_type = attnres_gate_type
        self.attnres_num_blocks = attnres_num_blocks
        self.attnres_recency_bias_init = attnres_recency_bias_init
        self.last_attnres_history = None
        self.last_project_fn = None

        # ---- ConvStem ----
        self.conv_stem = ConvStem(
            in_chans=in_chans,
            stem_channels=conv_stem_channels,
            stem_kernel_sizes=conv_stem_kernel_sizes,
            stem_strides=conv_stem_strides,
            norm_layer=conv_stem_norm,
        )
        feat_chans = self.conv_stem.out_chans

        # ---- Post-stem normalisation ----
        if conv_post_norm == "none" or conv_post_norm is None:
            self.conv_post_norm = nn.Identity()
        elif conv_post_norm == "batchnorm2d":
            self.conv_post_norm = nn.BatchNorm2d(feat_chans)
        elif conv_post_norm == "layernorm":
            self.conv_post_norm = nn.GroupNorm(1, feat_chans)
        elif conv_post_norm == "rmsnorm":
            self.conv_post_norm = RMSNorm2d(feat_chans)
        else:
            raise ValueError(f"Unknown conv_post_norm: {conv_post_norm}")

        # ---- Dynamic num_heads ----
        if num_heads is None:
            assert embed_dims % HEAD_SIZE == 0, (
                f"embed_dims={embed_dims} must be divisible by HEAD_SIZE={HEAD_SIZE}"
            )
            num_heads = embed_dims // HEAD_SIZE

        # ---- Tokenizer (stream split) ----
        self.tokenizer = ConvolutionalSuperpixelTokenizer(
            in_chans=in_chans,
            feat_chans=feat_chans,
            embed_dims=embed_dims,
            num_superpixels=num_superpixels,
            compactness=compactness,
            diff_slic_iters=diff_slic_iters,
            mode="soft",
            use_cpp=use_cpp,
            norm_layer=norm_layer,
            spixel_backend=spixel_backend,
        )
        self.patch_embed = self.tokenizer.patch_embed

        from spixrwkv7.models.common import init_backbone_tokens
        init_backbone_tokens(self, with_cls_token, register_tokens, embed_dims)

        self.blocks = self._make_blocks(
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            with_cls_token=with_cls_token,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_cpp=use_cpp,
            **kwargs,
        )

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = get_norm_layer(norm_layer)(embed_dims)

        from spixrwkv7.models.common import normalize_out_indices
        self.out_indices = normalize_out_indices(out_indices, depth)

        self._init_weights()

    # ------------------------------------------------------------------
    # _make_blocks — identical to Vision_RWKV7
    # ------------------------------------------------------------------

    def _make_blocks(
        self,
        embed_dims: int,
        num_heads: int,
        depth: int,
        drop_path_rate: float,
        init_values: Optional[float],
        with_cls_token: bool,
        norm_layer: str,
        act_layer: str,
        use_cpp: bool,
        **kwargs,
    ) -> nn.ModuleList:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        return nn.ModuleList(
            [
                Vision_RWKV7_Block(
                    embed_dims,
                    num_heads,
                    depth,
                    i,
                    drop_prob=dpr[i],
                    init_values=init_values,
                    with_cls_token=with_cls_token,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_attnres=self.use_attnres,
                    attnres_mode=self.attnres_mode,
                    attnres_gate_type=self.attnres_gate_type,
                    attnres_num_blocks=self.attnres_num_blocks,
                    attnres_recency_bias_init=self.attnres_recency_bias_init,
                    num_prepend_tokens=self.register_tokens,
                )
                for i in range(depth)
            ]
        )

    def _init_weights(self):
        from spixrwkv7.models.common import zero_init_backbone_tokens
        zero_init_backbone_tokens(self)

    # ------------------------------------------------------------------
    # _project_output
    # ------------------------------------------------------------------

    def _project_output(
        self,
        patch_tokens: torch.Tensor,
        inv_order: torch.Tensor,
        batch_idx: torch.Tensor,
        global_soft_mask: Optional[torch.Tensor],
        global_labels: Optional[torch.Tensor],
        Hf: int,
        Wf: int,
        h_s: int,
        w_s: int,
    ) -> torch.Tensor:
        """Reorder tokens and optionally scatter to feature-grid resolution."""
        patch_tokens = patch_tokens[batch_idx[:, None], inv_order]

        if self.scatter_output:
            if self.tokenizer.mode == "soft":
                assert global_soft_mask is not None
                feat = torch.einsum("bkd,bkhw->bhwd", patch_tokens, global_soft_mask)
                feat = feat.permute(0, 3, 1, 2)
            else:
                assert global_labels is not None
                feat = patch_tokens.gather(
                    1,
                    global_labels.view(-1, Hf * Wf, 1).expand(
                        -1, -1, self.embed_dims
                    ),
                )
                feat = feat.view(-1, Hf, Wf, self.embed_dims).permute(0, 3, 1, 2)
        else:
            feat = patch_tokens.view(-1, h_s, w_s, self.embed_dims).permute(0, 3, 1, 2)
        return feat

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        num_superpixels: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass returning multi-scale features.

        Args:
            x: (B, C, H, W) input tensor (6-channel OkLAB + alpha + xy).
            num_superpixels: optional override for target superpixel count.
            mask: optional (B, N) token mask.

        Returns:
            Tuple of (B, embed_dims, h_s, w_s) or (B, embed_dims, Hf, Wf)
            feature maps, one per ``out_indices`` entry.
        """
        B, C, H, W = x.shape
        assert C == self.in_chans, (
            f"Model in_chans={self.in_chans}, got {C}. "
            "Please prepare 6-channel (OkLAB+alpha+xy) input externally."
        )

        # ---- 1. ConvStem (feature extraction + downsampling) ----
        x_feat = self.conv_stem(x)  # (B, feat_chans, Hf, Wf)
        x_feat = self.conv_post_norm(x_feat)
        _, _, Hf, Wf = x_feat.shape

        # Downsample mask to match reduced resolution
        mask_down = None
        if mask is not None:
            mask_down = (
                F.interpolate(
                    mask.unsqueeze(1).float(), size=(Hf, Wf), mode="nearest"
                )
                .squeeze(1)
                .long()
            )

        # ---- 2. Tokenization (stream-split) ----
        out = self.tokenizer(
            x,
            x_feat,
            num_superpixels=num_superpixels,
            spixel_size=self.spixel_size,
            mask=mask_down,
        )
        tokens = out["tokens"]
        neighbors = out["neighbors"]
        neighbor_dists = out["neighbor_dists"]
        inv_order = out["inv_order"]
        batch_idx = out["batch_idx"]
        global_soft_mask = out["global_soft_mask"]
        global_labels = out["global_labels"]
        h_s, w_s = out["h_s"], out["w_s"]
        sorted_mask = out["mask"]

        # ---- 3. CLS + Register tokens ----
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat((tokens, cls_tokens), dim=1)

        if self.register_tokens > 0:
            assert self.reg_token is not None
            reg_tokens = self.reg_token.expand(B, -1, -1)
            tokens = torch.cat((reg_tokens, tokens), dim=1)

        # ---- 4. RWKV-7 Blocks ----
        outs: list = []
        vf_fwd, vf_bwd = None, None
        attnres_history = [tokens] if self.use_attnres else None

        def get_patches(toks):
            if self.with_cls_token:
                toks = toks[:, :-1]
            if self.register_tokens > 0:
                toks = toks[:, self.register_tokens :]
            return toks

        for i, block in enumerate(self.blocks):
            if self.use_attnres:
                tokens, vff, vfb = block(
                    tokens,
                    neighbors,
                    neighbor_dists,
                    vf_fwd,
                    vf_bwd,
                    mask=sorted_mask,
                    attnres_history=attnres_history,
                )
            else:
                tokens, vff, vfb = block(
                    tokens,
                    neighbors,
                    neighbor_dists,
                    vf_fwd,
                    vf_bwd,
                    mask=sorted_mask,
                )
            if i == 0:
                vf_fwd, vf_bwd = vff, vfb
            if i == len(self.blocks) - 1 and self.final_norm:
                tokens = self.ln1(tokens)

            if i in self.out_indices:
                if self.with_cls_token:
                    cls_out = tokens[:, -1]
                    tokens_for_out = tokens[:, :-1]
                else:
                    cls_out = None
                    tokens_for_out = tokens

                if self.register_tokens > 0:
                    patch_tokens = tokens_for_out[:, self.register_tokens :]
                else:
                    patch_tokens = tokens_for_out

                feat = self._project_output(
                    patch_tokens,
                    inv_order,
                    batch_idx,
                    global_soft_mask,
                    global_labels,
                    Hf,
                    Wf,
                    h_s,
                    w_s,
                )

                if self.output_cls_token and cls_out is not None:
                    outs.append((feat, cls_out))
                else:
                    outs.append(feat)

        if self.use_attnres:
            self.last_attnres_history = (
                [t.detach() for t in attnres_history] if attnres_history else None
            )
            self.last_attnres_history_patches = (
                [get_patches(t) for t in self.last_attnres_history]
                if self.last_attnres_history is not None
                else None
            )
            _inv_order = inv_order.detach()
            _batch_idx = batch_idx.detach()
            _global_soft_mask = global_soft_mask.detach() if global_soft_mask is not None else None
            _global_labels = global_labels.detach() if global_labels is not None else None
            self.last_project_fn = lambda pt: self._project_output(
                pt,
                _inv_order,
                _batch_idx,
                _global_soft_mask,
                _global_labels,
                Hf,
                Wf,
                h_s,
                w_s,
            )

        return tuple(outs)


# =====================================================================
# Model Builders
# =====================================================================


def create_conv_vision_rwkv7(
    img_size: int = 224,
    embed_dims: int = 192,
    num_heads: Optional[int] = None,
    depth: int = 12,
    drop_path_rate: float = 0.0,
    init_values: Optional[float] = 0.0,
    final_norm: bool = True,
    out_indices: Sequence[int] = (-1,),
    with_cls_token: bool = False,
    output_cls_token: bool = False,
    scatter_output: bool = False,
    num_superpixels: int = 256,
    spixel_size: Optional[int] = None,
    diff_slic_iters: int = 5,
    compactness: float = 0.5,
    register_tokens: int = 0,
    norm_layer: str = "layernorm",
    act_layer: str = "relu2",
    spixel_backend: str = "diff_slic",
    use_attnres: bool = False,
    attnres_mode: str = "block",
    attnres_gate_type: str = "bias",
    attnres_num_blocks: int = 8,
    attnres_recency_bias_init: float = 10.0,
    use_cpp: bool = False,
    use_jit: bool = False,
    # ConvStem config
    conv_stem_channels: Tuple[int, ...] = (32, 64, 128),
    conv_stem_kernel_sizes: Tuple[int, ...] = (3, 5, 5),
    conv_stem_strides: Tuple[int, ...] = (1, 2, 2),
    conv_stem_norm: str = "batchnorm2d",
    conv_post_norm: str = "layernorm",
) -> torch.nn.Module:
    """Create a ConvolutionalVision_RWKV7 model with 6-channel input.

    This is the standard entry point for the convolutional-stem variant.
    The stem defaults to a 4× spatial reduction (strides ``(1, 2, 2)``),
    expanding channels ``6 → 32 → 64 → 128`` before the superpixel and
    RWKV-7 stages.

    Args:
        img_size: Unused (kept for API compatibility with original).
        embed_dims: Token embedding dimension.
        num_heads: Number of attention heads (auto if None).
        depth: Number of RWKV-7 blocks.
        drop_path_rate: Stochastic depth drop rate.
        init_values: LayerScale initial value (0 = identity).
        final_norm: Apply LayerNorm after last block.
        out_indices: Indices of blocks to return features from.
        with_cls_token: Append a learnable CLS token.
        output_cls_token: Return CLS token alongside features.
        scatter_output: Scatter tokens back to pixel grid.
        num_superpixels: Target number of superpixels.
        spixel_size: Derive superpixel count from this size.
        diff_slic_iters: diffSLIC iterations.
        compactness: Superpixel compactness (multiplied on xy channels).
        register_tokens: DINOv2-style register tokens.
        norm_layer: Normalisation layer name.
        act_layer: Activation function.
        spixel_backend: Superpixel algorithm backend.
        use_attnres: Enable attention residuals.
        attnres_mode: Attention residual mode.
        attnres_gate_type: Gate type for attention residuals.
        attnres_num_blocks: Number of blocks for attention residuals.
        attnres_recency_bias_init: Initial recency bias.
        use_jit: Apply TorchScript JIT.
        conv_stem_channels: Output channels per ConvStem block.
        conv_stem_kernel_sizes: Kernel sizes per ConvStem block.
        conv_stem_strides: Strides per ConvStem block.
        conv_stem_norm: Normalisation for ConvStem blocks.

    Returns:
        ``ConvolutionalVision_RWKV7`` module (optionally JIT-compiled).
    """
    from spixrwkv7.jit import maybe_compile

    _model: torch.nn.Module = ConvolutionalVision_RWKV7(
        img_size=img_size,
        in_chans=6,
        embed_dims=embed_dims,
        num_heads=num_heads,
        depth=depth,
        drop_path_rate=drop_path_rate,
        init_values=init_values,
        final_norm=final_norm,
        out_indices=out_indices,
        with_cls_token=with_cls_token,
        output_cls_token=output_cls_token,
        scatter_output=scatter_output,
        num_superpixels=num_superpixels,
        spixel_size=spixel_size,
        diff_slic_iters=diff_slic_iters,
        compactness=compactness,
        register_tokens=register_tokens,
        norm_layer=norm_layer,
        act_layer=act_layer,
        spixel_backend=spixel_backend,
        use_attnres=use_attnres,
        attnres_mode=attnres_mode,
        attnres_gate_type=attnres_gate_type,
        attnres_num_blocks=attnres_num_blocks,
        attnres_recency_bias_init=attnres_recency_bias_init,
        use_cpp=use_cpp,
        conv_stem_channels=conv_stem_channels,
        conv_stem_kernel_sizes=conv_stem_kernel_sizes,
        conv_stem_strides=conv_stem_strides,
        conv_stem_norm=conv_stem_norm,
        conv_post_norm=conv_post_norm,
    )
    return maybe_compile(_model, use_jit=use_jit)
