"""VQ-RWKV-7: RWKV-7 vision backbone with VQ-VAE tokenization.

Replaces diffSLIC superpixel tokenization with discrete VQ-VAE tokenization.
The VQ-VAE encoder produces a grid of codebook indices; the corresponding
codebook embeddings serve as token inputs to the RWKV-7 backbone blocks.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spixrwkv7.layers.graph import HEAD_SIZE, build_knn_graph
from spixrwkv7.models.spixrwkv7 import (
    Vision_RWKV7_Block,
    get_norm_layer,
    hilbert_sort_batched,
    remap_neighbors,
)

# =====================================================================
# VectorQuantizer — Discrete codebook with straight-through gradients
# =====================================================================


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with straight-through gradient estimator.

    Supports standard codebook (gradient-based) and EMA-based codebook updates.
    Uses the efficient ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e formula.

    Args:
        n_e: Number of codebook entries.
        e_dim: Codebook embedding dimension.
        beta: Weight for commitment loss (standard mode only).
        use_ema: If True, use EMA-based codebook updates (no codebook loss).
        decay: EMA decay factor.
        epsilon: Laplace smoothing factor for EMA.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        use_ema: bool = False,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_ema = use_ema

        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(n_e))
            self.register_buffer("ema_w", torch.zeros(n_e, e_dim))
            self.decay = decay
            self.epsilon = epsilon
            embed = torch.randn(n_e, e_dim) * (1.0 / math.sqrt(e_dim))
            self.register_buffer("embedding", embed)
        else:
            embed = torch.randn(n_e, e_dim) * (1.0 / math.sqrt(e_dim))
            self.embedding = nn.Parameter(embed)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latent features to nearest codebook entries.

        Args:
            z: (B, C, H, W) latent features.
        Returns:
            z_q: (B, C, H, W) quantized features (with straight-through grad).
            indices: (B, H, W) codebook indices.
            loss: Scalar quantization loss.
        """
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        emb = self.embedding

        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        if self.use_ema:
            with torch.no_grad():
                dist = (
                    z_flat.pow(2).sum(1, keepdim=True)
                    + emb.pow(2).sum(1, keepdim=True).t()
                    - 2 * z_flat @ emb.t()
                )
        else:
            dist = (
                z_flat.pow(2).sum(1, keepdim=True)
                + emb.pow(2).sum(1, keepdim=True).t()
                - 2 * z_flat @ emb.t()
            )

        indices_flat = dist.argmin(-1)
        indices = indices_flat.view(B, H, W)
        z_q_flat = F.embedding(indices_flat, emb)
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2)

        if self.use_ema:
            with torch.no_grad():
                enc_onehot = F.one_hot(indices_flat, self.n_e).float()
                cluster_size = enc_onehot.sum(0)
                encoded_sum = enc_onehot.t() @ z_flat
                self.ema_cluster_size.data.mul_(self.decay).add_(
                    cluster_size * (1 - self.decay)
                )
                self.ema_w.data.mul_(self.decay).add_(
                    encoded_sum * (1 - self.decay)
                )
                smoothed_size = self.ema_cluster_size + self.epsilon
                embed_normalized = self.ema_w / smoothed_size.unsqueeze(1)
                self.embedding.data.copy_(embed_normalized)
            q_loss = F.mse_loss(z_q.detach(), z)
        else:
            codebook_loss = F.mse_loss(z_q, z.detach())
            commitment_loss = F.mse_loss(z_q.detach(), z)
            q_loss = codebook_loss + self.beta * commitment_loss

        z_q = z + (z_q - z).detach()
        return z_q, indices, q_loss


# =====================================================================
# ResidualBlock — Simple conv res-block for encoder/decoder
# =====================================================================


class _ResidualBlock(nn.Module):
    """Pre-activation residual block: GN → ReLU → Conv → GN → ReLU → Conv."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(1, dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _DecoderResidualBlock(nn.Module):
    """Residual block for decoder with pre-concatenated skip connections.

    The caller concatenates the skip features before calling this block,
    so ``in_chans`` already includes the skip channels. A 1×1 projection
    maps the concatenated features to ``out_chans``, and the residual
    connection is a learned projection from the decoder-only channels.
    """

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.pre_norm = nn.GroupNorm(1, in_chans)
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, padding=1)
        self.post_norm = nn.GroupNorm(1, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, 1)
        self.res_proj = nn.Conv2d(in_chans, out_chans, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.pre_norm(x))
        h = self.conv1(h)
        h = F.relu(self.post_norm(h))
        h = self.conv2(h)
        return self.res_proj(x) + h


# =====================================================================
# ConvolutionalVQVAE — Encoder → Quantizer → Decoder
# =====================================================================


class ConvolutionalVQVAE(nn.Module):
    """Convolutional VQ-VAE with U-Net skip connections.

    Encoder progressively downsamples and stores skip features at each level.
    Decoder upsamples and concatenates corresponding encoder features before
    each residual block, preserving spatial detail.

    Args:
        in_chans: Input image channels.
        latent_dim: Dimension of the latent code space.
        codebook_size: Number of discrete codes.
        downsample_factor: Total spatial downsampling factor (power of 2).
        num_res_blocks: Number of residual blocks at the latent bottleneck.
        use_ema: Use EMA-based codebook updates.
        beta: Commitment loss weight (standard mode only).
        decay: EMA decay factor.
    """

    def __init__(
        self,
        in_chans: int,
        latent_dim: int,
        codebook_size: int,
        downsample_factor: int = 16,
        num_res_blocks: int = 2,
        use_ema: bool = False,
        beta: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor
        self.in_chans = in_chans

        n_down = int(math.log2(downsample_factor))
        assert 2**n_down == downsample_factor, (
            f"downsample_factor must be a power of 2, got {downsample_factor}"
        )

        hidden_dim = max(64, latent_dim // 4)

        # --- Encoder ---
        self.enc_initial = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(),
        )

        self.enc_downsamples = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        channels = [hidden_dim]
        cur_dim = hidden_dim
        for i in range(n_down):
            out_dim = min(cur_dim * 2, latent_dim)
            self.enc_downsamples.append(
                nn.Conv2d(cur_dim, out_dim, 4, stride=2, padding=1)
            )
            self.enc_norms.append(nn.GroupNorm(1, out_dim))
            channels.append(out_dim)
            cur_dim = out_dim

        self.pre_quant_conv = nn.Conv2d(cur_dim, latent_dim, 1)
        self.enc_res = nn.Sequential(
            *[_ResidualBlock(latent_dim) for _ in range(num_res_blocks)]
        )

        # --- Quantizer ---
        self.quantizer = VectorQuantizer(
            n_e=codebook_size,
            e_dim=latent_dim,
            beta=beta,
            use_ema=use_ema,
            decay=decay,
        )

        # --- Decoder (with U-Net skip connections) ---
        self.post_quant_conv = nn.Conv2d(latent_dim, cur_dim, 1)
        self.dec_upsamples = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        self.dec_res_blocks = nn.ModuleList()

        for i in range(n_down):
            skip_chans = channels[-(i + 2)]  # encoder feature at this level
            out_dim = max(cur_dim // 2, hidden_dim)
            # Upsample first, then concat with skip → res_block
            self.dec_upsamples.append(
                nn.ConvTranspose2d(cur_dim, out_dim, 4, stride=2, padding=1)
            )
            self.dec_norms.append(nn.GroupNorm(1, out_dim))
            self.dec_res_blocks.append(
                _DecoderResidualBlock(out_dim + skip_chans, out_dim)
            )
            cur_dim = out_dim

        self.dec_final = nn.Sequential(
            nn.GroupNorm(1, cur_dim),
            nn.ReLU(),
            nn.Conv2d(cur_dim, in_chans, 3, padding=1),
            nn.Sigmoid(),
        )

        # Backward-compatible aliases for tests that access .encoder / .decoder
        self.encoder = nn.Sequential(self.enc_initial, *[
            item for pair in zip(self.enc_downsamples, self.enc_norms) for item in pair
        ])
        self.decoder = nn.Sequential(*[
            item for triple in zip(self.dec_upsamples, self.dec_norms, self.dec_res_blocks) for item in triple
        ], self.dec_final)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to quantized latents and codebook indices."""
        skips = []
        h = self.enc_initial(x)
        for down, norm in zip(self.enc_downsamples, self.enc_norms):
            skips.append(h)
            h = F.relu(norm(down(h)))
        h = self.pre_quant_conv(h)
        h = self.enc_res(h)
        z_q, indices, q_loss = self.quantizer(h)
        return z_q, indices, q_loss, skips

    def decode(
        self, z_q: torch.Tensor, skips: Optional[list] = None
    ) -> torch.Tensor:
        """Decode quantized latents to reconstruction with skip connections.

        Standard U-Net pattern: upsample → concat skip → residual block.
        """
        h = self.post_quant_conv(z_q)
        for i, (up, norm, res_block) in enumerate(
            zip(self.dec_upsamples, self.dec_norms, self.dec_res_blocks)
        ):
            h = F.relu(norm(up(h)))
            skip = skips[-(i + 1)] if skips is not None else None
            if skip is not None:
                h = torch.cat([h, skip], dim=1)
            h = res_block(h)
        return self.dec_final(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VQ-VAE forward: encode → quantize → decode."""
        z_q, indices, q_loss, skips = self.encode(x)
        recon = self.decode(z_q, skips)
        return recon, indices, q_loss


# =====================================================================
# VQTokenizer — VQ-VAE image-to-token pipeline
# =====================================================================


class VQTokenizer(nn.Module):
    """Tokenize images via VQ-VAE encoding.

    Produces a sequence of discrete tokens (via codebook lookup), arranges
    them in Hilbert order, and builds a 2-D grid KNN graph for the
    graph Q-shift in downstream RWKV-7 blocks.

    Args:
        in_chans: Input image channels.
        embed_dims: Projected token embedding dimension (backbone input dim).
        codebook_size: VQ codebook size.
        downsample_factor: Total spatial downsample (power of 2).
        latent_dim: Codebook latent dimension (defaults to embed_dims).
        num_res_blocks: Residual blocks in VQ-VAE encoder.
        use_ema: EMA-based codebook updates.
        beta: Commitment loss weight.
        knn_k: Number of KNN neighbours for graph Q-shift.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dims: int,
        codebook_size: int,
        downsample_factor: int = 16,
        latent_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        use_ema: bool = False,
        beta: float = 0.25,
        knn_k: int = 4,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.knn_k = knn_k
        self.downsample_factor = downsample_factor

        if latent_dim is None:
            latent_dim = embed_dims

        self.vqvae = ConvolutionalVQVAE(
            in_chans=in_chans,
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            downsample_factor=downsample_factor,
            num_res_blocks=num_res_blocks,
            use_ema=use_ema,
            beta=beta,
        )

        # Project latent_dim codebook vectors to embed_dims for backbone
        self.code_proj = nn.Linear(latent_dim, embed_dims)

    @staticmethod
    def _build_grid_centroids(
        h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build (N, 2) normalized centroids for a H×W grid."""
        grid_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        return torch.stack([gx.flatten(), gy.flatten()], dim=-1)

    def forward(
        self, x: torch.Tensor
    ) -> dict:
        """Tokenize an image batch into a token sequence.

        Args:
            x: (B, C, H, W) input tensor.
        Returns:
            dict with keys:
                tokens:      (B, N, embed_dims) token embeddings in Hilbert order.
                indices:     (B, H', W') codebook indices (original grid order).
                neighbors:   (B, N, K) KNN neighbour indices in Hilbert order.
                neighbor_dists: (B, N, K) distances in Hilbert order.
                inv_order:   (B, N) inverse Hilbert permutation.
                batch_idx:   (B, N) batch index tensor.
                q_loss:      Scalar quantization loss.
                h_s, w_s:    Token grid height and width.
        """
        B, C, H, W = x.shape
        H_tok = H // self.downsample_factor
        W_tok = W // self.downsample_factor
        N = H_tok * W_tok

        # 1. Encode and quantize → (B, latent_dim, H_tok, W_tok)
        z_q, indices, q_loss, _skips = self.vqvae.encode(x)

        # 2. Reshape to token sequence and project to embed_dims
        tokens = z_q.permute(0, 2, 3, 1).reshape(B, N, -1)
        tokens = self.code_proj(tokens)  # (B, N, embed_dims)

        # 3. Build 2-D grid centroid-based KNN graph
        # Use floating centroids in [-1, 1] for KNN graph (distance-based)
        centroids = self._build_grid_centroids(
            H_tok, W_tok, x.device, x.dtype
        ).unsqueeze(0).expand(B, -1, -1)
        neighbors, neighbor_dists = build_knn_graph(centroids, k=self.knn_k)

        # 4. Hilbert sort — use integer grid positions for the Hilbert curve
        grid_pos = torch.stack(
            torch.meshgrid(
                torch.arange(H_tok, device=x.device),
                torch.arange(W_tok, device=x.device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2).unsqueeze(0).expand(B, -1, -1)
        order = hilbert_sort_batched(grid_pos)
        batch_idx = torch.arange(B, device=x.device)  # (B,)
        inv_order = torch.zeros_like(order)
        inv_order.scatter_(
            1, order,
            torch.arange(N, device=order.device).unsqueeze(0).expand(B, -1),
        )

        tokens = tokens[batch_idx[:, None], order]
        neighbors = remap_neighbors(neighbors, order, inv_order)
        # neighbor_dists are float distances (not indices); permute directly
        neighbor_dists = neighbor_dists[batch_idx[:, None], order]

        return {
            "tokens": tokens,
            "indices": indices,
            "neighbors": neighbors,
            "neighbor_dists": neighbor_dists,
            "inv_order": inv_order,
            "batch_idx": batch_idx,
            "q_loss": q_loss,
            "h_s": H_tok,
            "w_s": W_tok,
        }


# =====================================================================
# VQ_RWKV7 — Backbone with VQ tokenization
# =====================================================================


class VQ_RWKV7(nn.Module):
    """VQ-RWKV-7 backbone with VQ-VAE tokenization and RWKV-7 blocks.

    Architecture follows Vision_RWKV7 but replaces the diffSLIC
    superpixel tokenizer with a learned VQ-VAE tokenizer. The VQ-VAE
    encoder produces a discrete codebook-index grid; codebook embeddings
    become the input tokens for the RWKV-7 backbone.
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
        codebook_size: int = 1024,
        downsample_factor: int = 16,
        latent_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        use_ema: bool = False,
        beta: float = 0.25,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        use_attnres: bool = False,
        attnres_mode: str = "block",
        attnres_gate_type: str = "bias",
        attnres_num_blocks: int = 8,
        attnres_recency_bias_init: float = 10.0,
        knn_k: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.num_layers = depth
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.scatter_output = scatter_output
        self.in_chans = in_chans
        self.downsample_factor = downsample_factor
        self.knn_k = knn_k

        if num_heads is None:
            assert embed_dims % HEAD_SIZE == 0, (
                f"embed_dims={embed_dims} not divisible by HEAD_SIZE={HEAD_SIZE}"
            )
            num_heads = embed_dims // HEAD_SIZE
        self.num_heads = num_heads

        # ---- VQ-VAE Tokenizer ----
        self.tokenizer = VQTokenizer(
            in_chans=in_chans,
            embed_dims=embed_dims,
            codebook_size=codebook_size,
            downsample_factor=downsample_factor,
            latent_dim=latent_dim,
            num_res_blocks=num_res_blocks,
            use_ema=use_ema,
            beta=beta,
            knn_k=self.knn_k,
        )

        # ---- CLS / Register tokens ----
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        self.register_tokens = register_tokens
        if register_tokens > 0:
            self.reg_token = nn.Parameter(
                torch.zeros(1, register_tokens, embed_dims)
            )
        else:
            self.reg_token = None

        # ---- RWKV-7 Blocks ----
        self.blocks = self._make_blocks(
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            with_cls_token=with_cls_token,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_attnres=use_attnres,
            attnres_mode=attnres_mode,
            attnres_gate_type=attnres_gate_type,
            attnres_num_blocks=attnres_num_blocks,
            attnres_recency_bias_init=attnres_recency_bias_init,
            **kwargs,
        )

        # ---- Final norm ----
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = get_norm_layer(norm_layer)(embed_dims)

        # ---- Output indices ----
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        indices = list(out_indices)
        for i, idx in enumerate(indices):
            if idx < 0:
                indices[i] = depth + idx
        self.out_indices = sorted(
            set(i for i in indices if 0 <= i < depth)
        ) or [depth - 1]

        # ---- AttnRes ----
        self.use_attnres = use_attnres
        self.attnres_mode = attnres_mode
        self.last_attnres_history = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Block factory
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
        use_attnres: bool,
        attnres_mode: str,
        attnres_gate_type: str,
        attnres_num_blocks: int,
        attnres_recency_bias_init: float,
        **kwargs,
    ) -> nn.ModuleList:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        return nn.ModuleList([
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
                use_attnres=use_attnres,
                attnres_mode=attnres_mode,
                attnres_gate_type=attnres_gate_type,
                attnres_num_blocks=attnres_num_blocks,
                attnres_recency_bias_init=attnres_recency_bias_init,
                num_prepend_tokens=self.register_tokens,
            )
            for i in range(depth)
        ])

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self):
        with torch.no_grad():
            if self.with_cls_token:
                self.cls_token.zero_()
            if self.reg_token is not None:
                self.reg_token.zero_()

    # ------------------------------------------------------------------
    # Output projection
    # ------------------------------------------------------------------

    def _project_output(
        self,
        patch_tokens: torch.Tensor,
        inv_order: torch.Tensor,
        batch_idx: torch.Tensor,
        H: int,
        W: int,
        h_s: int,
        w_s: int,
    ) -> torch.Tensor:
        """Reverse Hilbert sort and optionally scatter to pixel space."""
        patch_tokens = patch_tokens[batch_idx[:, None], inv_order]
        feat = patch_tokens.view(-1, h_s, w_s, self.embed_dims).permute(
            0, 3, 1, 2
        )
        if self.scatter_output:
            feat = F.interpolate(
                feat, size=(H, W), mode="bilinear", align_corners=False
            )
        return feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass returning multi-scale features.

        Args:
            x: (B, C, H, W) input tensor. C must equal ``in_chans``.

        Returns:
            Tuple of output feature maps, one per ``out_indices`` entry.
            Each feature map has shape (B, embed_dims, h_s, w_s) or
            (B, embed_dims, H, W) when ``scatter_output=True``.
        """
        B, C, H, W = x.shape
        assert C == self.in_chans, (
            f"Model initialized with in_chans={self.in_chans}, "
            f"but received input with C={C}."
        )

        # ---- Tokenization ----
        out = self.tokenizer(x)
        tokens = out["tokens"]
        neighbors = out["neighbors"]
        neighbor_dists = out["neighbor_dists"]
        inv_order = out["inv_order"]
        batch_idx = out["batch_idx"]
        h_s, w_s = out["h_s"], out["w_s"]
        self._last_q_loss = out["q_loss"]

        # Register tokens — prepended (DINOv2-style)
        # Neighbors stay at (B, N, K) — q_shift_graph_multihead handles
        # prepend tokens via num_prepend_tokens parameter.
        if self.register_tokens > 0:
            assert self.reg_token is not None
            reg_tokens = self.reg_token.expand(B, -1, -1)
            tokens = torch.cat((reg_tokens, tokens), dim=1)

        # CLS token — appended at sequence end
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat((tokens, cls_tokens), dim=1)

        # ---- RWKV-7 Blocks ----
        outs = []
        vf_fwd, vf_bwd = None, None
        attnres_history = [tokens] if self.use_attnres else None

        for i, block in enumerate(self.blocks):
            if self.use_attnres:
                tokens, vff, vfb = block(
                    tokens, neighbors, neighbor_dists, vf_fwd, vf_bwd,
                    mask=None, attnres_history=attnres_history,
                )
            else:
                tokens, vff, vfb = block(
                    tokens, neighbors, neighbor_dists, vf_fwd, vf_bwd,
                    mask=None,
                )
            if i == 0:
                vf_fwd, vf_bwd = vff, vfb
            if i == len(self.blocks) - 1 and self.final_norm:
                tokens = self.ln1(tokens)

            if i in self.out_indices:
                # Separate CLS and register tokens
                if self.with_cls_token:
                    cls_out = tokens[:, -1]
                    tokens_for_out = tokens[:, :-1]
                else:
                    cls_out = None
                    tokens_for_out = tokens

                if self.register_tokens > 0:
                    patch_tokens = tokens_for_out[:, self.register_tokens:]
                else:
                    patch_tokens = tokens_for_out

                feat = self._project_output(
                    patch_tokens, inv_order, batch_idx, H, W, h_s, w_s,
                )

                if self.output_cls_token and cls_out is not None:
                    outs.append((feat, cls_out))
                else:
                    outs.append(feat)

        if self.use_attnres:
            # Detach before storing — the history is for post-hoc visualization only.
            # Keeping live computation-graph tensors on self leaks memory across steps.
            self.last_attnres_history = [t.detach() for t in attnres_history] if attnres_history else None

        return tuple(outs)


# =====================================================================
# Model Builder
# =====================================================================


def create_vq_rwkv7(
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
    codebook_size: int = 1024,
    downsample_factor: int = 16,
    latent_dim: Optional[int] = None,
    num_res_blocks: int = 2,
    use_ema: bool = False,
    beta: float = 0.25,
    register_tokens: int = 0,
    norm_layer: str = "layernorm",
    act_layer: str = "relu2",
    use_attnres: bool = False,
    attnres_mode: str = "block",
    attnres_gate_type: str = "bias",
    attnres_num_blocks: int = 8,
    attnres_recency_bias_init: float = 10.0,
    knn_k: int = 4,
    use_jit: bool = False,
) -> torch.nn.Module:
    """Create a VQ_RWKV7 model enforced to 6-channel input.

    This is the standard entry point for this backbone.  ``in_chans=6``
    is enforced here; the class itself accepts arbitrary ``in_chans``
    for flexibility.
    """
    from spixrwkv7.jit import maybe_compile

    _model: torch.nn.Module = VQ_RWKV7(
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
        codebook_size=codebook_size,
        downsample_factor=downsample_factor,
        latent_dim=latent_dim,
        num_res_blocks=num_res_blocks,
        use_ema=use_ema,
        beta=beta,
        register_tokens=register_tokens,
        norm_layer=norm_layer,
        act_layer=act_layer,
        use_attnres=use_attnres,
        attnres_mode=attnres_mode,
        attnres_gate_type=attnres_gate_type,
        attnres_num_blocks=attnres_num_blocks,
        attnres_recency_bias_init=attnres_recency_bias_init,
        knn_k=knn_k,
    )
    return maybe_compile(_model, use_jit=use_jit)
