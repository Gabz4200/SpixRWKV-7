# Vision-RWKV-7: RWKV-7 vision backbone with Superpixel Tokenization (diffSLIC),
# Graph-Based Q-Shift, bidirectional scanning, gated fusion, and multi-scale output.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Sequence

# Import diffSLIC components from the same directory
from .diffSLIC import DiffSLIC, spixel_upsampling

HEAD_SIZE = 64
TIME_MIX_EXTRA_DIM = 32

# =====================================================================
# Utility modules
# =====================================================================


class Permute(nn.Module):
    """Channel-permute layer (compat for nn.Permute)."""

    __slots__ = ("dims",)

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


# =====================================================================
# Graph-Based Helpers (UPDATED FOR BATCHED GRAPHS)
# =====================================================================


def build_knn_graph(centroids: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Builds a K-Nearest Neighbors graph from superpixel centroids.
    Supports both single [N, 2] and batched [B, N, 2] centroids.
    """
    squeeze = False
    if centroids.dim() == 2:
        centroids = centroids.unsqueeze(0)
        squeeze = True

    B, N, _ = centroids.shape
    centroids = centroids.float()
    dists = torch.cdist(centroids, centroids)  # [B, N, N]

    # Fill diagonal with inf for each batch independently
    mask = (
        torch.eye(N, dtype=torch.bool, device=centroids.device)
        .unsqueeze(0)
        .expand(B, -1, -1)
    )
    dists = dists.masked_fill(mask, float("inf"))

    _, neighbors = torch.topk(dists, k, dim=2, largest=False)  # [B, N, k]

    if squeeze:
        neighbors = neighbors.squeeze(0)
    return neighbors


def q_shift_graph_multihead(
    input: torch.Tensor,
    neighbors: torch.Tensor,  # [N, K] or [B, N, K]
    head_dim: int = HEAD_SIZE,
    with_cls_token: bool = False,
) -> torch.Tensor:
    """Graph-based Q-Shift for superpixel or irregular grids.
    Now fully supports batched graphs [B, N, K] for data-dependent topologies.
    """
    B, N_total, C = input.shape
    assert C % head_dim == 0, f"C={C} not divisible by head_dim={head_dim}"
    n_head = C // head_dim

    # Handle unbatched neighbors by expanding to batch size
    if neighbors.dim() == 2:
        neighbors = neighbors.unsqueeze(0).expand(B, -1, -1)

    K = neighbors.shape[2]
    assert head_dim % K == 0, f"head_dim={head_dim} must be divisible by K={K}"
    group_size = head_dim // K

    cls_tokens = None
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
        N = N_total - 1
    else:
        N = N_total

    assert neighbors.shape[1] == N, (
        f"neighbors length {neighbors.shape[1]} must match N={N}"
    )

    x = input.view(B, N, n_head, head_dim)
    output = torch.zeros_like(x)
    safe_neighbors = neighbors.clamp(min=0)

    for k in range(K):
        neighbor_idx = safe_neighbors[:, :, k]  # [B, N]

        # Extract the k-th group from x: [B, N, n_head, group_size]
        x_group = x[:, :, :, k * group_size : (k + 1) * group_size]

        # Expand neighbor_idx for gathering: [B, N, n_head, group_size]
        idx = (
            neighbor_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_head, group_size)
        )

        # Gather features from the neighbors along the sequence dimension (dim=1)
        gathered_group = torch.gather(x_group, 1, idx)

        # Zero out features from invalid (-1) neighbors
        valid_k = (neighbors[:, :, k] != -1).view(B, N, 1, 1)
        gathered_group = gathered_group * valid_k

        # Assign to the k-th group of the output
        output[:, :, :, k * group_size : (k + 1) * group_size] = gathered_group

    output = output.view(B, N, C)
    if with_cls_token:
        output = torch.cat((output, cls_tokens), dim=1)
    return output


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    __slots__ = ("drop_prob",)

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# =====================================================================
# Vision_RWKV7_Block
# =====================================================================


class Vision_RWKV7_Block(nn.Module):
    """Vision-RWKV-7 block adapted for Graph-Based Q-Shift."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        layer_id: int,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        with_cls_token: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        assert self.head_size * n_head == n_embd
        self.with_cls_token = with_cls_token

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        # RWKV-7 HEAD-VARIANT PARAMETERS
        self.x = nn.Parameter(torch.zeros(6, n_embd))
        self.time_maa_x = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_w = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_v = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_g = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_a = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, TIME_MIX_EXTRA_DIM * 6))
        self.time_maa_w2 = nn.Parameter(torch.zeros(6, TIME_MIX_EXTRA_DIM, n_embd))

        # RWKV-7 DELTA-RULE PARAMETERS
        self.w0 = nn.Parameter(torch.zeros(n_embd))
        self.w1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.w2 = nn.Parameter(torch.zeros(32, n_embd))
        self.a0 = nn.Parameter(torch.zeros(n_embd))
        self.a1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.a2 = nn.Parameter(torch.zeros(32, n_embd))
        if layer_id != 0:
            self.v0 = nn.Parameter(torch.zeros(n_embd))
        self.v1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.v2 = nn.Parameter(torch.zeros(32, n_embd))
        self.g1 = nn.Parameter(torch.zeros(n_embd, 32))
        self.g2 = nn.Parameter(torch.zeros(32, n_embd))
        self.k_k = nn.Parameter(torch.zeros(n_embd))
        self.k_a = nn.Parameter(torch.zeros(n_embd))
        self.r_k = nn.Parameter(torch.zeros(n_head, self.head_size))

        self.att_receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.att_key = nn.Linear(n_embd, n_embd, bias=False)
        self.att_value = nn.Linear(n_embd, n_embd, bias=False)
        self.att_output = nn.Linear(n_embd, n_embd, bias=False)
        self.att_group_norm = nn.GroupNorm(
            n_head, n_embd, eps=self.n_head * 1e-5, affine=True
        )

        # Vision-specific additions
        self.fusion_gate = nn.Linear(n_embd, n_embd, bias=False)
        self.att_ln = nn.LayerNorm(n_embd)
        self.ffn_ln = nn.LayerNorm(n_embd)

        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd))
            self.gamma2 = nn.Parameter(init_values * torch.ones(n_embd))
        else:
            self.gamma1 = None
            self.gamma2 = None

        self.ffn_x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        dim_ffn = 4 * n_embd
        self.ffn_key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.ffn_value = nn.Linear(dim_ffn, n_embd, bias=False)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.n_layer <= 1:
                ratio_0_to_1, ratio_1_to_almost0 = 0.0, 0.5
            else:
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

            idx = torch.arange(self.n_embd, dtype=torch.float) / max(self.n_embd - 1, 1)
            ddd = idx.view(1, 1, self.n_embd)
            self.x.uniform_(-0.01, 0.01)

            def fancy_mix(base_pow):
                return 1.0 - torch.pow(ddd, base_pow)

            self.time_maa_x.copy_(fancy_mix(ratio_1_to_almost0))
            self.time_maa_w.copy_(fancy_mix(ratio_1_to_almost0))
            self.time_maa_k.copy_(fancy_mix(ratio_1_to_almost0))
            self.time_maa_v.copy_(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r.copy_(fancy_mix(0.5 * ratio_1_to_almost0))
            self.time_maa_g.copy_(fancy_mix(0.5 * ratio_1_to_almost0))
            self.time_maa_a.copy_(fancy_mix(0.5 * ratio_1_to_almost0))
            self.time_maa_w1.uniform_(-1e-4, 1e-4)
            self.time_maa_w2.uniform_(-1e-4, 1e-4)

            decay_speed = -3 + 5 * idx ** (0.7 + 1.3 * ratio_0_to_1)
            self.w0.copy_(decay_speed)

            tmp = torch.zeros(self.n_head, self.head_size)
            for h in range(self.n_head):
                for n in range(self.head_size):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[h, n] = ratio_0_to_1 * (1 - n / (self.head_size - 1)) + zigzag
            self.r_k.copy_(tmp)

            for p in [
                self.w1,
                self.w2,
                self.a1,
                self.a2,
                self.v1,
                self.v2,
                self.g1,
                self.g2,
            ]:
                p.uniform_(-1e-4, 1e-4)

    def _spatial_prep(self, xn: torch.Tensor, neighbors: torch.Tensor):
        """Graph-based Q-Shift + input-dependent dynamic offsets."""
        B, N, D = xn.shape
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
        )
        xx = xs - xn

        x_base = xn + xx * self.time_maa_x
        x_dyn = torch.tanh(x_base @ self.time_maa_w1)
        x_dyn = x_dyn.view(B * N, 6, -1).transpose(0, 1)
        x_dyn = torch.bmm(x_dyn, self.time_maa_w2)
        dm = x_dyn.view(6, B, N, D)
        return {"xx": xx, "dm": dm}

    def _scan(self, xn, sp, direction, v_first_seq):
        B, N, D = xn.shape
        Hd, S, dev = self.n_head, self.head_size, xn.device
        rev = direction == "backward"

        xn_seq = torch.flip(xn, dims=[1]) if rev else xn
        xx_seq = torch.flip(sp["xx"], dims=[1]) if rev else sp["xx"]
        dm_seq = torch.flip(sp["dm"], dims=[2]) if rev else sp["dm"]
        vf_seq = (
            torch.flip(v_first_seq, dims=[1])
            if rev and v_first_seq is not None
            else v_first_seq
        )

        state = torch.zeros(B, Hd, S, S, device=dev)
        state_time = torch.zeros(B, D, device=dev)

        sw, sk, sv, sr, sg, sa = [
            getattr(self, f"time_maa_{m}").reshape(-1)
            for m in ["w", "k", "v", "r", "g", "a"]
        ]
        x0, x1, x2, x3, x4, x5 = self.x.unbind(dim=0)

        outputs, v_first_list = [], []
        for t in range(N):
            token, xx_t = xn_seq[:, t, :], xx_seq[:, t, :]
            dm_t = dm_seq[:, :, t, :]
            dmw, dmk, dmv, dmr, dmg, dma = dm_t.unbind(dim=0)

            sx = state_time - token
            state_time.copy_(token)

            xw = token + sx * x0 + xx_t * (sw + dmw)
            xk = token + sx * x1 + xx_t * (sk + dmk)
            xv = token + sx * x2 + xx_t * (sv + dmv)
            xr = token + sx * x3 + xx_t * (sr + dmr)
            xg_in = token + sx * x4 + xx_t * (sg + dmg)
            xa_in = token + sx * x5 + xx_t * (sa + dma)

            w_raw = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
            w = torch.exp(-0.606531 * torch.sigmoid(w_raw.float()))

            r, k, v = self.att_receptance(xr), self.att_key(xk), self.att_value(xv)

            if self.layer_id == 0:
                vf = v
                v_first_list.append(vf)
            else:
                vf = vf_seq[:, t, :] if vf_seq is not None else v
                vr = self.v0 + (xv @ self.v1) @ self.v2
                v = vf + (v - vf) * torch.sigmoid(vr)

            a = torch.sigmoid(self.a0 + (xa_in @ self.a1) @ self.a2)
            g = torch.sigmoid(xg_in @ self.g1) @ self.g2

            kk = F.normalize((k * self.k_k).view(B, Hd, S), dim=-1, p=2.0).view(B, -1)
            kt = k * (1 + (a - 1) * self.k_a)

            vk = v.view(B, Hd, S, 1) @ kt.view(B, Hd, 1, S)
            ab = (-kk).view(B, Hd, S, 1) @ (kk * a).view(B, Hd, 1, S)
            state = state * w.view(B, Hd, 1, S) + state @ ab.float() + vk.float()

            r_h = r.view(B, Hd, S).unsqueeze(-1)
            out = (state @ r_h).squeeze(-1)
            out = self.att_group_norm(out.flatten(start_dim=1))

            # BONUS TERM FIX: Uses kt (replacement key) instead of k, per RWKV-7 Eq. 20
            bonus = (
                (r.view(B, Hd, S) * kt.view(B, Hd, S) * self.r_k.view(Hd, S)).sum(
                    dim=-1, keepdim=True
                )
                * v.view(B, Hd, S)
            ).view(B, D)
            out = self.att_output((out + bonus) * g)
            outputs.append(out)

        out = torch.stack(outputs, dim=1)
        if rev:
            out = torch.flip(out, dims=[1])

        v_first_out = None
        if self.layer_id == 0:
            v_first_out = torch.stack(v_first_list, dim=1)
            if rev:
                v_first_out = torch.flip(v_first_out, dims=[1])
        return out, v_first_out

    def forward(self, x, neighbors, v_first_fwd=None, v_first_bwd=None):
        if self.layer_id == 0:
            x = self.ln0(x)
        xn = self.ln1(x)
        sp = self._spatial_prep(xn, neighbors)
        x_gate = xn + sp["xx"] * 0.5

        out_fwd, vf_fwd = self._scan(xn, sp, "forward", v_first_fwd)
        out_bwd, vf_bwd = self._scan(xn, sp, "backward", v_first_bwd)

        gate = torch.sigmoid(self.fusion_gate(x_gate))
        att_out = gate * out_fwd + (1 - gate) * out_bwd
        if self.gamma1 is not None:
            att_out = self.gamma1 * att_out
        x = x + self.drop_path(self.att_ln(att_out))

        xn = self.ln2(x)
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
        )
        xx = xs - xn
        xk = xn + xx * self.ffn_x_k
        k = F.relu(self.ffn_key(xk)).pow(2)
        ffn_out = self.ffn_value(k)
        if self.gamma2 is not None:
            ffn_out = self.gamma2 * ffn_out
        x = x + self.drop_path(self.ffn_ln(ffn_out))

        return x, vf_fwd, vf_bwd


# =====================================================================
# Tokenization + Embedding (UPDATED FOR DYNAMIC K)
# =====================================================================


class SuperpixelEmbedding(nn.Module):
    """Convert 2D image to superpixel tokens via mask-based aggregation."""

    def __init__(
        self, in_chans: int, embed_dims: int, num_superpixels: int, mode: str = "hard"
    ):
        super().__init__()
        assert mode in ("hard", "soft"), "mode must be 'hard' or 'soft'"
        self.num_superpixels = num_superpixels
        self.mode = mode
        self.proj = nn.Linear(in_chans, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x: torch.Tensor, sp_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Dynamically infer K to prevent crashes if diffSLIC alters the grid slightly
        K = sp_map.max().item() + 1

        if self.mode == "hard":
            K = max(sp_map.max().item() + 1, self.num_superpixels)
            mask = F.one_hot(sp_map.long(), num_classes=K).permute(0, 3, 1, 2).float()
        else:
            mask = sp_map
            K = mask.shape[1]

        weights = mask / (mask.sum(dim=(2, 3), keepdim=True) + 1e-6)
        sp_features = torch.einsum("bkhw,bchw->bkc", weights, x)
        return self.norm(self.proj(sp_features))


# =====================================================================
# Vision_RWKV7 (Backbone)
# =====================================================================


class Vision_RWKV7(nn.Module):
    """Vision-RWKV-7 backbone with diffSLIC Superpixel Tokenization and Graph Q-Shift."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dims: int = 192,
        num_heads: int = 3,
        depth: int = 12,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 1e-5,
        final_norm: bool = True,
        out_indices: Sequence[int] = (-1,),
        with_cls_token: bool = False,
        output_cls_token: bool = False,
        num_superpixels: int = 196,
        diff_slic_iters: int = 5,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = depth
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token

        # 1. Differentiable SLIC for dynamic superpixel generation
        self.diff_slic = DiffSLIC(
            n_spixels=num_superpixels,
            n_iter=diff_slic_iters,
            tau=0.01,
            candidate_radius=1,
        )

        # 2. Superpixel Embedding (Mask-based aggregation)
        self.patch_embed = SuperpixelEmbedding(
            in_chans, embed_dims, num_superpixels, mode="hard"
        )

        # 3. 1D Positional Embedding (Size is K, we add a buffer just in case K varies slightly by aspect ratio)
        self.max_K = num_superpixels + 16
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_K, embed_dims))

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Vision_RWKV7_Block(
                    embed_dims, num_heads, depth, i, dpr[i], init_values, with_cls_token
                )
                for i in range(depth)
            ]
        )

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(embed_dims)

        out_indices = (
            [out_indices] if isinstance(out_indices, int) else list(out_indices)
        )
        for i, idx in enumerate(out_indices):
            if idx < 0:
                out_indices[i] = depth + idx
        self.out_indices = sorted(set(i for i in out_indices if 0 <= i < depth)) or [
            depth - 1
        ]

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.with_cls_token:
                self.cls_token.zero_()
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # ---------------------------------------------------------
        # 1. Generate Superpixels via diffSLIC
        # ---------------------------------------------------------
        clst_feats, p2s_assign, _ = self.diff_slic(x)
        h_s, w_s = clst_feats.shape[-2:]
        K = h_s * w_s
        radius = self.diff_slic.candidate_radius

        # ---------------------------------------------------------
        # 2. Tokenize & Embed (Mode-dependent logic)
        # ---------------------------------------------------------
        if self.patch_embed.mode == "hard":
            # HARD MODE: Use argmax to get discrete integer labels [B, H, W]
            neighbor_range = 2 * radius + 1
            hard_assign = (
                F.one_hot(p2s_assign.argmax(1), neighbor_range**2)
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
            )
            label_grid = (
                torch.arange(K, dtype=torch.float, device=x.device)
                .reshape(1, 1, h_s, w_s)
                .expand(B, -1, -1, -1)
            )

            global_labels = (
                spixel_upsampling(label_grid, hard_assign, candidate_radius=radius)
                .squeeze(1)
                .long()
            )
            tokens = self.patch_embed(x, global_labels)

        else:
            # SOFT MODE: Bypass argmax! Create one-hot superpixel IDs [1, K, H_s, W_s]
            spixel_ids = (
                torch.arange(K, device=x.device)
                .reshape(1, K, 1, 1)
                .expand(B, -1, h_s, w_s)
                .float()
            )

            # Upsample using the raw local soft assignments (p2s_assign) directly.
            # Because spixel_ids is one-hot, spixel_upsampling will output a global
            # soft probability map of shape [B, K, H, W].
            global_soft_mask = spixel_upsampling(
                spixel_ids, p2s_assign, candidate_radius=radius
            )

            tokens = self.patch_embed(x, global_soft_mask)

        # Add 1D positional embedding
        tokens = tokens + self.pos_embed[:, :K, :]

        # Compute Centroids to build KNN Graph
        # Note: For soft mode, we use the soft mask to compute weighted centroids
        if self.patch_embed.mode == "soft":
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(H, device=x.device),
                    torch.arange(W, device=x.device),
                    indexing="ij",
                ),
                dim=-1,
            ).float()
            # global_soft_mask is [B, K, H, W], coords is [H, W, 2]
            counts = global_soft_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            centroids = torch.einsum(
                "bkhw,hwc->bkc", global_soft_mask, coords
            ) / counts.squeeze(-1)
        else:
            # Hard mode centroid calculation (unchanged)
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(H, device=x.device),
                    torch.arange(W, device=x.device),
                    indexing="ij",
                ),
                dim=-1,
            ).float()
            one_hot = (
                F.one_hot(global_labels, num_classes=K).permute(0, 3, 1, 2).float()
            )
            counts = one_hot.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            centroids = torch.einsum(
                "bkhw,bhwc->bkc", one_hot, coords
            ) / counts.squeeze(-1)

        # Build Graph
        neighbors = build_knn_graph(centroids.detach(), k=4)

        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat((tokens, cls_tokens), dim=1)

        # ---------------------------------------------------------
        # 3. RWKV-7 Blocks (Unchanged)
        # ---------------------------------------------------------
        outs = []
        vf_fwd, vf_bwd = None, None
        for i, block in enumerate(self.blocks):
            tokens, vff, vfb = block(tokens, neighbors, vf_fwd, vf_bwd)
            if i == 0:
                vf_fwd, vf_bwd = vff, vfb
            if i == len(self.blocks) - 1 and self.final_norm:
                tokens = self.ln1(tokens)

            if i in self.out_indices:
                patch_tokens = tokens[:, :-1] if self.with_cls_token else tokens
                cls_out = tokens[:, -1] if self.with_cls_token else None

                # SCATTER BACK TO GRID
                if self.patch_embed.mode == "soft":
                    # For soft mode, we scatter using the soft mask (weighted sum of tokens)
                    # patch_tokens is [B, K, D], global_soft_mask is [B, K, H, W]
                    feat = torch.einsum(
                        "bkd,bkhw->bhwd", patch_tokens, global_soft_mask
                    )
                    feat = feat.permute(0, 3, 1, 2)  # [B, D, H, W]
                else:
                    # For hard mode, we gather using the hard labels
                    feat = patch_tokens.gather(
                        1,
                        global_labels.view(B, H * W, 1).expand(-1, -1, self.embed_dims),
                    )
                    feat = feat.view(B, H, W, self.embed_dims).permute(0, 3, 1, 2)

                if self.output_cls_token and cls_out is not None:
                    outs.append((feat, cls_out))
                else:
                    outs.append(feat)

        return tuple(outs)
