# Vision-RWKV-7: RWKV-7 vision backbone with Superpixel Tokenization (diffSLIC),
# Graph-Based Q-Shift, bidirectional scanning, gated fusion, and multi-scale output.

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spixrwkv7.data.diff_slic import DiffSLIC, spixel_upsampling
from spixrwkv7.data.lnsnet import LNSNet, download_lnsnet_weights, lnsnet_assignment
from spixrwkv7.layers.drop import DropPath
from spixrwkv7.layers.graph import HEAD_SIZE, build_knn_graph, q_shift_graph_multihead
from spixrwkv7.models.common import (
    TIME_MIX_EXTRA_DIM,
    DynamicOffset,
    apply_activation,
    apply_attnres_gate,
    compute_attnres_config,
    init_attnres_params,
    init_backbone_tokens,
    normalize_out_indices,
    resolve_num_heads,
    zero_init_backbone_tokens,
)

_HILBERT_BITS = 13


def _hilbert_xy_to_d(x: torch.Tensor, y: torch.Tensor, order: int) -> torch.Tensor:
    """Vectorized 2-D Hilbert index — pure PyTorch, no Python loop over N.

    Runs `order` loop iterations (one per bit level), each doing vectorized
    ops over all (B, N) points simultaneously.  Replaces the original
    per-point Python call loop which synced GPU→CPU for every centroid.

    Args:
        x, y: integer coordinate tensors, shape (B, N), values in [0, 2^order).
        order: number of bits (curve resolution = 2^order).

    Returns:
        Hilbert distance tensor, shape (B, N), dtype long.
    """
    n = 1 << order          # total grid size per axis
    x, y = x.clone(), y.clone()
    d = torch.zeros_like(x, dtype=torch.long)
    s = n >> 1              # start at n//2, halve each iteration
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d = d + s * s * ((3 * rx) ^ ry)
        # rot(n, x, y, rx, ry): if ry==0 and rx==1, reflect; then if ry==0, swap.
        flip = (ry == 0) & (rx == 1)
        x = torch.where(flip, n - 1 - x, x)
        y = torch.where(flip, n - 1 - y, y)
        swap = (ry == 0)
        x, y = torch.where(swap, y, x), torch.where(swap, x, y)
        s >>= 1
    return d


def hilbert_sort_batched(coords_int: torch.Tensor) -> torch.Tensor:
    """Sort token indices by 2-D Hilbert order for each batch item.

    Fully vectorized: no Python loop over tokens and no GPU→CPU sync.
    """
    x = coords_int[..., 0].long()
    y = coords_int[..., 1].long()
    distances = _hilbert_xy_to_d(x, y, _HILBERT_BITS)
    return torch.argsort(distances, dim=-1)


def remap_neighbors(
    neighbors: torch.Tensor, order: torch.Tensor, inv_order: torch.Tensor
) -> torch.Tensor:
    """Convert neighbor indices from original token order to Hilbert order."""
    valid = neighbors != -1
    batch_idx = torch.arange(neighbors.shape[0], device=neighbors.device).view(
        -1, 1, 1
    )
    remapped = inv_order[batch_idx, neighbors.clamp(min=0)]
    return remapped.masked_fill(~valid, -1).to(dtype=neighbors.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.scale


def get_norm_layer(name: str):
    if name == "rmsnorm":
        return RMSNorm
    return nn.LayerNorm

# =====================================================================
# RecurrentScan — Single-direction RWKV-7 recurrent scan
# =====================================================================


class _TimeMixParams(nn.Module):
    """RWKV-7 time-mixing (head-variant) parameters shared by all directions."""

    def __init__(self, n_embd: int):
        super().__init__()
        # Head-variant mixing weights
        self.time_maa_w = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_v = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_g = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_a = nn.Parameter(torch.zeros(1, 1, n_embd))


class RecurrentScan(nn.Module):
    """RWKV-7 head-variant + delta-rule recurrent scan for one direction.

    Encapsulates all core RWKV-7 recurrence parameters: the 6 head vectors (x),
    time-mixing coefficients, delta-rule parameters (w, a, v, g), decoupled keys
    (k_k, k_a), bonus receptance (r_k), linear projections, and group norm.
    """

    def __init__(self, n_embd: int, n_head: int, layer_id: int, n_layer: int, use_cpp: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        self.use_cpp = use_cpp
        assert self.head_size * n_head == n_embd

        # Head-variant parameters
        self.x = nn.Parameter(torch.zeros(6, n_embd))
        self.time_mix = _TimeMixParams(n_embd)

        # Delta-rule parameters
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
        self.init_state = nn.Parameter(torch.zeros(n_head, self.head_size, self.head_size))

        # Linear projections
        self.att_receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.att_key = nn.Linear(n_embd, n_embd, bias=False)
        self.att_value = nn.Linear(n_embd, n_embd, bias=False)
        self.att_output = nn.Linear(n_embd, n_embd, bias=False)
        self.att_group_norm = nn.GroupNorm(
            n_head, n_embd, eps=1e-5, affine=True
        )

    def forward(
        self,
        xn: torch.Tensor,
        xx: torch.Tensor,
        dm: torch.Tensor,
        direction: str,
        v_first_seq: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = xn.shape
        Hd, S, dev = self.n_head, self.head_size, xn.device
        rev = direction == "backward"

        xn_seq = torch.flip(xn, dims=[1]) if rev else xn
        xx_seq = torch.flip(xx, dims=[1]) if rev else xx
        dm_seq = torch.flip(dm, dims=[2]) if rev else dm
        vf_seq = (
            torch.flip(v_first_seq, dims=[1])
            if rev and v_first_seq is not None
            else v_first_seq
        )
        mask_seq = torch.flip(mask, dims=[1]) if (mask is not None and rev) else mask

        state = self.init_state[:Hd, :S, :S].unsqueeze(0).expand(B, -1, -1, -1).clone()
        state_time = torch.zeros(B, D, device=dev)

        tm = self.time_mix
        sw, sk, sv, sr, sg, sa = [
            getattr(tm, f"time_maa_{m}").reshape(-1)
            for m in ["w", "k", "v", "r", "g", "a"]
        ]
        x0, x1, x2, x3, x4, x5 = self.x.unbind(dim=0)

        if self.use_cpp:
            r_all = torch.empty(B, N, Hd, S, device=dev)
            v_all = torch.empty(B, N, Hd, S, device=dev)
            w_all = torch.empty(B, N, Hd, S, device=dev)
            a_all = torch.empty(B, N, Hd, S, device=dev)
            kk_all = torch.empty(B, N, Hd, S, device=dev)
            kt_all = torch.empty(B, N, Hd, S, device=dev)
            g_list = []
            v_first_list = []

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
                w_t = torch.exp(-0.606531 * torch.sigmoid(w_raw.float()))
                if mask_seq is not None:
                    w_t = torch.where(mask_seq[:, t, None] == 0, torch.ones_like(w_t), w_t)

                r_t = self.att_receptance(xr)
                k_t = self.att_key(xk)
                v_t = self.att_value(xv)

                if self.layer_id == 0:
                    v_first_list.append(v_t)
                else:
                    vf_t = vf_seq[:, t, :] if vf_seq is not None else v_t
                    vr = self.v0 + (xv @ self.v1) @ self.v2
                    v_t = vf_t + (v_t - vf_t) * torch.sigmoid(vr)

                a_t = torch.sigmoid(self.a0 + (xa_in @ self.a1) @ self.a2)
                g_t = torch.sigmoid(xg_in @ self.g1) @ self.g2

                kk_t = F.normalize(
                    (k_t * self.k_k).view(B, Hd, S), dim=-1, p=2.0
                ).view(B, D)
                kt_t = k_t * (1 + (a_t - 1) * self.k_a)

                r_all[:, t] = r_t.view(B, Hd, S)
                v_all[:, t] = v_t.view(B, Hd, S)
                w_all[:, t] = w_t.view(B, Hd, S)
                a_all[:, t] = a_t.view(B, Hd, S)
                kk_all[:, t] = kk_t.view(B, Hd, S)
                kt_all[:, t] = kt_t.view(B, Hd, S)
                g_list.append(g_t)

            out_scan = torch.ops.spixrwkv7.rwkv7_recurrent_scan(
                state, r_all, v_all, w_all, a_all, kk_all, kt_all,
            )

            outputs = []
            for t in range(N):
                raw = out_scan[:, t]
                r_t = r_all[:, t]
                kt_t = kt_all[:, t]
                v_t = v_all[:, t]
                g_t = g_list[t]

                out = raw.view(B, D) if raw.shape[-1] == S else raw
                out = self.att_group_norm(out)

                bonus_scalar = (r_t * kt_t * self.r_k.unsqueeze(0)).sum(dim=-1, keepdim=True)
                bonus = (bonus_scalar * v_t).reshape(B, D)
                out = out + bonus

                out = self.att_output(out * g_t)
                if mask_seq is not None:
                    out = out * mask_seq[:, t, None]
                outputs.append(out)

            out = torch.stack(outputs, dim=1)
        else:
            outputs, v_first_list = [], []
            for t in range(N):
                mask_t = mask_seq[:, t, None, None] if mask_seq is not None else 1.0
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
                if mask_seq is not None:
                    w = torch.where(mask_seq[:, t, None] == 0, torch.ones_like(w), w)

                r, k, v = self.att_receptance(xr), self.att_key(xk), self.att_value(xv)

                if self.layer_id == 0:
                    vf = v
                    v_first_list.append(vf)
                else:
                    vf = vf_seq[:, t, :] if vf_seq is not None else v
                    vr = self.v0 + (xv @ self.v1) @ self.v2
                    v = vf + (v - vf) * torch.sigmoid(vr)

                a_out = torch.sigmoid(self.a0 + (xa_in @ self.a1) @ self.a2)
                g = torch.sigmoid(xg_in @ self.g1) @ self.g2

                kk = F.normalize((k * self.k_k).view(B, Hd, S), dim=-1, p=2.0).view(B, -1)
                kt = k * (1 + (a_out - 1) * self.k_a)

                vk = v.view(B, Hd, S, 1) @ kt.view(B, Hd, 1, S)
                ab = (-kk).view(B, Hd, S, 1) @ (kk * a_out).view(B, Hd, 1, S)
                state = state * w.view(B, Hd, 1, S) + (state @ ab.float() + vk.float()) * mask_t

                r_h = r.view(B, Hd, S).unsqueeze(-1)
                out = (state @ r_h).squeeze(-1)
                out = self.att_group_norm(out.flatten(start_dim=1))

                bonus = (
                    (r.view(B, Hd, S) * kt.view(B, Hd, S) * self.r_k.view(Hd, S)).sum(
                        dim=-1, keepdim=True
                    )
                    * v.view(B, Hd, S)
                ).view(B, D)
                out = self.att_output((out + bonus) * g)
                if mask_seq is not None:
                    out = out * mask_seq[:, t, None]
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

    def init_weights(
        self,
        ratio_0_to_1: float,
        ratio_1_to_almost0: float,
        ddd: torch.Tensor,
    ):
        """Fancy initialization of all RWKV-7 scan parameters."""
        with torch.no_grad():
            n_embd = self.n_embd
            idx = torch.arange(n_embd, dtype=torch.float) / max(n_embd - 1, 1)
            self.x.uniform_(-0.01, 0.01)

            def fancy_mix(base_pow):
                return 1.0 - torch.pow(ddd, base_pow)

            tm = self.time_mix
            tm.time_maa_w.copy_(fancy_mix(ratio_1_to_almost0))
            tm.time_maa_k.copy_(fancy_mix(ratio_1_to_almost0))
            tm.time_maa_v.copy_(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            tm.time_maa_r.copy_(fancy_mix(0.5 * ratio_1_to_almost0))
            tm.time_maa_g.copy_(fancy_mix(0.5 * ratio_1_to_almost0))
            tm.time_maa_a.copy_(fancy_mix(0.5 * ratio_1_to_almost0))

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


# =====================================================================
# SpatialMixer — Graph Q-Shift + bidirectional RWKV-7 recurrence
# =====================================================================


class SpatialMixer(nn.Module):
    """Bidirectional spatial (time) mixing with graph Q-shift and RWKV-7 recurrence.

    Composes:
      1. Graph Q-shift of input tokens
      2. Input-dependent dynamic offset computation
      3. Forward + backward RecurrentScan
      4. Gated bidirectional fusion with residual
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        layer_id: int,
        drop_prob: float = 0.0,
        init_values: Optional[float] = None,
        with_cls_token: bool = False,
        norm_layer: str = "layernorm",
        num_prepend_tokens: int = 0,
        use_cpp: bool = False,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        self.with_cls_token = with_cls_token
        self.num_prepend_tokens = num_prepend_tokens
        self.layer_id = layer_id
        self.n_layer = n_layer

        self.dynamic_offset = DynamicOffset(n_embd)
        self.scan = RecurrentScan(n_embd, n_head, layer_id, n_layer, use_cpp=use_cpp)
        self.fusion_gate = nn.Linear(n_embd, n_embd, bias=False)
        self.gate_scale = nn.Parameter(torch.tensor(0.5))
        self.att_ln = get_norm_layer(norm_layer)(n_embd)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

        # LayerScale (CaiT et al., 2021; SwitchBack, Wortsman et al., 2023):
        # Per-channel learnable scale applied after sublayer output, before residual.
        # Zero initialization suppresses extreme feature magnitudes at start of training,
        # allowing gradual warm-up as gamma learns non-zero values.
        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd))
        else:
            self.gamma1 = None

    def _spatial_prep(
        self,
        xn: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Graph Q-shift followed by dynamic offset computation."""
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
            num_prepend_tokens=self.num_prepend_tokens,
        )
        xx = xs - xn
        dm = self.dynamic_offset(xn, xx)
        return xx, dm

    def forward(
        self,
        x: torch.Tensor,
        xn: torch.Tensor,
        neighbors: torch.Tensor,
        dists: Optional[torch.Tensor] = None,
        v_first_fwd: Optional[torch.Tensor] = None,
        v_first_bwd: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        xx, dm = self._spatial_prep(xn, neighbors)
        x_gate = xn + xx * self.gate_scale

        out_fwd, vf_fwd = self.scan(xn, xx, dm, "forward", v_first_fwd, mask=mask)
        out_bwd, vf_bwd = self.scan(xn, xx, dm, "backward", v_first_bwd, mask=mask)

        gate = torch.sigmoid(self.fusion_gate(x_gate))
        att_out = gate * out_fwd + (1 - gate) * out_bwd
        att_out = self.att_ln(att_out)
        if self.gamma1 is not None:
            att_out = self.gamma1 * att_out  # LayerScale (CaiT et al., 2021)
        x = x + self.drop_path(att_out)
        return x, vf_fwd, vf_bwd


# =====================================================================
# ChannelMix — Q-shift gated feed-forward network
# =====================================================================


class ChannelMix(nn.Module):
    """Graph Q-shift gated feed-forward network with residual.

    Applies q_shift_graph_multihead to the input, then a gated FFN
    (ReLU² activation) with learnable input-dependent mixing (ffn_x_k).
    """

    def __init__(
        self,
        n_embd: int,
        drop_prob: float = 0.0,
        init_values: Optional[float] = None,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        num_prepend_tokens: int = 0,
    ):
        super().__init__()
        self.act_layer = act_layer
        dim_ffn = 4 * n_embd
        self.ffn_x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        if act_layer == "swiglu":
            self.ffn_key = nn.Linear(n_embd, 2 * dim_ffn, bias=False)
        else:
            self.ffn_key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.ffn_value = nn.Linear(dim_ffn, n_embd, bias=False)
        norm_cls = get_norm_layer(norm_layer)
        self.norm = norm_cls(n_embd)
        self.num_prepend_tokens = num_prepend_tokens
        self.ffn_dropout = nn.Dropout(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

        # LayerScale: per-channel learnable scale after sublayer output.
        # Zero initialization prevents extreme feature blow-up at training start.
        if init_values is not None:
            self.gamma2 = nn.Parameter(init_values * torch.ones(n_embd))
        else:
            self.gamma2 = None
    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        dists: Optional[torch.Tensor] = None,
        head_dim: int = 64,
        with_cls_token: bool = False,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xn = self.norm(h if h is not None else x)
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            head_dim=head_dim,
            with_cls_token=with_cls_token,
            num_prepend_tokens=self.num_prepend_tokens,
        )
        xx = xs - xn
        xk = xn + xx * self.ffn_x_k
        k = apply_activation(xk, self.act_layer, self.ffn_key)
        k = self.ffn_dropout(k)
        ffn_out = self.ffn_value(k)
        if self.gamma2 is not None:
            ffn_out = self.gamma2 * ffn_out  # LayerScale (CaiT et al., 2021)
        return x + self.drop_path(ffn_out)


# =====================================================================
# Vision_RWKV7_Block (refactored)
# =====================================================================


# =====================================================================
# Core Block-AttnRes helper
# =====================================================================

def block_attn_res(
    blocks: list[torch.Tensor],   # completed blocks  [B, Seq, D] each
    partial_block: torch.Tensor,  # current intra-block partial sum  [B, Seq, D]
    proj: nn.Linear,              # learned pseudo-query weight  (d,)
    norm: nn.Module,              # Norm applied to keys before scoring
    recency_bias: nn.Parameter,   # scalar bias added to partial_block's logit
) -> torch.Tensor:
    """Attend over all block representations + the current partial block."""
    # Stack everything: shape [N+1, B, Seq, D]
    V = torch.stack(blocks + [partial_block], dim=0)

    # Keys = normalised values
    K = norm(V)

    # Scalar logit per (block, batch, token) via the single learned query
    query = proj.weight.view(-1)                              # (D,)
    logits = torch.einsum("d, n b t d -> n b t", query, K)   # (N+1, B, T)

    # Recency bias: boost the last element (partial_block)
    logits[-1] = logits[-1] + recency_bias

    # Softmax across block dimension
    weights = logits.softmax(dim=0)                           # (N+1, B, T)

    # Weighted sum of values
    h = torch.einsum("n b t, n b t d -> b t d", weights, V)  # (B, T, D)
    return h


class Vision_RWKV7_Block(nn.Module):
    """Vision-RWKV-7 block composing SpatialMixer and ChannelMix.

    Architecture:
      LN0 (layer 0 only) → LN1 → SpatialMixer → LN → LayerScale → residual
      → LN2 → ChannelMix → LN → LayerScale → residual

    LayerScale (per-channel learnable scale, initialized to init_values) is
    applied after each sublayer's LayerNorm, before the residual connection.
    Zero initialization effectively starts the block as identity, suppressing
    extreme feature magnitudes early in training per SwitchBack (Wortsman et al., 2023).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        layer_id: int,
        drop_prob: float = 0.0,
        init_values: Optional[float] = None,
        with_cls_token: bool = False,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        use_attnres: bool = False,
        attnres_mode: str = "block",
        attnres_gate_type: str = "bias",
        attnres_num_blocks: int = 8,
        attnres_recency_bias_init: float = 10.0,
        num_prepend_tokens: int = 0,
        use_cpp: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        self.with_cls_token = with_cls_token
        self.num_prepend_tokens = num_prepend_tokens

        # Attention Residuals configuration
        self.use_attnres = use_attnres
        self.attnres_mode = attnres_mode
        self.attnres_gate_type = attnres_gate_type
        self.attnres_num_blocks = attnres_num_blocks
        self.attnres_recency_bias_init = attnres_recency_bias_init

        if use_attnres:
            self.attn_res_proj = nn.Linear(n_embd, 1, bias=False)
            self.attn_res_norm = get_norm_layer(norm_layer)(n_embd)
            self.attn_res_bias = nn.Parameter(torch.tensor(attnres_recency_bias_init))

            self.mlp_res_proj = nn.Linear(n_embd, 1, bias=False)
            self.mlp_res_norm = get_norm_layer(norm_layer)(n_embd)
            self.mlp_res_bias = nn.Parameter(torch.tensor(attnres_recency_bias_init))

            if attnres_gate_type == "sigmoid_scalar":
                self.attn_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
                self.mlp_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
            elif attnres_gate_type == "sigmoid_vector":
                self.attn_res_gate_proj = nn.Linear(n_embd, n_embd, bias=True)
                self.mlp_res_gate_proj = nn.Linear(n_embd, n_embd, bias=True)
            elif attnres_gate_type == "learnable_alpha":
                self.attn_res_alpha = nn.Parameter(torch.tensor(0.0))
                self.mlp_res_alpha = nn.Parameter(torch.tensor(0.0))

            layers_per_block = max(1, (n_layer + attnres_num_blocks - 1) // attnres_num_blocks)
            self.is_block_boundary = ((layer_id + 1) % layers_per_block == 0) or ((layer_id + 1) == n_layer)

        # Shared distance-softening scale for graph Q-shift.
        # NOTE: sigma was removed — the q_shift_graph_multihead op currently uses
        # uniform neighbor weighting.  Distance-weighted Q-shift (exp(-d/sigma²))
        # is a meaningful future extension but is not yet implemented.

        norm_cls = get_norm_layer(norm_layer)
        self.ln1 = norm_cls(n_embd)
        if layer_id == 0:
            self.ln0 = norm_cls(n_embd)

        self.spatial_mixer = SpatialMixer(
            n_embd, n_head, n_layer, layer_id,
            drop_prob=drop_prob, init_values=init_values,
            with_cls_token=with_cls_token,
            norm_layer=norm_layer,
            num_prepend_tokens=num_prepend_tokens,
            use_cpp=use_cpp,
        )
        self.channel_mix = ChannelMix(
            n_embd, drop_prob=drop_prob, init_values=init_values,
            norm_layer=norm_layer, act_layer=act_layer,
            num_prepend_tokens=num_prepend_tokens,
        )
        self._init_weights()

    def _apply_gate(self, partial_block: torch.Tensor, h_attn: torch.Tensor, sublayer: str) -> torch.Tensor:
        if sublayer == "attn":
            return apply_attnres_gate(
                partial_block, h_attn, self.attnres_gate_type,
                gate_logit=getattr(self, "attn_res_gate_logit", None),
                gate_proj=getattr(self, "attn_res_gate_proj", None),
                alpha=getattr(self, "attn_res_alpha", None),
            )
        else:
            return apply_attnres_gate(
                partial_block, h_attn, self.attnres_gate_type,
                gate_logit=getattr(self, "mlp_res_gate_logit", None),
                gate_proj=getattr(self, "mlp_res_gate_proj", None),
                alpha=getattr(self, "mlp_res_alpha", None),
            )

    def _init_weights(self):
        with torch.no_grad():
            if self.n_layer <= 1:
                ratio_0_to_1, ratio_1_to_almost0 = 0.0, 0.5
            else:
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

            idx = torch.arange(self.n_embd, dtype=torch.float) / max(self.n_embd - 1, 1)
            ddd = idx.view(1, 1, self.n_embd)

            # Delegate to sub-module initializers
            self.spatial_mixer.dynamic_offset.init_weights(ratio_1_to_almost0, ddd)
            self.spatial_mixer.scan.init_weights(ratio_0_to_1, ratio_1_to_almost0, ddd)

            if self.use_attnres:
                init_attnres_params(self, self.attnres_gate_type, self.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        dists: Optional[torch.Tensor] = None,
        v_first_fwd: Optional[torch.Tensor] = None,
        v_first_bwd: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attnres_history: Optional[list[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.layer_id == 0:
            x = self.ln0(x)  # always applied for layer 0, regardless of attnres path

        if self.use_attnres and attnres_history is not None and len(attnres_history) > 0:
            h_attn = block_attn_res(
                attnres_history, x,
                self.attn_res_proj, self.attn_res_norm, self.attn_res_bias
            )
            h = self._apply_gate(x, h_attn, "attn")
            xn = self.ln1(h)
        else:
            xn = self.ln1(x)

        x, vf_fwd, vf_bwd = self.spatial_mixer(
            x, xn, neighbors, dists,
            v_first_fwd=v_first_fwd, v_first_bwd=v_first_bwd,
            mask=mask,
        )

        if self.use_attnres and attnres_history is not None:
            if self.attnres_mode == "full":
                attnres_history.append(x)

        if self.use_attnres and attnres_history is not None and len(attnres_history) > 0:
            h_mlp_attn = block_attn_res(
                attnres_history, x,
                self.mlp_res_proj, self.mlp_res_norm, self.mlp_res_bias
            )
            h = self._apply_gate(x, h_mlp_attn, "mlp")
            x = self.channel_mix(
                x, neighbors, dists,
                head_dim=self.head_size,
                with_cls_token=self.with_cls_token,
                h=h,
            )
        else:
            x = self.channel_mix(
                x, neighbors, dists,
                head_dim=self.head_size,
                with_cls_token=self.with_cls_token,
            )

        if self.use_attnres and attnres_history is not None:
            if self.attnres_mode == "full" or self.is_block_boundary:
                attnres_history.append(x)

        return x, vf_fwd, vf_bwd


# =====================================================================
# Tokenization + Embedding
# =====================================================================


class SuperpixelEmbedding(nn.Module):
    """
    Converts 2D image to superpixel tokens.
    Preserves raw input channels, adds conv features, and explicitly injects
    normalized centroids and areas to maintain spatial/size priors.
    """

    def __init__(self, in_chans: int, embed_dims: int, num_superpixels: int, mode: str = "soft", norm_layer: str = "layernorm"):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.num_superpixels = num_superpixels
        self.mode = mode

        # Calculate dynamic conv channels: embed_dims - in_chans - 3 (for x, y, area)
        self.conv_chans = embed_dims - in_chans - 3
        if self.conv_chans <= 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be strictly greater than "
                f"in_chans + 3 ({in_chans + 3})."
            )

        self.conv = nn.Conv2d(in_chans, self.conv_chans, kernel_size=3, padding=1)
        self.raw_norm = nn.LayerNorm(in_chans)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.norm = get_norm_layer(norm_layer)(embed_dims)
        self.num_freqs = 8
        self.pos_mlp = nn.Sequential(
            nn.Linear(4 * self.num_freqs, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        # Lazy coordinate grid cache: keyed by (H, W, device_str, dtype_str).
        # Avoids re-running torch.meshgrid on every forward when image size is fixed.
        self._coord_cache: dict = {}

    def forward(self, x: torch.Tensor, sp_map: torch.Tensor, K: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C_in, H, W = x.shape
        assert C_in == self.in_chans, f"Expected {self.in_chans} channels, got {C_in}"

        # 1. Prepare mask and pooling weights
        if self.mode == "hard":
            if K is None:
                K = self.num_superpixels
            mask = F.one_hot(sp_map.long(), num_classes=K).permute(0, 3, 1, 2).float()
        else:
            mask = sp_map
            K = mask.shape[1]

        # Normalize mask over spatial dimensions to get soft/hard weights
        weights = mask / (mask.sum(dim=(2, 3), keepdim=True) + 1e-6)

        # 2. Pool raw features directly (preserves in_chans) and normalize
        pooled_raw = torch.einsum("bkhw,bchw->bkc", weights, x)
        pooled_raw = self.raw_norm(pooled_raw)

        # 3. Conv and pool (adds conv_chans)
        x_conv = self.conv(x)
        pooled_conv = torch.einsum("bkhw,bchw->bkc", weights, x_conv)

        # Compute centroids and areas dynamically from the mask
        areas = mask.sum(dim=(2, 3)) / (H * W)
        areas_norm = 2.0 * areas - 1.0

        # Lazy coord grid: recompute only when (H, W, device, dtype) change.
        cache_key = (H, W, str(x.device), str(x.dtype))
        if cache_key not in self._coord_cache:
            grid_y = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
            grid_x = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            self._coord_cache[cache_key] = torch.stack([gx, gy], dim=-1)
        coords = self._coord_cache[cache_key]

        centroids = torch.einsum("bkhw,hwc->bkc", weights, coords)

        # 5. Concatenate: centroid_x, centroid_y, area, superpixel_features
        final_tokens = torch.cat(
            [
                centroids,
                areas_norm.unsqueeze(-1),
                pooled_raw,
                pooled_conv,
            ],
            dim=-1,
        )

        freqs = 2.0 ** torch.arange(
            self.num_freqs, device=centroids.device, dtype=centroids.dtype
        )
        f = centroids.unsqueeze(-1) * freqs
        fourier = torch.cat([f.sin(), f.cos()], dim=-1).flatten(2)
        pos = self.pos_mlp(fourier)

        return self.norm(self.proj(final_tokens) + pos), centroids


# =====================================================================
# SuperpixelTokenizer — Vision-to-tokens pipeline
# =====================================================================


class SuperpixelTokenizer(nn.Module):
    """End-to-end vision-to-token pipeline: diffSLIC → embedding → graph → reorder.

    Takes a raw image tensor and returns Hilbert-ordered superpixel tokens
    with their KNN graph, ready for the RWKV-7 backbone.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dims: int,
        num_superpixels: int,
        compactness: float,
        diff_slic_iters: int = 5,
        mode: str = "soft",
        use_cpp: bool = False,
        norm_layer: str = "layernorm",
        spixel_backend: str = "diff_slic",
        downsample_factor: float = 1.0,
        knn_k: int = 4,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.mode = mode
        self.spixel_backend = spixel_backend
        self.diff_slic_iters = diff_slic_iters
        self.downsample_factor = float(downsample_factor)
        self.knn_k = knn_k
        if self.downsample_factor < 1.0:
            raise ValueError(
                f"downsample_factor must be >= 1.0, got {self.downsample_factor}"
            )

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

        self.patch_embed = SuperpixelEmbedding(
            in_chans, embed_dims, num_superpixels, mode=mode, norm_layer=norm_layer,
        )

    def _downsample_for_tokenizer(self, x: torch.Tensor):
        factor = self.downsample_factor
        if factor <= 1.0:
            return x, None
        new_h = max(1, int(round(x.shape[-2] / factor)))
        new_w = max(1, int(round(x.shape[-1] / factor)))
        if (new_h, new_w) == x.shape[-2:]:
            return x, None
        x_down = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x_down, (new_h, new_w)

    @staticmethod
    def _interpolate_mask(mask: Optional[torch.Tensor], orig_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        if mask is None:
            return mask
        if mask.shape[-2:] == orig_hw:
            return mask
        return F.interpolate(
            mask,
            size=orig_hw,
            mode="bilinear" if mask.dtype.is_floating_point else "nearest",
            align_corners=False,
        )

    @staticmethod
    def _interpolate_labels(labels: Optional[torch.Tensor], orig_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        if labels is None:
            return labels
        if labels.shape[-2:] == orig_hw:
            return labels
        return F.interpolate(
            labels.unsqueeze(1).float(), size=orig_hw, mode="nearest"
        ).squeeze(1).long()

    def forward(
        self,
        x: torch.Tensor,
        num_superpixels: Optional[int] = None,
        spixel_size: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Tokenize image and return ordered tokens, graph, and spatial metadata.

        Returns a dict with keys:
          tokens         — (B, N, D) Hilbert-ordered token sequence
          neighbors      — (B, N, K) KNN indices in Hilbert order
          neighbor_dists — (B, N, K) distances
          inv_order      — (B, N) inverse permutation for un-reordering
          batch_idx      — (B,) range for batch indexing
          global_soft_mask — (B, K, H, W) or None
          global_labels  — (B, H, W) or None
          h_s, w_s       — superpixel grid dimensions
        """
        B, _, H, W = x.shape
        x_slic = x
        orig_hw = (H, W)
        if self.downsample_factor > 1.0:
            new_h = max(1, int(round(H / self.downsample_factor)))
            new_w = max(1, int(round(W / self.downsample_factor)))
            if (new_h, new_w) != (H, W):
                x_slic = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Determine superpixel count
        if num_superpixels is None:
            if spixel_size is not None:
                n_sp = int(round((H * W) / (spixel_size**2)))
            else:
                n_sp = self.num_superpixels
        else:
            n_sp = num_superpixels


        # Determine grid shape from the downsampled spatial size
        h_src, w_src = x_slic.shape[-2:]
        height_s = max(1, int(math.sqrt(n_sp * h_src / w_src)))
        width_s = max(1, int(math.sqrt(n_sp * w_src / h_src)))
        h_s, w_s = height_s, width_s
        K = h_s * w_s

        # Tokenization
        global_soft_mask: Optional[torch.Tensor] = None
        global_labels: Optional[torch.Tensor] = None

        if self.spixel_backend == "diff_slic":
            x_for_slic = torch.cat([x_slic[:, :-2], x_slic[:, -2:] * self.compactness], dim=1)
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
                    torch.arange(K, dtype=torch.float, device=x_slic.device)
                    .reshape(1, 1, h_s, w_s)
                    .expand(B, -1, -1, -1)
                )
                global_labels = (
                    spixel_upsampling(label_grid, hard_assign, candidate_radius=radius)
                    .squeeze(1)
                    .long()
                )
                if self.downsample_factor > 1.0 and global_labels.shape[-2:] != orig_hw:
                    global_labels = self._interpolate_labels(global_labels, orig_hw)
                tokens, centroids = self.patch_embed(x, global_labels, K=K)
            else:
                spixel_ids = (
                    torch.arange(K, device=x_slic.device)
                    .reshape(1, K, 1, 1)
                    .expand(B, -1, h_s, w_s)
                    .float()
                )
                global_soft_mask = spixel_upsampling(
                    spixel_ids, p2s_assign, candidate_radius=radius
                )
                if self.downsample_factor > 1.0 and global_soft_mask.shape[-2:] != orig_hw:
                    global_soft_mask = self._interpolate_mask(global_soft_mask, orig_hw)
                tokens, centroids = self.patch_embed(x, global_soft_mask)
        elif self.spixel_backend == "lnsnet":
            x_lnsnet = torch.cat([x_slic[:, :3], x_slic[:, 4:6]], dim=1)
            x_lnsnet = (x_lnsnet - x_lnsnet.mean(dim=(2, 3), keepdim=True)) / (x_lnsnet.std(dim=(2, 3), keepdim=True) + 1e-6)

            assert self.lnsnet_model is not None
            cx, cy, f, probs = self.lnsnet_model(x_lnsnet)

            S = h_src * w_src / max(n_sp, 1)
            sp_h = max(1, int(math.floor(math.sqrt(S) / (w_src / float(h_src)))))
            sp_w = max(1, int(math.floor(S / math.floor(sp_h))))
            h_s = int(math.ceil(h_src / sp_h))
            w_s = int(math.ceil(w_src / sp_w))
            K = h_s * w_s

            p2s_assign = lnsnet_assignment(f, x_lnsnet, cx, cy) # shape (B, K, h_src, w_src)

            if self.mode == "hard":
                global_labels = p2s_assign.argmax(dim=1)
                if self.downsample_factor > 1.0 and global_labels.shape[-2:] != orig_hw:
                    global_labels = self._interpolate_labels(global_labels, orig_hw)
                tokens, centroids = self.patch_embed(x, global_labels, K=K)
            else:
                global_soft_mask = p2s_assign
                if self.downsample_factor > 1.0 and global_soft_mask.shape[-2:] != orig_hw:
                    global_soft_mask = self._interpolate_mask(global_soft_mask, orig_hw)
                tokens, centroids = self.patch_embed(x, global_soft_mask)
        elif self.spixel_backend == "grid":
            grid_y = torch.arange(h_src, device=x_slic.device) * h_s // h_src
            grid_x = torch.arange(w_src, device=x_slic.device) * w_s // w_src
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            global_labels = (gy * w_s + gx).unsqueeze(0).expand(B, -1, -1).long()

            if self.mode == "hard":
                if self.downsample_factor > 1.0 and global_labels.shape[-2:] != orig_hw:
                    global_labels = self._interpolate_labels(global_labels, orig_hw)
                tokens, centroids = self.patch_embed(x, global_labels, K=K)
            else:
                global_soft_mask = F.one_hot(global_labels, num_classes=K).permute(0, 3, 1, 2).float()
                if self.downsample_factor > 1.0 and global_soft_mask.shape[-2:] != orig_hw:
                    global_soft_mask = self._interpolate_mask(global_soft_mask, orig_hw)
                tokens, centroids = self.patch_embed(x, global_soft_mask)
        elif self.spixel_backend in ("slic", "slico"):
            import numpy as np
            import skimage.segmentation as seg
            labels_list = []
            comp = self.compactness * 20.0
            slic_zero = (self.spixel_backend == "slico")
            for i in range(B):
                img_np = x_slic[i, :3].permute(1, 2, 0).detach().cpu().numpy()
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
                labels_list.append(torch.from_numpy(lbls).to(device=x_slic.device, dtype=torch.long))
            global_labels = torch.stack(labels_list, dim=0)

            if self.mode == "hard":
                if self.downsample_factor > 1.0 and global_labels.shape[-2:] != orig_hw:
                    global_labels = self._interpolate_labels(global_labels, orig_hw)
                tokens, centroids = self.patch_embed(x, global_labels, K=K)
            else:
                global_soft_mask = F.one_hot(global_labels, num_classes=K).permute(0, 3, 1, 2).float()
                if self.downsample_factor > 1.0 and global_soft_mask.shape[-2:] != orig_hw:
                    global_soft_mask = self._interpolate_mask(global_soft_mask, orig_hw)
                tokens, centroids = self.patch_embed(x, global_soft_mask)
        else:
            raise ValueError(f"Unknown spixel_backend: {self.spixel_backend}")

        # KNN graph + Hilbert reorder
        neighbors, neighbor_dists = build_knn_graph(centroids.detach(), k=self.knn_k)
        coords_int = ((centroids + 1.0) * 4095).long().clamp(0, 8191)
        order = hilbert_sort_batched(coords_int)
        inv_order = torch.argsort(order, dim=1)
        batch_idx = torch.arange(B, device=x.device)
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
# Vision_RWKV7 (Backbone) — refactored
# =====================================================================


class Vision_RWKV7(nn.Module):
    """Vision-RWKV-7 backbone with superpixel tokenization and graph Q-shift.

    Composes SuperpixelTokenizer → Hilbert sort → RWKV-7 blocks → output projection.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dims: int = 192,
        num_heads: Optional[int] = None,
        depth: int = 12,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 0.0,
        final_norm: bool = True,
        out_indices: Sequence[int] = (-1,),
        with_cls_token: bool = False,
        output_cls_token: bool = False,
        register_tokens: int = 0,  # DINOv2-style register tokens
        scatter_output: bool = False,
        num_superpixels: int = 256,
        spixel_size: Optional[int] = None,
        diff_slic_iters: int = 5,
        compactness: float = 0.5,
        use_cpp: bool = False,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        spixel_backend: str = "diff_slic",
        use_attnres: bool = False,
        attnres_mode: str = "block",
        attnres_gate_type: str = "bias",
        attnres_num_blocks: int = 8,
        attnres_recency_bias_init: float = 10.0,
        downsample_factor: float = 1.0,
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
        self.compactness = compactness
        self.in_chans = in_chans
        self.num_superpixels = num_superpixels
        self.spixel_size = spixel_size
        self.spixel_backend = spixel_backend
        self.downsample_factor = downsample_factor
        self.knn_k = knn_k

        # Attention Residuals configuration
        self.use_attnres = use_attnres
        self.attnres_mode = attnres_mode
        self.attnres_gate_type = attnres_gate_type
        self.attnres_num_blocks = attnres_num_blocks
        self.attnres_recency_bias_init = attnres_recency_bias_init
        self.last_attnres_history = None
        self.last_project_fn = None

        # Dynamic calculation of num_heads if not provided
        if num_heads is None:
            assert (
                embed_dims % HEAD_SIZE == 0
            ), f"embed_dims={embed_dims} must be divisible by HEAD_SIZE={HEAD_SIZE} if num_heads is not provided"
            num_heads = embed_dims // HEAD_SIZE

        # Vision-to-token pipeline
        self.tokenizer = SuperpixelTokenizer(
                    in_chans=in_chans,
                    embed_dims=embed_dims,
                    num_superpixels=num_superpixels,
                    compactness=compactness,
                    diff_slic_iters=diff_slic_iters,
                    mode="soft",
                    use_cpp=use_cpp,
                    norm_layer=norm_layer,
                    spixel_backend=spixel_backend,
                    downsample_factor=downsample_factor,
                    knn_k=knn_k,
                )
        # Keep patch_embed as a public alias for backward compat (tests access .patch_embed.mode)
        self.patch_embed = self.tokenizer.patch_embed

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

        self.out_indices = normalize_out_indices(out_indices, depth)

        self._init_weights()

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
                    use_cpp=use_cpp,
                )
                for i in range(depth)
            ]
        )

    def _init_weights(self):
        zero_init_backbone_tokens(self)

    def _project_output(
        self,
        patch_tokens: torch.Tensor,
        inv_order: torch.Tensor,
        batch_idx: torch.Tensor,
        global_soft_mask: Optional[torch.Tensor],
        global_labels: Optional[torch.Tensor],
        H: int,
        W: int,
        h_s: int,
        w_s: int,
    ) -> torch.Tensor:
        """Reorder tokens back and optionally scatter to pixel grid."""
        patch_tokens = patch_tokens[batch_idx[:, None], inv_order]

        if self.scatter_output:
            if self.tokenizer.mode == "soft":
                assert global_soft_mask is not None
                feat = torch.einsum(
                    "bkd,bkhw->bhwd", patch_tokens, global_soft_mask
                )
                feat = feat.permute(0, 3, 1, 2)
            else:
                assert global_labels is not None
                feat = patch_tokens.gather(
                    1,
                    global_labels.view(-1, H * W, 1).expand(
                        -1, -1, self.embed_dims
                    ),
                )
                feat = feat.view(-1, H, W, self.embed_dims).permute(0, 3, 1, 2)
        else:
            feat = patch_tokens.view(-1, h_s, w_s, self.embed_dims).permute(
                0, 3, 1, 2
            )
        return feat

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

        Returns:
            tuple of (B, embed_dims, h_s, w_s) or (B, embed_dims, H, W)
            feature maps, one per out_indices entry.
        """
        B, C, H, W = x.shape

        assert C == self.in_chans, (
            f"Model initialized with in_chans={self.in_chans}, "
            f"but received input with C={C}. "
            "Please handle data preparation externally."
        )

        # ---- Tokenization ----
        out = self.tokenizer(x, num_superpixels=num_superpixels, spixel_size=self.spixel_size, mask=mask)
        tokens = out["tokens"]
        neighbors = out["neighbors"]
        neighbor_dists = out["neighbor_dists"]
        inv_order = out["inv_order"]
        batch_idx = out["batch_idx"]
        global_soft_mask = out["global_soft_mask"]
        global_labels = out["global_labels"]
        h_s, w_s = out["h_s"], out["w_s"]
        sorted_mask = out["mask"]

        # CLS token - appended to sequence
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat((tokens, cls_tokens), dim=1)

        # Register tokens - prepended to sequence (DINOv2-style)
        if self.register_tokens > 0:
            assert self.reg_token is not None
            reg_tokens = self.reg_token.expand(B, -1, -1)
            tokens = torch.cat((reg_tokens, tokens), dim=1)

        # ---- RWKV-7 Blocks ----
        outs: list = []
        vf_fwd, vf_bwd = None, None
        attnres_history = [tokens] if self.use_attnres else None

        def get_patches(toks):
            if self.with_cls_token:
                toks = toks[:, :-1]
            if self.register_tokens > 0:
                toks = toks[:, self.register_tokens:]
            return toks

        for i, block in enumerate(self.blocks):
            if self.use_attnres:
                tokens, vff, vfb = block(
                    tokens, neighbors, neighbor_dists, vf_fwd, vf_bwd, mask=sorted_mask,
                    attnres_history=attnres_history
                )
            else:
                tokens, vff, vfb = block(
                    tokens, neighbors, neighbor_dists, vf_fwd, vf_bwd, mask=sorted_mask
                )
            if i == 0:
                vf_fwd, vf_bwd = vff, vfb
            if i == len(self.blocks) - 1 and self.final_norm:
                tokens = self.ln1(tokens)

            if i in self.out_indices:
                # Extract patch tokens: exclude CLS (last) and register tokens (first)
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
                    patch_tokens, inv_order, batch_idx,
                    global_soft_mask, global_labels,
                    H, W, h_s, w_s,
                )

                if self.output_cls_token and cls_out is not None:
                    outs.append((feat, cls_out))
                else:
                    outs.append(feat)

        if self.use_attnres:
            # Detach before pinning — this history is for visualization only.
            # Keeping live computation-graph tensors on the model leaks memory
            # across forward passes (prevents graph free after backward).
            self.last_attnres_history = [t.detach() for t in attnres_history] if attnres_history else None
            self.last_attnres_history_patches = [get_patches(t) for t in self.last_attnres_history] if self.last_attnres_history is not None else None
            _inv_order = inv_order.detach()
            _batch_idx = batch_idx.detach()
            _global_soft_mask = global_soft_mask.detach() if global_soft_mask is not None else None
            _global_labels = global_labels.detach() if global_labels is not None else None
            self.last_project_fn = lambda patch_tokens: self._project_output(
                patch_tokens, _inv_order, _batch_idx,
                _global_soft_mask, _global_labels,
                H, W, h_s, w_s,
            )

        return tuple(outs)


# =====================================================================
# Model Builders
# =====================================================================


class ClassificationHead(nn.Module):
    """Classification head for Vision-RWKV-7 dense feature outputs.

    Takes the output feature map (B, embed_dims, H, W) from the backbone,
    applies global average pooling, LayerNorm, and a linear projection to
    class logits. Designed as a separate module — not integrated into the
    backbone — so the backbone remains usable for dense prediction tasks.

    Args:
        embed_dims: Token embedding dimension.
        num_classes: Number of output classes.
        norm_layer: Normalization layer name.
        use_attnres: Enable attention residual components for block-level
            cross-attention over backbone features.
    """

    def __init__(
        self,
        embed_dims: int,
        num_classes: int,
        norm_layer: str = "layernorm",
        use_attnres: bool = False,
    ):
        super().__init__()
        self.norm = get_norm_layer(norm_layer)(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes)
        self.use_attnres = use_attnres

        if use_attnres:
            self.out_res_proj = nn.Linear(embed_dims, 1, bias=False)
            self.out_res_norm = get_norm_layer(norm_layer)(embed_dims)
            self.out_res_bias = nn.Parameter(torch.tensor(10.0))
            nn.init.zeros_(self.out_res_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        attnres_history: Optional[list[torch.Tensor]] = None,
        project_fn=None,
    ) -> torch.Tensor:
        # x: (B, embed_dims, h, w) from backbone output tuple entry
        if self.use_attnres and attnres_history is not None and len(attnres_history) > 0 and project_fn is not None:
            # V: (L, B, SeqLen, D)
            V = torch.stack(attnres_history, dim=0)
            K = self.out_res_norm(V)
            query = self.out_res_proj.weight.view(-1)
            logits = torch.einsum("d, l b s d -> l b s", query, K)
            logits[-1] = logits[-1] + self.out_res_bias
            weights = logits.softmax(dim=0)
            h = torch.einsum("l b s, l b s d -> b s d", weights, V)
            feat = project_fn(h)
            x = feat.mean(dim=[-2, -1])
        else:
            x = x.mean(dim=[-2, -1])  # global average pool over spatial dims
        x = self.norm(x)
        return self.head(x)


def create_vision_rwkv7(
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
    use_jit: bool = False,
    downsample_factor: float = 1.0,
    knn_k: int = 4,
) -> torch.nn.Module:
    """
    Creates a Vision_RWKV7 model enforced to 6-channel input (L, a, b, alpha, x, y).
    This is the standard entry point for this repository.
    """
    from spixrwkv7.jit import maybe_compile

    _model: torch.nn.Module = Vision_RWKV7(
        img_size=img_size,
        in_chans=6,  # Enforced externally to the class
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
        downsample_factor=downsample_factor,
        knn_k=knn_k,
    )
    return maybe_compile(_model, use_jit=use_jit)
