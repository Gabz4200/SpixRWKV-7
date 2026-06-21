# Vision-RWKV-7: RWKV-7 vision backbone with Superpixel Tokenization (diffSLIC),
# Graph-Based Q-Shift, bidirectional scanning, gated fusion, and multi-scale output.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple

from hilbertcurve.hilbertcurve import HilbertCurve

from spixrwkv7.data.diff_slic import DiffSLIC, spixel_upsampling
from spixrwkv7.layers.graph import build_knn_graph, q_shift_graph_multihead, HEAD_SIZE
from spixrwkv7.layers.drop import DropPath

TIME_MIX_EXTRA_DIM = 32
_HILBERT_BITS = 13
_HILBERT_CURVE = HilbertCurve(_HILBERT_BITS, 2)


def hilbert_sort_batched(coords_int: torch.Tensor) -> torch.Tensor:
    """Sort token indices by 2-D Hilbert order for each batch item."""
    B, N, _ = coords_int.shape
    coords = coords_int.detach().to(device="cpu", dtype=torch.long).reshape(-1, 2)
    distances = torch.tensor(
        [_HILBERT_CURVE.distance_from_point(point.tolist()) for point in coords],
        dtype=torch.long,
        device=coords_int.device,
    ).view(B, N)
    return torch.argsort(distances, dim=1)


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

    def __init__(self, n_embd: int, n_head: int, layer_id: int, n_layer: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
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

        # Linear projections
        self.att_receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.att_key = nn.Linear(n_embd, n_embd, bias=False)
        self.att_value = nn.Linear(n_embd, n_embd, bias=False)
        self.att_output = nn.Linear(n_embd, n_embd, bias=False)
        self.att_group_norm = nn.GroupNorm(
            n_head, n_embd, eps=self.n_head * 1e-5, affine=True
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

        state = torch.zeros(B, Hd, S, S, device=dev)
        state_time = torch.zeros(B, D, device=dev)

        tm = self.time_mix
        sw, sk, sv, sr, sg, sa = [
            getattr(tm, f"time_maa_{m}").reshape(-1)
            for m in ["w", "k", "v", "r", "g", "a"]
        ]
        x0, x1, x2, x3, x4, x5 = self.x.unbind(dim=0)

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

            # BONUS TERM: Uses kt (replacement key) per RWKV-7 Eq. 20
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


class _DynamicOffset(nn.Module):
    """Input-dependent dynamic offset computation for time-mixing."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.time_maa_x = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, TIME_MIX_EXTRA_DIM * 6))
        self.time_maa_w2 = nn.Parameter(torch.zeros(6, TIME_MIX_EXTRA_DIM, n_embd))

    def forward(
        self,
        xn: torch.Tensor,
        xx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute input-dependent mixing offsets dm from normalized input and shift delta.

        Returns dm of shape (6, B, N, D) — one offset per mixing path.
        """
        B, N, D = xn.shape
        x_base = xn + xx * self.time_maa_x
        x_dyn = torch.tanh(x_base @ self.time_maa_w1)
        x_dyn = x_dyn.view(B * N, 6, -1).transpose(0, 1)
        x_dyn = torch.bmm(x_dyn, self.time_maa_w2)
        dm = x_dyn.view(6, B, N, D)
        return dm

    def init_weights(self, ratio_1_to_almost0: float, ddd: torch.Tensor):
        with torch.no_grad():
            def fancy_mix(base_pow):
                return 1.0 - torch.pow(ddd, base_pow)
            self.time_maa_x.copy_(fancy_mix(ratio_1_to_almost0))
            self.time_maa_w1.uniform_(-1e-4, 1e-4)
            self.time_maa_w2.uniform_(-1e-4, 1e-4)


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
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        self.with_cls_token = with_cls_token
        self.layer_id = layer_id
        self.n_layer = n_layer

        self.dynamic_offset = _DynamicOffset(n_embd)
        self.scan = RecurrentScan(n_embd, n_head, layer_id, n_layer)
        self.fusion_gate = nn.Linear(n_embd, n_embd, bias=False)
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
        dists: Optional[torch.Tensor],
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Graph Q-shift followed by dynamic offset computation."""
        assert sigma is not None, "SpatialMixer requires sigma (set by Vision_RWKV7_Block)"
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            dists=dists,
            head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
            sigma=sigma,
        )
        xx = xs - xn
        dm = self.dynamic_offset(xn, xx)
        return xx, dm

    def forward(
        self,
        x: torch.Tensor,
        xn: torch.Tensor,
        neighbors: torch.Tensor,
        sigma: torch.Tensor,
        dists: Optional[torch.Tensor] = None,
        v_first_fwd: Optional[torch.Tensor] = None,
        v_first_bwd: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        xx, dm = self._spatial_prep(xn, neighbors, dists, sigma)
        x_gate = xn + xx * 0.5

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
        self.ffn_ln = norm_cls(n_embd)
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
        sigma: Optional[torch.Tensor] = None,
        head_dim: int = 64,
        with_cls_token: bool = False,
    ) -> torch.Tensor:
        assert sigma is not None, "ChannelMix requires sigma (set by Vision_RWKV7_Block)"
        xn = self.norm(x)
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            dists=dists,
            head_dim=head_dim,
            with_cls_token=with_cls_token,
            sigma=sigma,
        )
        xx = xs - xn
        xk = xn + xx * self.ffn_x_k
        if self.act_layer == "relu2":
            k = F.relu(self.ffn_key(xk)).pow(2)
        elif self.act_layer == "gelu":
            k = F.gelu(self.ffn_key(xk))
        elif self.act_layer == "silu":
            k = F.silu(self.ffn_key(xk))
        elif self.act_layer == "swiglu":
            gate, val = self.ffn_key(xk).chunk(2, dim=-1)
            k = F.silu(gate) * val
        else:
            raise ValueError(f"Unknown activation layer: {self.act_layer}")
        ffn_out = self.ffn_value(k)
        ffn_out = self.ffn_ln(ffn_out)
        if self.gamma2 is not None:
            ffn_out = self.gamma2 * ffn_out  # LayerScale (CaiT et al., 2021)
        return x + self.drop_path(ffn_out)


# =====================================================================
# Vision_RWKV7_Block (refactored)
# =====================================================================


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
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        self.with_cls_token = with_cls_token

        # Shared distance-softening scale for graph Q-shift.
        self.sigma = nn.Parameter(torch.full((), 0.1))

        norm_cls = get_norm_layer(norm_layer)
        self.ln1 = norm_cls(n_embd)
        if layer_id == 0:
            self.ln0 = norm_cls(n_embd)

        self.spatial_mixer = SpatialMixer(
            n_embd, n_head, n_layer, layer_id,
            drop_prob=drop_prob, init_values=init_values,
            with_cls_token=with_cls_token,
            norm_layer=norm_layer,
        )
        self.channel_mix = ChannelMix(
            n_embd, drop_prob=drop_prob, init_values=init_values,
            norm_layer=norm_layer, act_layer=act_layer,
        )
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

            # Delegate to sub-module initializers
            self.spatial_mixer.dynamic_offset.init_weights(ratio_1_to_almost0, ddd)
            self.spatial_mixer.scan.init_weights(ratio_0_to_1, ratio_1_to_almost0, ddd)

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        dists: Optional[torch.Tensor] = None,
        v_first_fwd: Optional[torch.Tensor] = None,
        v_first_bwd: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.layer_id == 0:
            x = self.ln0(x)
        xn = self.ln1(x)

        x, vf_fwd, vf_bwd = self.spatial_mixer(
            x, xn, neighbors, self.sigma, dists,
            v_first_fwd=v_first_fwd, v_first_bwd=v_first_bwd,
            mask=mask,
        )

        x = self.channel_mix(
            x, neighbors, dists,
            sigma=self.sigma, head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
        )

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
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.norm = get_norm_layer(norm_layer)(embed_dims)
        self.num_freqs = 8
        self.pos_mlp = nn.Sequential(
            nn.Linear(4 * self.num_freqs, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, x: torch.Tensor, sp_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C_in, H, W = x.shape
        assert C_in == self.in_chans, f"Expected {self.in_chans} channels, got {C_in}"

        # 1. Prepare mask and pooling weights
        if self.mode == "hard":
            K = int(sp_map.max().item()) + 1
            mask = F.one_hot(sp_map.long(), num_classes=K).permute(0, 3, 1, 2).float()
        else:
            mask = sp_map
            K = mask.shape[1]

        # Normalize mask over spatial dimensions to get soft/hard weights
        weights = mask / (mask.sum(dim=(2, 3), keepdim=True) + 1e-6)

        # 2. Pool raw features directly (preserves in_chans)
        pooled_raw = torch.einsum("bkhw,bchw->bkc", weights, x)

        # 3. Conv and pool (adds conv_chans)
        x_conv = self.conv(x)
        pooled_conv = torch.einsum("bkhw,bchw->bkc", weights, x_conv)

        # 4. Compute Centroids and Areas dynamically from the mask
        areas = mask.sum(dim=(2, 3)) / (H * W)
        areas_norm = 2.0 * areas - 1.0

        grid_y = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1)

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
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.mode = mode

        self.diff_slic = DiffSLIC(
            n_spixels=num_superpixels,
            n_iter=diff_slic_iters,
            tau=0.01,
            candidate_radius=1,
            stable=True,
            use_cpp=use_cpp,
        )
        self.patch_embed = SuperpixelEmbedding(
            in_chans, embed_dims, num_superpixels, mode=mode, norm_layer=norm_layer,
        )

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

        # Determine superpixel count
        if num_superpixels is None:
            if spixel_size is not None:
                n_sp = int(round((H * W) / (spixel_size**2)))
            else:
                n_sp = self.num_superpixels
        else:
            n_sp = num_superpixels

        # diffSLIC
        x_for_slic = torch.cat([x[:, :-2], x[:, -2:] * self.compactness], dim=1)
        clst_feats, p2s_assign, _ = self.diff_slic(x_for_slic, n_spixels=n_sp)
        h_s, w_s = clst_feats.shape[-2:]
        K = h_s * w_s
        radius = self.diff_slic.candidate_radius

        # Tokenization
        global_soft_mask: Optional[torch.Tensor] = None
        global_labels: Optional[torch.Tensor] = None

        if self.mode == "hard":
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
            tokens, centroids = self.patch_embed(x, global_labels)
        else:
            spixel_ids = (
                torch.arange(K, device=x.device)
                .reshape(1, K, 1, 1)
                .expand(B, -1, h_s, w_s)
                .float()
            )
            global_soft_mask = spixel_upsampling(
                spixel_ids, p2s_assign, candidate_radius=radius
            )
            tokens, centroids = self.patch_embed(x, global_soft_mask)

        # KNN graph + Hilbert reorder
        neighbors, neighbor_dists = build_knn_graph(centroids.detach(), k=4)
        coords_int = ((centroids + 1.0) * 4096).long().clamp(0, 8191)
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
        num_superpixels: int = 196,
        spixel_size: Optional[int] = None,
        diff_slic_iters: int = 5,
        compactness: float = 0.5,
        use_cpp: bool = False,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
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
        )
        # Keep patch_embed as a public alias for backward compat (tests access .patch_embed.mode)
        self.patch_embed = self.tokenizer.patch_embed

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        # Register tokens (DINOv2-style learnable tokens prepended to sequence)
        self.register_tokens = register_tokens
        if register_tokens > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, register_tokens, embed_dims))
        else:
            self.reg_token = None

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

        indices: list[int] = (
            [out_indices] if isinstance(out_indices, int) else list(out_indices)
        )
        for i, idx in enumerate(indices):
            if idx < 0:
                indices[i] = depth + idx
        self.out_indices = sorted(set(i for i in indices if 0 <= i < depth)) or [
            depth - 1
        ]

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
                )
                for i in range(depth)
            ]
        )

    def _init_weights(self):
        with torch.no_grad():
            if self.with_cls_token:
                self.cls_token.zero_()
            if self.reg_token is not None:
                self.reg_token.zero_()

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
            reg_tokens = self.reg_token.expand(B, -1, -1)
            tokens = torch.cat((reg_tokens, tokens), dim=1)

        # ---- RWKV-7 Blocks ----
        outs: list = []
        vf_fwd, vf_bwd = None, None
        for i, block in enumerate(self.blocks):
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
    """

    def __init__(self, embed_dims: int, num_classes: int, norm_layer: str = "layernorm"):
        super().__init__()
        self.norm = get_norm_layer(norm_layer)(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, embed_dims, h, w) from backbone output tuple entry
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
    num_superpixels: int = 196,
    spixel_size: Optional[int] = None,
    diff_slic_iters: int = 5,
    compactness: float = 0.5,
    register_tokens: int = 0,
    norm_layer: str = "layernorm",
    act_layer: str = "relu2",
) -> Vision_RWKV7:
    """
    Creates a Vision_RWKV7 model enforced to 6-channel input (L, a, b, alpha, x, y).
    This is the standard entry point for this repository.
    """
    return Vision_RWKV7(
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
    )
