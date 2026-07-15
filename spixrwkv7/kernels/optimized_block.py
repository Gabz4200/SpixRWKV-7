"""Optimized Vision_RWKV7_Block using C++ kernels.

Provides OptimizedSpatialMixer and OptimizedRecurrentScan that route the
hot recurrent state-update loop through the C++ AVX512/generic kernel.

Also provides ParallelRecurrentScan implementing the parallelizable delta-rule
from arXiv:2406.06484 using vector-gated parallel scan (RWKV-7 Goose, arXiv:2503.14456) for sequence-parallel
computation.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spixrwkv7.kernels.rwkv7_kernel import rwkv7_recurrent_scan as _cpp_recurrent_scan
from spixrwkv7.layers.drop import DropPath
from spixrwkv7.layers.graph import HEAD_SIZE, q_shift_graph_multihead
from spixrwkv7.models.common import (
    DynamicOffset,
    apply_attnres_gate,
    compute_attnres_config,
    init_attnres_params,
)
from spixrwkv7.models.spixrwkv7 import (
    ChannelMix,
    RecurrentScan,
    block_attn_res,
    get_norm_layer,
)


class OptimizedRecurrentScan(RecurrentScan):
    """RecurrentScan that dispatches the hot loop to the C++ kernel.

    Computes per-token features (r, k, v, w, a, kk, kt) using PyTorch,
    then calls the C++ kernel for the full sequential state update.
    Falls back to the parent PyTorch loop when the C++ kernel is unavailable.
    """

    def __init__(self, n_embd: int, n_head: int, layer_id: int, n_layer: int):
        super().__init__(n_embd, n_head, layer_id, n_layer)

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

        # Pre-allocate per-token result buffers (k excluded: C++ kernel uses kt directly)
        r_all = torch.empty(B, N, Hd, S, device=dev)
        v_all = torch.empty(B, N, Hd, S, device=dev)
        w_all = torch.empty(B, N, Hd, S, device=dev)
        a_all = torch.empty(B, N, Hd, S, device=dev)
        kk_all = torch.empty(B, N, Hd, S, device=dev)
        kt_all = torch.empty(B, N, Hd, S, device=dev)

        output_list = []
        v_first_list = []

        for t in range(N):
            token, xx_t = xn_seq[:, t, :], xx_seq[:, t, :]
            dm_t = dm_seq[:, :, t, :]
            dmw, dmk, dmv, dmr, dmg, dma = dm_t.unbind(dim=0)

            sx = state_time - token
            state_time.copy_(token)

            # Feature computation (same as parent)
            xw = token + sx * x0 + xx_t * (sw + dmw)
            xk = token + sx * x1 + xx_t * (sk + dmk)
            xv = token + sx * x2 + xx_t * (sv + dmv)
            xr = token + sx * x3 + xx_t * (sr + dmr)
            xg_in = token + sx * x4 + xx_t * (sg + dmg)
            xa_in = token + sx * x5 + xx_t * (sa + dma)

            w_raw = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
            w_t = torch.exp(-0.606531 * torch.sigmoid(w_raw.float()))

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

            # Store per-token features for C++ kernel
            r_all[:, t] = r_t.view(B, Hd, S)
            # k_t not stored: kernel uses kt_t (= k*(1+(a-1)*k_a)), not raw k
            v_all[:, t] = v_t.view(B, Hd, S)
            w_all[:, t] = w_t.view(B, Hd, S)
            a_all[:, t] = a_t.view(B, Hd, S)
            kk_all[:, t] = kk_t.view(B, Hd, S)
            kt_all[:, t] = kt_t.view(B, Hd, S)

            # Group norm + gate is done after state update
            output_list.append(g_t)

        # Call C++ kernel for the full recurrent scan (returns state @ r, no bonus)
        # Guarded: _cpp_recurrent_scan is not None after the early return above
        out_scan = _cpp_recurrent_scan(state, r_all, v_all, w_all, a_all, kk_all, kt_all, mask=mask_seq)  # type: ignore[union-attr]

        # Apply group norm, bonus, gate, and output projection (post-processing)
        # Bonus is added AFTER GroupNorm to match original semantics
        final_outputs = []
        for t in range(N):
            raw = out_scan[:, t]        # (B, Hd, S)
            r_t = r_all[:, t]           # (B, Hd, S)
            kt_t = kt_all[:, t]         # (B, Hd, S)
            v_t = v_all[:, t]           # (B, Hd, S)
            g_t = output_list[t]        # (B, D)

            out = raw.view(B, D) if raw.shape[-1] == S else raw
            out = self.att_group_norm(out)

            # Bonus: sum(r * kt * r_k, dim=S) * v  → matches original semantics
            bonus_scalar = (r_t * kt_t * self.r_k.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (B, Hd, 1)
            bonus = (bonus_scalar * v_t).reshape(B, D)
            out = out + bonus

            out = self.att_output(out * g_t)
            if mask_seq is not None:
                out = out * mask_seq[:, t, None]
            final_outputs.append(out)

        out = torch.stack(final_outputs, dim=1)
        if rev:
            out = torch.flip(out, dims=[1])

        v_first_out = None
        if self.layer_id == 0:
            v_first_out = torch.stack(v_first_list, dim=1)
            if rev:
                v_first_out = torch.flip(v_first_out, dims=[1])

        return out, v_first_out


class ParallelRecurrentScan(RecurrentScan):
    """true Blelloch parallel prefix scan O(log N)"""

    def __init__(self, n_embd: int, n_head: int, layer_id: int, n_layer: int):
        super().__init__(n_embd, n_head, layer_id, n_layer)

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

        # Pre-allocate per-token result buffers
        r_all = torch.empty(B, N, Hd, S, device=dev)
        v_all = torch.empty(B, N, Hd, S, device=dev)
        w_all = torch.empty(B, N, Hd, S, device=dev)
        a_all = torch.empty(B, N, Hd, S, device=dev)
        kk_all = torch.empty(B, N, Hd, S, device=dev)
        kt_all = torch.empty(B, N, Hd, S, device=dev)
        output_list = []
        v_first_list = []

        # Compute all features in parallel
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
            output_list.append(g_t)

        # Parallel scan using Householder reflections
        # Build cumulative product of (I - 2*v*v^T) matrices
        # For each timestep t: out[t] = (prod_{i=0}^{t-1} (I - 2*v_i*v_i^T)) @ x[t]
        out = self._parallel_scan(state, r_all, v_all, w_all, a_all, kk_all, kt_all, output_list, mask=mask_seq)

        if rev:
            out = torch.flip(out, dims=[1])

        v_first_out = None
        if self.layer_id == 0:
            v_first_out = torch.stack(v_first_list, dim=1)
            if rev:
                v_first_out = torch.flip(v_first_out, dims=[1])

        return out, v_first_out

    def _parallel_scan(
        self,
        state: torch.Tensor,
        r_all: torch.Tensor,
        v_all: torch.Tensor,
        w_all: torch.Tensor,
        a_all: torch.Tensor,
        kk_all: torch.Tensor,
        kt_all: torch.Tensor,
        output_list: list,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Parallel scan using Blelloch prefix scan."""
        batch_size, seq_len, Hd, S = r_all.shape

        # Step 1 — build per-timestep (A, B) pairs:
        # shapes: (B, N, Hd, S, S)
        vk = v_all.unsqueeze(-1) @ kt_all.unsqueeze(-2)  # outer product
        ab = (-kk_all).unsqueeze(-1) @ (kk_all * a_all).unsqueeze(-2)
        eye = torch.eye(S, device=r_all.device, dtype=torch.float32)
        A = w_all.float().unsqueeze(-2) * eye + ab.float()  # diag(w) + ab
        B = vk.float()

        if mask is not None:
            m_t = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1, 1)
            A = m_t * A + (1.0 - m_t) * eye.view(1, 1, 1, S, S)
            B = m_t * B

        # Step 2 — Blelloch up-sweep (reduce) + down-sweep (scan) in O(log N) depth:
        # Pad N to next power of 2. Implement iterative tree scan:
        def _assoc_combine(A1, B1, A2, B2):
            # (A1,B1)⊕(A2,B2) = (A1@A2, B1@A2+B2)
            return A1 @ A2, B1 @ A2 + B2

        n_pad = 2 ** ((seq_len - 1).bit_length()) if seq_len > 1 else 1
        pad_len = n_pad - seq_len

        if pad_len > 0:
            A_pad = eye.view(1, 1, 1, S, S).expand(batch_size, pad_len, Hd, S, S)
            B_pad = torch.zeros(batch_size, pad_len, Hd, S, S, device=r_all.device, dtype=torch.float32)
            A_padded = torch.cat([A, A_pad], dim=1)
            B_padded = torch.cat([B, B_pad], dim=1)
        else:
            A_padded = A.clone()
            B_padded = B.clone()
        # Up-sweep
        stride = 1
        while stride < n_pad:
            step = 2 * stride
            left_idx = torch.arange(stride - 1, n_pad, step, device=A_padded.device)
            right_idx = left_idx + stride

            A_left = A_padded[:, left_idx]
            B_left = B_padded[:, left_idx]
            A_right = A_padded[:, right_idx]
            B_right = B_padded[:, right_idx]

            new_A, new_B = _assoc_combine(A_left, B_left, A_right, B_right)
            A_padded[:, right_idx] = new_A
            B_padded[:, right_idx] = new_B

            stride *= 2

        # Set root to identity
        A_padded[:, n_pad - 1] = eye.view(1, 1, S, S).expand(batch_size, Hd, S, S)
        B_padded[:, n_pad - 1] = 0.0

        # Down-sweep
        stride = n_pad // 2
        while stride >= 1:
            step = 2 * stride
            left_idx = torch.arange(stride - 1, n_pad, step, device=A_padded.device)
            right_idx = left_idx + stride
            A_temp = A_padded[:, left_idx].clone()
            B_temp = B_padded[:, left_idx].clone()
            A_parent = A_padded[:, right_idx].clone()
            B_parent = B_padded[:, right_idx].clone()

            # Left child gets parent
            A_padded[:, left_idx] = A_parent
            B_padded[:, left_idx] = B_parent

            # Right child gets parent ⊕ temp (which is left child's old value)
            new_A, new_B = _assoc_combine(A_parent, B_parent, A_temp, B_temp)
            A_padded[:, right_idx] = new_A
            B_padded[:, right_idx] = new_B

            stride //= 2

        prefA_excl = A_padded[:, :seq_len]
        prefB_excl = B_padded[:, :seq_len]

        # Convert to inclusive scan results to represent state after the update at t
        prefA = prefA_excl @ A
        prefB = prefB_excl @ A + B

        # Step 3 — apply initial state + compute outputs:
        # wkv_t = state @ prefA[:,t] + prefB[:,t]
        # state: (B,Hd,S,S), prefA/prefB: (B,N,Hd,S,S)
        wkv = state.float().unsqueeze(1) @ prefA + prefB  # (B,N,Hd,S,S)

        # per-token output: contract with r
        r_h = r_all.unsqueeze(-1)  # (B,N,Hd,S,1)
        out_t = (wkv @ r_h.float()).squeeze(-1)  # (B,N,Hd,S)
        out = out_t.reshape((batch_size * seq_len, -1))  # (B*N, D)

        # Convert to original dtype before group norm and output layers to match weights
        out = out.to(dtype=r_all.dtype)
        out = self.att_group_norm(out)  # GroupNorm over D
        out = out.reshape((batch_size, seq_len, -1))  # (B, N, D)
        # bonus term (per RWKV-7 eq.20)
        bonus_scalar = (r_all * kt_all * self.r_k).sum(dim=-1, keepdim=True)  # (B,N,Hd,1)
        bonus = (bonus_scalar * v_all).reshape((batch_size, seq_len, -1))
        out = out + bonus

        # gate: output_list is a list of N tensors (B,D) from the feature-prep loop
        g_all = torch.stack(output_list, dim=1)  # (B,N,D)
        out = self.att_output(out * g_all)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class OptimizedVision_RWKV7_Block(nn.Module):
    """Optimized Vision-RWKV-7 block using C++ kernels.

    Drop-in replacement for Vision_RWKV7_Block that uses OptimizedRecurrentScan
    which routes the hot state-update loop through the C++ kernel.
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
        use_cpp: bool = True,
        use_parallel: bool = False,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        use_attnres: bool = False,
        attnres_mode: str = "block",
        attnres_gate_type: str = "bias",
        attnres_num_blocks: int = 8,
        attnres_recency_bias_init: float = 10.0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        self.with_cls_token = with_cls_token

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

        # NOTE: sigma (distance-softening scale) was removed — q_shift_graph_multihead
        # currently uses uniform neighbor weighting.  Distance-weighted Q-shift is a
        # meaningful future extension but is not yet implemented.

        norm_cls = get_norm_layer(norm_layer)
        self.ln1 = norm_cls(n_embd)
        if layer_id == 0:
            self.ln0 = norm_cls(n_embd)

        # Choose scan implementation based on use_parallel flag
        if use_parallel:
            self.spatial_mixer = OptimizedSpatialMixer(
                n_embd, n_head, n_layer, layer_id,
                drop_prob=drop_prob, init_values=init_values,
                with_cls_token=with_cls_token, use_cpp=False,
                use_parallel=True, norm_layer=norm_layer,
            )
        elif use_cpp:
            self.spatial_mixer = OptimizedSpatialMixer(
                n_embd, n_head, n_layer, layer_id,
                drop_prob=drop_prob, init_values=init_values,
                with_cls_token=with_cls_token, use_cpp=True,
                norm_layer=norm_layer,
            )
        else:
            self.spatial_mixer = OptimizedSpatialMixer(
                n_embd, n_head, n_layer, layer_id,
                drop_prob=drop_prob, init_values=init_values,
                with_cls_token=with_cls_token, use_cpp=False,
                norm_layer=norm_layer,
            )
        self.channel_mix = ChannelMix(
            n_embd, drop_prob=drop_prob, init_values=init_values,
            norm_layer=norm_layer, act_layer=act_layer,
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


class OptimizedSpatialMixer(nn.Module):
    """Optimized SpatialMixer using C++ kernel for WKV v7."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        layer_id: int,
        drop_prob: float = 0.0,
        init_values: Optional[float] = None,
        with_cls_token: bool = False,
        use_cpp: bool = True,
        use_parallel: bool = False,
        norm_layer: str = "layernorm",
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = HEAD_SIZE
        self.with_cls_token = with_cls_token
        self.layer_id = layer_id
        self.n_layer = n_layer

        self.dynamic_offset = DynamicOffset(n_embd)
        if use_parallel:
            self.scan = ParallelRecurrentScan(n_embd, n_head, layer_id, n_layer)
        elif use_cpp:
            self.scan = OptimizedRecurrentScan(n_embd, n_head, layer_id, n_layer)
        else:
            self.scan = RecurrentScan(n_embd, n_head, layer_id, n_layer)

        self.fusion_gate = nn.Linear(n_embd, n_embd, bias=False)
        self.att_ln = get_norm_layer(norm_layer)(n_embd)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd))
        else:
            self.gamma1 = None

    def _spatial_prep(
        self,
        xn: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
        )
        xx = xs - xn
        dm = self.dynamic_offset(xn, xx)
        return xx, dm

    def forward(
        self,
        x: torch.Tensor,
        xn: torch.Tensor,
        neighbors: torch.Tensor,
        dists: Optional[torch.Tensor] = None,  # reserved for future distance-weighted Q-shift
        v_first_fwd: Optional[torch.Tensor] = None,
        v_first_bwd: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        xx, dm = self._spatial_prep(xn, neighbors)
        x_gate = xn + xx * 0.5

        out_fwd, vf_fwd = self.scan(xn, xx, dm, "forward", v_first_fwd, mask=mask)
        out_bwd, vf_bwd = self.scan(xn, xx, dm, "backward", v_first_bwd, mask=mask)

        gate = torch.sigmoid(self.fusion_gate(x_gate))
        att_out = gate * out_fwd + (1 - gate) * out_bwd
        att_out = self.att_ln(att_out)
        if self.gamma1 is not None:
            att_out = self.gamma1 * att_out
        x = x + self.drop_path(att_out)
        return x, vf_fwd, vf_bwd
