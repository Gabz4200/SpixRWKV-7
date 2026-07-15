# Common shared components for Vision-RWKV-7 model variants.
# Extracted to eliminate copy-paste duplication across spixrwkv7.py,
# conv_spixrwkv7.py, gnn_spixrwkv7.py, vq_rwkv7.py, and optimized_block.py.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spixrwkv7.layers.drop import DropPath
from spixrwkv7.layers.graph import HEAD_SIZE

TIME_MIX_EXTRA_DIM = 32


def apply_attnres_gate(
    partial_block: torch.Tensor,
    h_attn: torch.Tensor,
    gate_type: str,
    gate_logit: Optional[nn.Parameter] = None,
    gate_proj: Optional[nn.Linear] = None,
    alpha: Optional[nn.Parameter] = None,
) -> torch.Tensor:
    """Apply attention-residual gate (shared across all block variants)."""
    if gate_type == "sigmoid_scalar":
        gate = torch.sigmoid(gate_logit)
        return (1 - gate) * partial_block + gate * h_attn
    elif gate_type == "sigmoid_vector":
        gate = torch.sigmoid(gate_proj(partial_block))
        return (1 - gate) * partial_block + gate * h_attn
    elif gate_type == "learnable_alpha":
        return (1 - alpha) * partial_block + alpha * h_attn
    else:
        return h_attn


def apply_activation(xk: torch.Tensor, act_layer: str, key_proj: nn.Linear) -> torch.Tensor:
    """Apply activation function dispatch (shared between ChannelMix and GNNFeedForward)."""
    if act_layer == "relu2":
        return F.relu(key_proj(xk)).pow(2)
    elif act_layer == "gelu":
        return F.gelu(key_proj(xk))
    elif act_layer == "silu":
        return F.silu(key_proj(xk))
    elif act_layer == "swiglu":
        gate, val = key_proj(xk).chunk(2, dim=-1)
        return F.silu(gate) * val
    else:
        raise ValueError(f"Unknown activation layer: {act_layer}")


def init_backbone_tokens(
    module: nn.Module,
    with_cls_token: bool,
    register_tokens: int,
    embed_dims: int,
):
    """Initialize CLS and register tokens (shared across all backbone variants)."""
    if with_cls_token:
        module.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
    module.register_tokens = register_tokens
    if register_tokens > 0:
        module.reg_token = nn.Parameter(torch.zeros(1, register_tokens, embed_dims))
    else:
        module.reg_token = None


def zero_init_backbone_tokens(module: nn.Module):
    """Zero-initialize CLS and register tokens (identical in all 4 backbones)."""
    with torch.no_grad():
        if getattr(module, "with_cls_token", False) and hasattr(module, "cls_token"):
            module.cls_token.zero_()
        if getattr(module, "reg_token", None) is not None:
            module.reg_token.zero_()


def normalize_out_indices(out_indices, depth: int):
    """Normalize out_indices: handle int, negative indices, deduplicate, sort.

    Returns a sorted list of valid layer indices.
    """
    indices: list[int] = (
        [out_indices] if isinstance(out_indices, int) else list(out_indices)
    )
    for i, idx in enumerate(indices):
        if idx < 0:
            indices[i] = depth + idx
    return sorted(set(i for i in indices if 0 <= i < depth)) or [depth - 1]


def resolve_num_heads(embed_dims: int, num_heads: Optional[int]) -> int:
    """Compute num_heads from embed_dims if not explicitly provided."""
    if num_heads is None:
        assert embed_dims % HEAD_SIZE == 0, (
            f"embed_dims={embed_dims} must be divisible by HEAD_SIZE={HEAD_SIZE}"
        )
        return embed_dims // HEAD_SIZE
    return num_heads


def compute_attnres_config(
    layer_id: int,
    n_layer: int,
    attnres_num_blocks: int,
    use_attnres: bool,
):
    """Compute block boundary flag for attnres history."""
    if not use_attnres:
        return False
    layers_per_block = max(1, (n_layer + attnres_num_blocks - 1) // attnres_num_blocks)
    return ((layer_id + 1) % layers_per_block == 0) or ((layer_id + 1) == n_layer)


def init_attnres_params(module: nn.Module, gate_type: str, n_embd: int):
    """Initialize attention-residual parameters (shared across block variants)."""
    nn.init.zeros_(module.attn_res_proj.weight)
    nn.init.zeros_(module.mlp_res_proj.weight)
    if gate_type == "sigmoid_vector":
        nn.init.zeros_(module.attn_res_gate_proj.weight)
        nn.init.constant_(module.attn_res_gate_proj.bias, -2.0)
        nn.init.zeros_(module.mlp_res_gate_proj.weight)
        nn.init.constant_(module.mlp_res_gate_proj.bias, -2.0)


class DynamicOffset(nn.Module):
    """Input-dependent dynamic offset computation for time-mixing.

    Shared between SpatialMixer (spixrwkv7.py) and OptimizedSpatialMixer
    (optimized_block.py).
    """

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
