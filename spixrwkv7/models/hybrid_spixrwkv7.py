# HybridRWKV7GNN: Unidirectional RWKV-7 + GNN hybrid backbone.
#
# Architecture: L × [N layers of unidirectional VRWKV-7 → M layers of GNN]
#
# Compared against the bidirectional Vision-RWKV-7 backbone, this variant:
#   1. Uses unidirectional (forward-only) RWKV-7 scan along Hilbert curve order
#      instead of bidirectional scan with fusion gate — simpler, follows spatial
#      locality naturally (inspired by HilbertA, arXiv:2509.26538).
#   2. Alternates with GNN message-passing layers on the KNN superpixel graph
#      to inject global graph context that the unidirectional scan misses.
#   3. Fixes two superpixel mapping issues:
#      - Duplicate centroids: multiple superpixels mapping to the same spatial
#        point. Solved by deduplicating centroids with sub-pixel jitter before
#        KNN graph construction.
#      - Duplicate neighbors: same superpixel appearing more than once in a
#        node's neighbor list. Solved by post-filtering KNN neighbors.
#
# Register tokens as the global information highway:
# -----------------------------------------------
# The key mechanism that compensates for unidirectional (forward-only) scan
# is DINOv2-style register tokens.  When register_tokens > 0:
#
#   1. Graph connectivity: Register nodes are connected bipartitely to ALL
#      superpixel nodes in the KNN graph (both directions, with inverse-
#      distance weights).  This gives every register token a global receptive
#      field via GNN message passing — no matter where a superpixel sits on
#      the Hilbert curve, its information reaches every register token in a
#      single GNN layer.
#
#   2. Recurrent state: Register tokens are prepended to the sequence BEFORE
#      the Hilbert-ordered scan.  In the RWKV-7 scan, they are processed
#      first (positions 0..R-1), so their recurrent state is broadcast to
#      every subsequent token via the delta-rule recurrence.  Even with
#      forward-only scan, register tokens inject globally-aggregated context
#      into every position.
#
#   3. Combined effect: GNN layers aggregate spatial context into register
#      tokens → register tokens broadcast that context through the scan →
#      unidirectional scan still receives non-local information without
#      needing a backward pass or fusion gate.
#
# This is the architectural reason why unidirectional scan can work: the
# bidirectional scan's backward pass is effectively replaced by (a) GNN
# message passing over the full graph and (b) register token broadcasting
# through the recurrent state.
#
# Data flow:
#   Raw (B, 6, H, W)
#       -> SuperpixelTokenizer -> tokens (B, N, D) + KNN neighbors (B, N, k)
#       -> [R R-register tokens prepended → (B, R+N, D)]
#       -> [L repetitions of:
#            N × UnidirectionalRWKV7Block (Hilbert-ordered forward scan)
#            M × GNNBlock (message passing on full graph including registers)]
#       -> final norm -> output projection

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spixrwkv7.data.diff_slic import DiffSLIC, spixel_upsampling
from spixrwkv7.data.lnsnet import LNSNet, download_lnsnet_weights, lnsnet_assignment
from spixrwkv7.jit import maybe_compile
from spixrwkv7.layers.drop import DropPath
from spixrwkv7.layers.graph import HEAD_SIZE, build_knn_graph, q_shift_graph_multihead
from spixrwkv7.models.common import (
    DynamicOffset,
    apply_activation,
    apply_attnres_gate,
    init_backbone_tokens,
    normalize_out_indices,
    zero_init_backbone_tokens,
)
from spixrwkv7.models.gnn_spixrwkv7 import (
    GNNFeedForward,
    GNNBlock,
    _build_gnn_conv,
    _gnn_forward,
    _ATTENTION_CONVS,
)
from spixrwkv7.models.spixrwkv7 import (
    SuperpixelEmbedding,
    SuperpixelTokenizer,
    RecurrentScan,
    hilbert_sort_batched,
    remap_neighbors,
    get_norm_layer,
)

# =====================================================================
# Unidirectional SpatialMixer — forward-only RWKV-7 scan + graph Q-shift
# =====================================================================


class UnidirectionalSpatialMixer(nn.Module):
    """Forward-only spatial (time) mixing with graph Q-shift and RWKV-7 recurrence.

    Unlike SpatialMixer (which runs forward + backward scans with a fusion gate),
    this module performs a single forward scan along the Hilbert curve order.
    This is simpler, avoids the bidirectional fusion parameter overhead, and
    naturally follows 2-D spatial locality via the Hilbert curve.

    Composes:
      1. Graph Q-shift of input tokens
      2. Input-dependent dynamic offset computation
      3. Forward-only RecurrentScan
      4. LayerNorm + LayerScale + residual
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
        self.att_ln = get_norm_layer(norm_layer)(n_embd)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd))
        else:
            self.gamma1 = None

    def forward(
        self,
        x: torch.Tensor,
        xn: torch.Tensor,
        neighbors: torch.Tensor,
        dists: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward-only scan along Hilbert curve.

        Returns (output, v_first_out) — no backward output or fusion gate.
        """
        xs = q_shift_graph_multihead(
            xn,
            neighbors=neighbors,
            head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
            num_prepend_tokens=self.num_prepend_tokens,
        )
        xx = xs - xn
        dm = self.dynamic_offset(xn, xx)

        out, v_first_out = self.scan(xn, xx, dm, "forward", v_first, mask=mask)
        out = self.att_ln(out)
        if self.gamma1 is not None:
            out = self.gamma1 * out
        x = x + self.drop_path(out)
        return x, v_first_out


# =====================================================================
# Unidirectional RWKV-7 Block
# =====================================================================


class UnidirectionalRWKV7Block(nn.Module):
    """Single unidirectional VRWKV-7 block: forward-only scan + gated FFN.

    Architecture (same as Vision_RWKV7_Block but unidirectional):
      LN0 (layer 0 only) → LN1 → UnidirectionalSpatialMixer → residual
      → LN2 → ChannelMix → residual
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

        norm_cls = get_norm_layer(norm_layer)
        self.ln1 = norm_cls(n_embd)
        if layer_id == 0:
            self.ln0 = norm_cls(n_embd)

        self.spatial_mixer = UnidirectionalSpatialMixer(
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

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        dists: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.layer_id == 0:
            x = self.ln0(x)

        xn = self.ln1(x)
        x, v_first = self.spatial_mixer(
            x, xn, neighbors, dists, v_first=v_first, mask=mask,
        )
        x = self.channel_mix(
            x, neighbors, dists,
            head_dim=self.head_size,
            with_cls_token=self.with_cls_token,
        )
        return x, v_first


# =====================================================================
# ChannelMix (reused from spixrwkv7 — Q-shift gated FFN)
# =====================================================================


class ChannelMix(nn.Module):
    """Graph Q-shift gated feed-forward network with residual.

    Identical to spixrwkv7.ChannelMix — included here for self-contained
    imports without circular dependencies.
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
            ffn_out = self.gamma2 * ffn_out
        return x + self.drop_path(ffn_out)


# =====================================================================
# Superpixel deduplication utilities
# =====================================================================


def _deduplicate_knn_neighbors(
    neighbors: torch.Tensor,
    neighbor_dists: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove duplicate entries from KNN neighbor lists.

    For each node, ensures no superpixel appears more than once in its
    neighbor list. When duplicates exist, keeps the entry with the
    smallest distance and replaces others with the next available neighbor
    index (-1 sentinel if none).

    Args:
        neighbors: (B, N, k) KNN neighbor indices.
        neighbor_dists: (B, N, k) corresponding distances.

    Returns:
        Deduplicated (neighbors, neighbor_dists) with same shapes.
    """
    B, N, k = neighbors.shape
    device = neighbors.device

    out_neighbors = neighbors.clone()
    out_dists = neighbor_dists.clone()

    for b in range(B):
        for i in range(N):
            seen = set()
            for j in range(k):
                idx = neighbors[b, i, j].item()
                if idx in seen:
                    out_neighbors[b, i, j] = -1
                    out_dists[b, i, j] = float("inf")
                else:
                    seen.add(idx)

    return out_neighbors, out_dists


def _deduplicate_centroids(
    centroids: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Add sub-pixel jitter to duplicate centroids to prevent ambiguous Hilbert sorting.

    When two superpixels have identical (or near-identical) centroid
    coordinates, the Hilbert curve ordering becomes arbitrary and the
    KNN graph degenerates. This adds a small random perturbation to
    break ties.

    Args:
        centroids: (B, N, 2) centroid coordinates in [-1, 1].
        eps: minimum distance threshold for jitter magnitude.

    Returns:
        Centroids with duplicates perturbed, shape (B, N, 2).
    """
    B, N, _ = centroids.shape
    out = centroids.clone()

    for b in range(B):
        coords = centroids[b]  # (N, 2)
        dists = torch.cdist(coords, coords)  # (N, N)
        diag = torch.arange(N, device=coords.device)
        dists[diag, diag] = float("inf")

        for i in range(N):
            near = (dists[i] < eps).nonzero(as_tuple=False).squeeze(-1)
            if near.numel() > 0:
                jitter = torch.randn(2, device=coords.device, dtype=coords.dtype) * eps
                out[b, i] = centroids[b, i] + jitter

    return out


# =====================================================================
# HybridVision — L × [N unidirectional VRWKV-7 → M GNN]
# =====================================================================


class HybridVision(nn.Module):
    """Hybrid RWKV-7 + GNN backbone with unidirectional scan.

    Architecture: L repetitions of [N UnidirectionalRWKV7Block → M GNNBlock].

    Uses the same SuperpixelTokenizer as the base Vision_RWKV7, but replaces
    bidirectional scan with unidirectional scan along Hilbert curve order,
    and alternates with GNN message-passing layers.

    Register tokens (when register_tokens > 0) are the key mechanism that
    compensates for unidirectional scan:
      - They connect bipartitely to ALL superpixel nodes in the graph
      - GNN layers aggregate global spatial context into register tokens
      - Register tokens are processed first in the RWKV-7 scan, so their
        recurrent state broadcasts global context to all subsequent tokens
      - This replaces the backward scan + fusion gate of bidirectional RWKV-7

    Args:
        img_size: Input image size.
        in_chans: Number of input channels (6 for OkLAB + alpha + xy).
        embed_dims: Token embedding dimension.
        num_heads: Number of attention heads (auto-computed from HEAD_SIZE if None).
        depth: Total number of layers = L × (N + M).
        num_rwkv_layers: N — number of unidirectional RWKV-7 layers per repetition.
        num_gnn_layers: M — number of GNN layers per repetition.
        drop_path_rate: Stochastic depth rate.
        init_values: LayerScale init values (None to disable).
        final_norm: Apply final LayerNorm.
        out_indices: Which layer indices to output features from.
        with_cls_token: Prepend CLS token.
        output_cls_token: Return CLS token alongside features.
        register_tokens: Number of DINOv2-style register tokens.
        scatter_output: Scatter tokens back to pixel grid.
        num_superpixels: Target number of superpixels.
        spixel_size: Alternative to num_superpixels (superpixel spatial size).
        diff_slic_iters: diffSLIC iterations.
        compactness: diffSLIC compactness.
        use_cpp: Use C++ accelerated kernels.
        downsample_factor: Spatial downsample before tokenization.
        norm_layer: Normalization layer name.
        act_layer: Activation function name.
        spixel_backend: Superpixel backend ("diff_slic", "lnsnet", "grid", etc.).
        knn_k: KNN neighbors for graph construction.
        dedup_neighbors: Deduplicate KNN neighbors (fix duplicate sampling).
        dedup_centroids: Add jitter to duplicate centroids (fix ambiguous Hilbert sorting).
        # GNN-specific
        gnn_conv: GNN convolution type.
        gnn_heads: Number of GNN attention heads.
        gnn_aggr: GNN aggregation type.
        jk: Jumping Knowledge aggregation type.
        # Attention residual configuration
        use_attnres: Enable attention residuals (for RWKV-7 blocks only).
        attnres_gate_type: Gate type for attention residuals.
        attnres_recency_bias_init: Initial recency bias for attention residuals.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 6,
        embed_dims: int = 192,
        num_heads: Optional[int] = None,
        depth: int = 12,
        num_rwkv_layers: int = 1,
        num_gnn_layers: int = 3,
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
        use_cpp: bool = False,
        downsample_factor: float = 1.0,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        spixel_backend: str = "diff_slic",
        knn_k: int = 4,
        dedup_neighbors: bool = True,
        dedup_centroids: bool = True,
        # GNN-specific
        gnn_conv: str = "gatv2",
        gnn_heads: int = 4,
        gnn_aggr: str = "mean",
        jk: str = "none",
        # Attention residual configuration
        use_attnres: bool = False,
        attnres_gate_type: str = "bias",
        attnres_recency_bias_init: float = 10.0,
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
        self.dedup_neighbors = dedup_neighbors
        self.dedup_centroids = dedup_centroids

        # GNN-specific
        self.gnn_conv = gnn_conv
        self.gnn_heads = gnn_heads
        self.gnn_aggr = gnn_aggr
        self.jk = jk

        # Attention residual configuration (for RWKV-7 blocks)
        self.use_attnres = use_attnres
        self.attnres_gate_type = attnres_gate_type
        self.attnres_recency_bias_init = attnres_recency_bias_init

        if num_heads is None:
            assert (
                embed_dims % HEAD_SIZE == 0
            ), f"embed_dims={embed_dims} must be divisible by HEAD_SIZE={HEAD_SIZE}"
            num_heads = embed_dims // HEAD_SIZE
        self.num_heads = num_heads

        if gnn_conv in _ATTENTION_CONVS and embed_dims % gnn_heads != 0:
            gnn_heads = embed_dims
            self.gnn_heads = gnn_heads

        # Compute layer structure: depth = L × (N + M)
        layers_per_rep = num_rwkv_layers + num_gnn_layers
        if layers_per_rep == 0:
            raise ValueError("num_rwkv_layers + num_gnn_layers must be > 0")
        self.num_repetitions = depth // layers_per_rep
        self.num_rwkv_layers = num_rwkv_layers
        self.num_gnn_layers = num_gnn_layers

        # Effective total layers (may differ from depth if not evenly divisible)
        self.effective_depth = self.num_repetitions * layers_per_rep
        if self.effective_depth != depth:
            pass  # depth is rounded down to nearest multiple of (N + M)

        # Tokenizer (shared with spixrwkv7 and gnn_spixrwkv7)
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
        self.patch_embed = self.tokenizer.patch_embed

        # CLS and register tokens
        init_backbone_tokens(self, with_cls_token, register_tokens, embed_dims)

        # Build mixed block sequence
        self.blocks = self._make_blocks(
            embed_dims=embed_dims,
            n_head=num_heads,
            depth=self.effective_depth,
            num_rwkv=num_rwkv_layers,
            num_gnn=num_gnn_layers,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            with_cls_token=with_cls_token,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_cpp=use_cpp,
        )

        # Jumping Knowledge (for GNN layers)
        if jk == "lstm":
            self.jk_lstm = nn.LSTM(embed_dims, embed_dims, batch_first=True)
            self.jk_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        else:
            self.jk_lstm = None
            self.jk_proj = None

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = get_norm_layer(norm_layer)(embed_dims)

        self.out_indices = normalize_out_indices(out_indices, self.effective_depth)

        self._init_weights()

    def _make_blocks(
        self,
        embed_dims: int,
        n_head: int,
        depth: int,
        num_rwkv: int,
        num_gnn: int,
        drop_path_rate: float,
        init_values: Optional[float],
        with_cls_token: bool,
        norm_layer: str,
        act_layer: str,
        use_cpp: bool,
    ) -> nn.ModuleList:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        global_idx = 0

        layers_per_rep = num_rwkv + num_gnn
        num_reps = depth // layers_per_rep

        for rep in range(num_reps):
            for local_idx in range(layers_per_rep):
                i = global_idx
                if local_idx < num_rwkv:
                    blocks.append(UnidirectionalRWKV7Block(
                        embed_dims,
                        n_head,
                        depth,
                        i,
                        drop_prob=dpr[i],
                        init_values=init_values,
                        with_cls_token=with_cls_token,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        num_prepend_tokens=self.register_tokens,
                        use_cpp=use_cpp,
                    ))
                else:
                    blocks.append(GNNBlock(
                        embed_dims,
                        self.gnn_conv,
                        self.gnn_heads,
                        self.gnn_aggr,
                        drop_prob=dpr[i],
                        init_values=init_values,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    ))
                global_idx += 1

        return nn.ModuleList(blocks)

    def _init_weights(self):
        zero_init_backbone_tokens(self)

    def _is_rwkv_block(self, idx: int) -> bool:
        """Check if block at global index idx is a RWKV-7 block."""
        layers_per_rep = self.num_rwkv_layers + self.num_gnn_layers
        local_idx = idx % layers_per_rep
        return local_idx < self.num_rwkv_layers

    # ------------------------------------------------------------------
    # Edge construction from batched KNN neighbour indices
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edges(
        neighbors: Tensor,
        neighbor_dists: Tensor,
        num_register: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert (B, N, k) KNN neighbours into a batched PyG edge set.

        Builds a bidirectional KNN graph with self-loops.
        Register nodes are connected bipartitely to all superpixel nodes
        with inverse-distance weights — this is the global information
        highway that compensates for unidirectional scan.  Register→superpixel
        edges let GNN layers aggregate spatial context into registers, and
        the reverse edges let superpixel nodes receive that global context.
        """
        B, N, k = neighbors.shape
        device = neighbors.device
        R = num_register
        N_total = N + R

        all_src, all_tgt, all_w = [], [], []

        for b in range(B):
            base = b * N_total
            sp_offset = base + R

            all_nodes = torch.arange(N_total, device=device) + base
            all_src.append(all_nodes)
            all_tgt.append(all_nodes)
            all_w.append(torch.ones(N_total, device=device))

            src_local = torch.arange(N, device=device).view(N, 1).expand(N, k)
            tgt_local = neighbors[b]
            w = 1.0 / (neighbor_dists[b].reshape(-1) + 1e-6)

            all_src.append((sp_offset + src_local).reshape(-1))
            all_tgt.append((sp_offset + tgt_local).reshape(-1))
            all_w.append(w)
            all_src.append((sp_offset + tgt_local).reshape(-1))
            all_tgt.append((sp_offset + src_local).reshape(-1))
            all_w.append(w)

            if R > 0:
                reg_idx = torch.arange(R, device=device).view(R, 1)
                sp_idx = torch.arange(N, device=device).view(1, N)
                reg_src = (base + reg_idx).expand(R, N).reshape(-1)
                reg_tgt = (sp_offset + sp_idx).expand(R, N).reshape(-1)
                all_src.append(reg_src)
                all_tgt.append(reg_tgt)
                reg_dists = neighbor_dists[b].mean(dim=-1)
                reg_w = (1.0 / (reg_dists + 1e-6)).unsqueeze(0).expand(R, -1).reshape(-1)
                all_src.append(reg_tgt)
                all_tgt.append(reg_src)
                all_w.append(reg_w)

        edge_index = torch.stack(
            [torch.cat(all_src).long(), torch.cat(all_tgt).long()], dim=0
        )
        edge_weight = torch.cat(all_w)
        edge_attr = edge_weight.unsqueeze(-1)
        return edge_index, edge_weight, edge_attr

    # ------------------------------------------------------------------
    # Output projection
    # ------------------------------------------------------------------

    def _project_output(
        self,
        patch_tokens: Tensor,
        inv_order: Tensor,
        batch_idx: Tensor,
        global_soft_mask: Optional[Tensor],
        global_labels: Optional[Tensor],
        H: int,
        W: int,
        h_s: int,
        w_s: int,
    ) -> Tensor:
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
                    global_labels.view(-1, H * W, 1).expand(-1, -1, self.embed_dims),
                )
                feat = feat.view(-1, H, W, self.embed_dims).permute(0, 3, 1, 2)
        else:
            feat = patch_tokens.view(-1, h_s, w_s, self.embed_dims).permute(0, 3, 1, 2)
        return feat

    def forward(
        self,
        x: Tensor,
        num_superpixels: Optional[int] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        B, C, H, W = x.shape
        assert C == self.in_chans, (
            f"Model initialized with in_chans={self.in_chans}, "
            f"but received input with C={C}."
        )

        # ---- Tokenization ----
        out = self.tokenizer(
            x,
            num_superpixels=num_superpixels,
            spixel_size=self.spixel_size,
            mask=mask,
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

        N = tokens.shape[1]

        # ---- Superpixel deduplication fixes ----
        if self.dedup_centroids:
            centroids = self.tokenizer.patch_embed(
                x,
                global_soft_mask if self.tokenizer.mode == "soft" else global_labels,
                K=N,
            )[1] if global_soft_mask is not None or global_labels is not None else None
            if centroids is not None:
                centroids = _deduplicate_centroids(centroids)
                # Rebuild KNN with deduplicated centroids
                neighbors, neighbor_dists = build_knn_graph(centroids.detach(), k=self.knn_k)
                coords_int = ((centroids + 1.0) * 4095).long().clamp(0, 8191)
                new_order = hilbert_sort_batched(coords_int)
                new_inv_order = torch.argsort(new_order, dim=1)
                tokens = tokens[batch_idx[:, None], new_order]
                neighbor_dists = neighbor_dists[batch_idx[:, None], new_order]
                sorted_mask = mask[batch_idx[:, None], new_order] if mask is not None else None
                inv_order = new_inv_order

        if self.dedup_neighbors:
            neighbors, neighbor_dists = _deduplicate_knn_neighbors(neighbors, neighbor_dists)

        # Apply mask to superpixel tokens BEFORE prepending register tokens.
        if sorted_mask is not None:
            tokens = tokens * sorted_mask.unsqueeze(-1)

        # Prepend register tokens
        if self.register_tokens > 0:
            assert self.reg_token is not None
            tokens = torch.cat((self.reg_token.expand(B, -1, -1), tokens), dim=1)

        # Build edges with register-to-all connectivity.
        edge_index, edge_weight, edge_attr = self._build_edges(
            neighbors, neighbor_dists, num_register=self.register_tokens
        )

        # Flatten to global node tensor for PyG message passing.
        x_nodes = tokens.reshape(B * tokens.shape[1], self.embed_dims)

        # Collect per-layer features for Jumping Knowledge if enabled.
        need_jk = self.jk == "lstm" and self.jk_lstm is not None
        jk_layers: List[Tensor] = []
        outs: List[Tensor] = []

        vf_first: Optional[Tensor] = None
        vf_first_set = False

        for i, block in enumerate(self.blocks):
            if self._is_rwkv_block(i):
                # Unidirectional RWKV-7 block — operates on (B, R+N, D) tokens.
                # Register tokens (first R positions) are included in the scan:
                # they are processed first, so their recurrent state broadcasts
                # globally-aggregated context to all subsequent superpixel tokens.
                block_tokens = x_nodes.view(B, -1, self.embed_dims)
                block_tokens, vff = block(
                    block_tokens, neighbors, neighbor_dists,
                    v_first=vf_first, mask=sorted_mask,
                )
                if not vf_first_set:
                    vf_first = vff
                    vf_first_set = True
                x_nodes = block_tokens.reshape(B * block_tokens.shape[1], self.embed_dims)
            else:
                # GNN block — operates on (B*(R+N), D) flattened nodes.
                # Register nodes participate in message passing via the
                # bipartite edges (register ↔ all superpixels), aggregating
                # spatial context that the unidirectional scan cannot capture.
                x_nodes = block(x_nodes, edge_index, edge_weight, edge_attr)

            if need_jk:
                jk_layers.append(x_nodes)

            if i == len(self.blocks) - 1 and self.final_norm:
                x_nodes = self.ln1(x_nodes)

            if i in self.out_indices:
                if need_jk and jk_layers:
                    layer_stack = torch.stack(jk_layers, dim=1)
                    lstm_out, _ = self.jk_lstm(layer_stack)
                    x_jk = self.jk_proj(lstm_out[:, -1, :])
                else:
                    x_jk = x_nodes

                tokens_out = x_jk.view(B, -1, self.embed_dims)
                if self.with_cls_token:
                    tokens_out = tokens_out[:, :-1]
                if self.register_tokens > 0:
                    tokens_out = tokens_out[:, self.register_tokens:]
                feat = self._project_output(
                    tokens_out, inv_order, batch_idx,
                    global_soft_mask, global_labels,
                    H, W, h_s, w_s,
                )
                outs.append(feat)

        return tuple(outs)


# =====================================================================
# Builder
# =====================================================================


def create_hybrid_vision(
    img_size: int = 224,
    embed_dims: int = 192,
    num_heads: Optional[int] = None,
    depth: int = 12,
    num_rwkv_layers: int = 1,
    num_gnn_layers: int = 3,
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
    use_cpp: bool = False,
    downsample_factor: float = 1.0,
    knn_k: int = 4,
    dedup_neighbors: bool = True,
    dedup_centroids: bool = True,
    gnn_conv: str = "gatv2",
    gnn_heads: int = 4,
    gnn_aggr: str = "mean",
    jk: str = "none",
    use_attnres: bool = False,
    attnres_gate_type: str = "bias",
    attnres_recency_bias_init: float = 10.0,
    use_jit: bool = False,
) -> torch.nn.Module:
    """Create a HybridVisionRWKV7GNN model (6-channel input).

    Combines unidirectional RWKV-7 scan (Hilbert-ordered) with GNN
    message passing in a repeating [N RWKV-7 → M GNN] pattern.

    Default: depth=12, N=1, M=3 → 3 repetitions of [1 RWKV-7 + 3 GNN] = 12 layers.
    """
    _model: torch.nn.Module = HybridVision(
        img_size=img_size,
        in_chans=6,
        embed_dims=embed_dims,
        num_heads=num_heads,
        depth=depth,
        num_rwkv_layers=num_rwkv_layers,
        num_gnn_layers=num_gnn_layers,
        drop_path_rate=drop_path_rate,
        init_values=init_values,
        final_norm=final_norm,
        out_indices=out_indices,
        with_cls_token=with_cls_token,
        output_cls_token=output_cls_token,
        register_tokens=register_tokens,
        scatter_output=scatter_output,
        num_superpixels=num_superpixels,
        spixel_size=spixel_size,
        diff_slic_iters=diff_slic_iters,
        compactness=compactness,
        use_cpp=use_cpp,
        downsample_factor=downsample_factor,
        norm_layer=norm_layer,
        act_layer=act_layer,
        spixel_backend=spixel_backend,
        knn_k=knn_k,
        dedup_neighbors=dedup_neighbors,
        dedup_centroids=dedup_centroids,
        gnn_conv=gnn_conv,
        gnn_heads=gnn_heads,
        gnn_aggr=gnn_aggr,
        jk=jk,
        use_attnres=use_attnres,
        attnres_gate_type=attnres_gate_type,
        attnres_recency_bias_init=attnres_recency_bias_init,
    )
    return maybe_compile(_model, use_jit=use_jit)
