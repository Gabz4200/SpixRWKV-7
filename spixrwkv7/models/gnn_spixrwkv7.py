# GNN-RWKV7: Superpixel Graph Neural Network ablation
#
# Compared against the recurrent Vision-RWKV-7 backbone, this variant keeps the
# same vision front-end (SuperpixelTokenizer -> Hilbert-sorted superpixel
# tokens + KNN graph) but replaces the RWKV-7 recurrence with a stack of
# Graph Neural Network (GNN) layers from PyTorch Geometric.
#
# Design intent
# -------------
# The standard backbone encodes the KNN graph only implicitly, through the
# graph Q-shift mixer.  Here the graph becomes the first-class message-passing
# topology: every superpixel token is a node, and the 4-nearest-neighbour edges
# carry message passing.  This isolates "how much of the superpixel
# representation is explained by plain GNN message passing vs. the recurrent
# delta-rule scan" — a clean ablation.
#
# Data flow (mirrors Vision_RWKV7 downstream of the tokenizer):
#   Raw (B, 6, H, W)
#       -> SuperpixelTokenizer -> tokens (B, N, D) + KNN neighbors (B, N, 4)
#       -> batched edge_index + inverse-distance edge weights
#       -> GNNBlock stack (conv message passing + feed-forward residual)
#       -> reorder to raster + project to (B, D, h_s, w_s) or (B, D, H, W)

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    GatedGraphConv,
    GraphConv,
    MultiAggregation,
    ResGatedGraphConv,
    SAGEConv,
    TransformerConv,
)
from torch_geometric.utils import dense_to_sparse

from spixrwkv7.jit import maybe_compile
from spixrwkv7.layers.drop import DropPath
from spixrwkv7.layers.graph import HEAD_SIZE
from spixrwkv7.models.spixrwkv7 import SuperpixelTokenizer, get_norm_layer

# Convs whose output head-multiplies features (need embed_dims % heads == 0).
_ATTENTION_CONVS = ("gat", "gatv2", "transformer")


# =====================================================================
# GNN conv construction + dispatch
# =====================================================================


def _make_aggr(aggr: str):
    """Build a PyG aggregation operator for SAGEConv.

    ``"multi"`` wires a :class:`MultiAggregation` (mean + max + std) so the
    ablation exercises richer neighbourhood statistics; any other string is
    passed through as a single named aggregation.
    """
    if aggr == "multi":
        return MultiAggregation(["mean", "max", "std"])
    return aggr


def _build_gnn_conv(
    embed_dims: int, conv_type: str, heads: int, aggr: str
) -> nn.Module:
    """Construct a PyG message-passing layer with output dim == ``embed_dims``."""
    if conv_type == "gcn":
        return GCNConv(embed_dims, embed_dims)
    if conv_type == "graphconv":
        return GraphConv(embed_dims, embed_dims)
    if conv_type == "sage":
        return SAGEConv(embed_dims, embed_dims, aggr=_make_aggr(aggr))
    if conv_type == "gin":
        return GINConv(
            nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
            ),
            eps=0.0,
        )
    if conv_type == "gat":
        return GATConv(embed_dims, embed_dims // heads, heads=heads)
    if conv_type == "gatv2":
        return GATv2Conv(embed_dims, embed_dims // heads, heads=heads, edge_dim=1, add_self_loops=False)
    if conv_type == "transformer":
        return TransformerConv(embed_dims, embed_dims // heads, heads=heads, edge_dim=1)
    if conv_type == "resgated":
        return ResGatedGraphConv(embed_dims, embed_dims)
    if conv_type == "gated":
        return GatedGraphConv(out_channels=embed_dims, num_layers=2)
    raise ValueError(f"Unknown gnn_conv: {conv_type}")


def _gnn_forward(
    conv: nn.Module,
    conv_type: str,
    x: Tensor,
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    edge_attr: Optional[Tensor],
) -> Tensor:
    """Dispatch a forward call, passing edge info only where the layer accepts it.

    GCN/Graph/SAGE accept a 1-D ``edge_weight``; GATv2/Transformer accept
    multi-dim ``edge_attr`` (enabling distance-aware attention); GIN/Gated/
    ResGated operate on node features alone.
    """
    if conv_type in ("gcn", "graphconv"):
        assert edge_weight is not None
        return conv(x, edge_index, edge_weight=edge_weight)
    if conv_type in ("gatv2", "transformer"):
        return conv(x, edge_index, edge_attr=edge_attr)
    return conv(x, edge_index)


# =====================================================================
# GNN feed-forward (graph-Q-shift-free MLP), mirrors ChannelMix layout
# =====================================================================


class GNNFeedForward(nn.Module):
    """Pre-norm residual feed-forward network for GNN blocks.

    Mirrors :class:`ChannelMix` (norm -> activate -> project -> norm ->
    LayerScale -> residual) but without the graph Q-shift, since the GNN
    conv already encodes the neighbourhood.
    """

    def __init__(
        self,
        embed_dims: int,
        drop_prob: float = 0.0,
        init_values: Optional[float] = None,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
    ):
        super().__init__()
        self.act_layer = act_layer
        dim_ffn = 4 * embed_dims
        # swigLU splits the projection into two equal halves (gate, value),
        # so ffn_key must emit 2*dim_ffn and ffn_value consumes dim_ffn.
        if act_layer == "swiglu":
            self.ffn_key = nn.Linear(embed_dims, 2 * dim_ffn, bias=False)
            self.ffn_value = nn.Linear(dim_ffn, embed_dims, bias=False)
        else:
            self.ffn_key = nn.Linear(embed_dims, dim_ffn, bias=False)
            self.ffn_value = nn.Linear(dim_ffn, embed_dims, bias=False)
        norm_cls = get_norm_layer(norm_layer)
        self.norm = norm_cls(embed_dims)
        self.ffn_ln = norm_cls(embed_dims)
        self.ffn_dropout = nn.Dropout(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        # LayerScale: zero-init suppresses feature blow-up at training start.
        if init_values is not None:
            self.gamma2 = nn.Parameter(init_values * torch.ones(embed_dims))
        else:
            self.gamma2 = None

    def forward(self, x: Tensor) -> Tensor:
        from spixrwkv7.models.common import apply_activation
        xn = self.norm(x)
        k = apply_activation(xn, self.act_layer, self.ffn_key)
        k = self.ffn_dropout(k)
        out = self.ffn_value(k)
        out = self.ffn_ln(out)
        if self.gamma2 is not None:
            out = self.gamma2 * out
        return x + self.drop_path(out)


# =====================================================================
# GNN block = message-passing conv + feed-forward (both residual)
# =====================================================================


class GNNBlock(nn.Module):
    """Single GNN residual block: pre-norm conv message passing + FFN.

    Supports optional attention residuals (block-level cross-attention
    over previous block representations) for parity with the base
    Vision_RWKV7_Block.
    """

    def __init__(
        self,
        embed_dims: int,
        conv_type: str,
        gnn_heads: int,
        gnn_aggr: str,
        drop_prob: float = 0.0,
        init_values: Optional[float] = None,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        use_attnres: bool = False,
        attnres_gate_type: str = "bias",
        attnres_recency_bias_init: float = 10.0,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.conv = _build_gnn_conv(embed_dims, conv_type, gnn_heads, gnn_aggr)
        self.norm1 = get_norm_layer(norm_layer)(embed_dims)
        self.ffn = GNNFeedForward(
            embed_dims,
            drop_prob=drop_prob,
            init_values=init_values,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(embed_dims))
        else:
            self.gamma1 = None

        # Attention residuals
        self.use_attnres = use_attnres
        self.attnres_gate_type = attnres_gate_type
        if use_attnres:
            self.attn_res_proj = nn.Linear(embed_dims, 1, bias=False)
            self.attn_res_norm = get_norm_layer(norm_layer)(embed_dims)
            self.attn_res_bias = nn.Parameter(torch.tensor(attnres_recency_bias_init))
            nn.init.zeros_(self.attn_res_proj.weight)
            if attnres_gate_type == "sigmoid_scalar":
                self.attn_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
            elif attnres_gate_type == "sigmoid_vector":
                self.attn_res_gate_proj = nn.Linear(embed_dims, embed_dims, bias=True)
                nn.init.zeros_(self.attn_res_gate_proj.weight)
                nn.init.constant_(self.attn_res_gate_proj.bias, -2.0)
            elif attnres_gate_type == "learnable_alpha":
                self.attn_res_alpha = nn.Parameter(torch.tensor(0.0))

    def _apply_attnres_gate(
        self, partial: Tensor, h_attn: Tensor
    ) -> Tensor:
        from spixrwkv7.models.common import apply_attnres_gate
        return apply_attnres_gate(
            partial, h_attn, self.attnres_gate_type,
            gate_logit=getattr(self, "attn_res_gate_logit", None),
            gate_proj=getattr(self, "attn_res_gate_proj", None),
            alpha=getattr(self, "attn_res_alpha", None),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        edge_attr: Optional[Tensor],
        attnres_history: Optional[list] = None,
    ) -> Tensor:
        # Optional: attend over previous block representations
        if self.use_attnres and attnres_history is not None and len(attnres_history) > 0:
            from spixrwkv7.models.spixrwkv7 import block_attn_res
            # GNN features are (B*(R+N), D); reshape to (B, R+N, D) for attnres
            n_nodes = x.shape[0]
            B = attnres_history[0].shape[0]
            T = n_nodes // B
            x_3d = x.view(B, T, -1)
            hist_3d = [h.view(B, T, -1) if h.dim() == 2 else h for h in attnres_history]
            h_attn = block_attn_res(
                hist_3d, x_3d,
                self.attn_res_proj, self.attn_res_norm, self.attn_res_bias,
            )
            x = self._apply_attnres_gate(x, h_attn.view(B * T, -1))

        h = self.norm1(x)
        out = _gnn_forward(
            self.conv, self.conv_type, h, edge_index, edge_weight, edge_attr
        )
        if self.gamma1 is not None:
            out = self.gamma1 * out
        x = x + self.drop_path(out)
        x = self.ffn(x)
        return x


# =====================================================================
# Global Attention Block — full-graph message passing
# =====================================================================


class GlobalAttentionBlock(nn.Module):
    """Full-graph attention layer that lets every node attend to every other node.

    When depth < num_superpixels, local KNN message passing cannot propagate
    information across the full graph within the available layers.  This block
    adds a complete-graph TransformerConv so every node can exchange features
    with all other nodes in a single layer.

    Architecture: LayerNorm → TransformerConv(complete graph) → LayerScale
                  → residual → FFN → residual

    The complete-graph edge index is precomputed once per forward pass and
    shared across all global attention layers.
    """

    def __init__(
        self,
        embed_dims: int,
        n_head: int,
        drop_prob: float = 0.0,
        init_values: Optional[float] = None,
        norm_layer: str = "layernorm",
    ):
        super().__init__()
        self.norm1 = get_norm_layer(norm_layer)(embed_dims)
        self.conv = TransformerConv(
            embed_dims, embed_dims // n_head, heads=n_head, edge_dim=1,
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.ffn = GNNFeedForward(
            embed_dims,
            drop_prob=drop_prob,
            init_values=init_values,
            norm_layer=norm_layer,
        )
        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones(embed_dims))
        else:
            self.gamma1 = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.norm1(x)
        out = self.conv(h, edge_index, edge_attr=edge_attr)
        if self.gamma1 is not None:
            out = self.gamma1 * out
        x = x + self.drop_path(out)
        x = self.ffn(x)
        return x


# =====================================================================
# GNNVision backbone
# =====================================================================


class GNNVision(nn.Module):
    """Superpixel GNN backbone: SuperpixelTokenizer -> GNN message passing.

    Reuses the project's superpixel tokenization (diffSLIC/LNSNet/grid) and
    Hilbert reordering, then applies a stack of GNN conv layers over the
    KNN superpixel graph instead of the RWKV-7 recurrence.
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
        use_cpp: bool = False,
        downsample_factor: float = 1.0,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        spixel_backend: str = "diff_slic",
        knn_k: int = 4,
        # GNN-specific configuration
        gnn_conv: str = "gatv2",
        gnn_heads: int = 4,
        gnn_aggr: str = "mean",
        jk: str = "none",
        register_edge_weight_scale: float = 1.0,
        use_global_attn: bool = True,
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
        self.gnn_conv = gnn_conv
        self.gnn_heads = gnn_heads
        self.gnn_aggr = gnn_aggr
        self.jk = jk
        self.register_edge_weight_scale = register_edge_weight_scale
        self.use_attnres = use_attnres
        self.attnres_gate_type = attnres_gate_type
        self.attnres_recency_bias_init = attnres_recency_bias_init

        if num_heads is None:
            assert (
                embed_dims % HEAD_SIZE == 0
            ), f"embed_dims={embed_dims} must be divisible by HEAD_SIZE={HEAD_SIZE} if num_heads is not provided"
            num_heads = embed_dims // HEAD_SIZE

        # Force embed_dims divisible by gnn heads for attention convs.
        if gnn_conv in _ATTENTION_CONVS and embed_dims % gnn_heads != 0:
            gnn_heads = embed_dims
            self.gnn_heads = gnn_heads

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
        # Public alias expected by benchmark scripts (tokenizer access).
        self.patch_embed = self.tokenizer.patch_embed

        from spixrwkv7.models.common import init_backbone_tokens
        init_backbone_tokens(self, with_cls_token, register_tokens, embed_dims)

        self.blocks = self._make_blocks(
            embed_dims,
            depth,
            drop_path_rate,
            init_values,
            norm_layer,
            act_layer,
        )

        # Jumping Knowledge (JK) aggregation across layers.
        # When jk="lstm", an LSTM reads the per-layer feature stack and
        # produces a single fused representation (Xu et al., Representation
        # Learning on Graphs with Jumping Knowledge Networks, ICML 2018).
        if jk == "lstm":
            self.jk_lstm = nn.LSTM(embed_dims, embed_dims, batch_first=True)
            self.jk_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        else:
            self.jk_lstm = None
            self.jk_proj = None

        # Global attention layers — full-graph TransformerConv at middle and
        # end of the stack.  When depth < num_superpixels, local KNN message
        # passing cannot reach all nodes within the available layers.  These
        # global layers ensure every node can attend to every other node.
        self.use_global_attn = use_global_attn
        self.global_attn_layers = nn.ModuleList()
        self.global_attn_positions: List[int] = []
        if use_global_attn and depth >= 2:
            mid = depth // 2
            end = depth - 1
            self.global_attn_positions = [mid, end]
            for pos in [mid, end]:
                dp_val = drop_path_rate * pos / max(depth - 1, 1)
                self.global_attn_layers.append(
                    GlobalAttentionBlock(
                        embed_dims, gnn_heads, drop_prob=dp_val,
                        init_values=init_values, norm_layer=norm_layer,
                    )
                )

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = get_norm_layer(norm_layer)(embed_dims)

        from spixrwkv7.models.common import normalize_out_indices
        self.out_indices = normalize_out_indices(out_indices, depth)

        self._init_weights()

    def _make_blocks(
        self,
        embed_dims: int,
        depth: int,
        drop_path_rate: float,
        init_values: Optional[float],
        norm_layer: str,
        act_layer: str,
    ) -> nn.ModuleList:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        return nn.ModuleList(
            [
                GNNBlock(
                    embed_dims,
                    self.gnn_conv,
                    self.gnn_heads,
                    self.gnn_aggr,
                    drop_prob=dpr[i],
                    init_values=init_values,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_attnres=self.use_attnres,
                    attnres_gate_type=self.attnres_gate_type if hasattr(self, 'attnres_gate_type') else "bias",
                    attnres_recency_bias_init=self.attnres_recency_bias_init if hasattr(self, 'attnres_recency_bias_init') else 10.0,
                )
                for i in range(depth)
            ]
        )

    def _init_weights(self):
        from spixrwkv7.models.common import zero_init_backbone_tokens
        zero_init_backbone_tokens(self)

    # ------------------------------------------------------------------
    # Edge construction from batched KNN neighbour indices
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edges(
        neighbors: Tensor,
        neighbor_dists: Tensor,
        num_register: int = 0,
        register_edge_weight_scale: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert (B, N, k) KNN neighbours into a batched PyG edge set.

        Builds a **bidirectional** KNN graph: for each directed edge (i→j)
        the reverse edge (j→i) is also added, ensuring symmetric message
        passing.  When ``num_register > 0``, register nodes (first R indices
        per batch item) are connected to ALL superpixel nodes (bipartite)
        with inverse-distance weights, scaled by ``register_edge_weight_scale``
        to prevent register token hub domination.

        Returns ``(edge_index (2, E), edge_weight (E,), edge_attr (E, 1))``
        with global node indices ``b*(N+R) + local`` so graphs for each batch
        item stay disjoint.
        """
        B, N, k = neighbors.shape
        device = neighbors.device
        R = num_register
        N_total = N + R

        all_src, all_tgt, all_w = [], [], []

        for b in range(B):
            base = b * N_total
            sp_offset = base + R

            # --- Self-loops for all nodes (needed when add_self_loops=False) ---
            all_nodes = torch.arange(N_total, device=device) + base
            all_src.append(all_nodes)
            all_tgt.append(all_nodes)
            all_w.append(torch.ones(N_total, device=device))

            # --- Bidirectional KNN edges for superpixel nodes ---
            src_local = torch.arange(N, device=device).view(N, 1).expand(N, k)
            tgt_local = neighbors[b]  # (N, k)
            w = 1.0 / (neighbor_dists[b].reshape(-1) + 1e-6)  # (N*k,)

            # Forward edges: i → j
            all_src.append((sp_offset + src_local).reshape(-1))
            all_tgt.append((sp_offset + tgt_local).reshape(-1))
            all_w.append(w)
            # Reverse edges: j → i (ensures symmetric connectivity)
            all_src.append((sp_offset + tgt_local).reshape(-1))
            all_tgt.append((sp_offset + src_local).reshape(-1))
            all_w.append(w)

            # --- Register → all superpixels (bipartite, inverse-distance) ---
            if R > 0:
                reg_idx = torch.arange(R, device=device).view(R, 1)
                sp_idx = torch.arange(N, device=device).view(1, N)
                reg_src = (base + reg_idx).expand(R, N).reshape(-1)
                reg_tgt = (sp_offset + sp_idx).expand(R, N).reshape(-1)
                # Distance-decayed weights for register edges, scaled to
                # prevent register hub domination (see over-smoothing notes).
                reg_dists = neighbor_dists[b].mean(dim=-1)  # (N,) mean KNN dist per node
                reg_w = (1.0 / (reg_dists + 1e-6)).unsqueeze(0).expand(R, -1).reshape(-1)
                reg_w = reg_w * register_edge_weight_scale
                # Forward: register → superpixel
                all_src.append(reg_src)
                all_tgt.append(reg_tgt)
                all_w.append(reg_w)
                # Backward: superpixel → register
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
    # Output projection (identical semantics to the RWKV backbone)
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
        """Forward pass returning multi-scale features.

        Args:
            x: (B, C, H, W) 6-channel (OkLAB + alpha + xy) input.
            num_superpixels: optional override for target superpixel count.

        Returns:
            Tuple of (B, embed_dims, h_s, w_s) or (B, embed_dims, H, W)
            feature maps, one per ``out_indices`` entry.
        """
        B, C, H, W = x.shape
        assert C == self.in_chans, (
            f"Model initialized with in_chans={self.in_chans}, "
            f"but received input with C={C}."
        )

        out = self.tokenizer(
            x,
            num_superpixels=num_superpixels,
            spixel_size=self.spixel_size,
            mask=mask,
        )
        tokens = out["tokens"]  # (B, N, D) — superpixel tokens only
        neighbors = out["neighbors"]
        neighbor_dists = out["neighbor_dists"]
        inv_order = out["inv_order"]
        batch_idx = out["batch_idx"]
        global_soft_mask = out["global_soft_mask"]
        global_labels = out["global_labels"]
        h_s, w_s = out["h_s"], out["w_s"]
        sorted_mask = out["mask"]

        N = tokens.shape[1]  # number of superpixel tokens

        # Apply mask to superpixel tokens BEFORE prepending register tokens.
        # sorted_mask has shape (B, N), matching tokens (B, N, D) exactly.
        if sorted_mask is not None:
            tokens = tokens * sorted_mask.unsqueeze(-1)

        # Prepend register tokens (DINOv2-style) — they connect to ALL nodes.
        if self.register_tokens > 0:
            assert self.reg_token is not None
            tokens = torch.cat((self.reg_token.expand(B, -1, -1), tokens), dim=1)

        # Build edges with register-to-all connectivity.
        edge_index, edge_weight, edge_attr = self._build_edges(
            neighbors, neighbor_dists, num_register=self.register_tokens,
            register_edge_weight_scale=self.register_edge_weight_scale,
        )

        # Flatten to global node tensor for PyG message passing.
        # Shape: (B * (R + N), D) where R = register_tokens.
        x_nodes = tokens.reshape(B * tokens.shape[1], self.embed_dims)

        # Collect per-layer features for Jumping Knowledge if enabled.
        # Only collect when an output index needs them (optimization for Issue 16).
        need_jk = self.jk == "lstm" and self.jk_lstm is not None
        jk_layers: List[Tensor] = []
        attnres_history: Optional[list] = [x_nodes] if self.use_attnres else None
        outs: List[Tensor] = []

        # Build complete-graph edge index for global attention layers.
        # Shape: (2, N_total^2) — every node attends to every other node.
        N_total = tokens.shape[1]  # R + N
        if self.use_global_attn and len(self.global_attn_layers) > 0:
            adj = torch.ones(N_total, N_total, device=x_nodes.device)
            global_edge_index, _ = dense_to_sparse(adj)
            global_edge_attr = torch.ones(
                global_edge_index.shape[1], 1, device=x_nodes.device,
            )
        global_attn_idx = 0

        for i, block in enumerate(self.blocks):
            if self.use_attnres and attnres_history is not None:
                x_nodes = block(x_nodes, edge_index, edge_weight, edge_attr, attnres_history=attnres_history)
                if i < len(self.blocks) - 1:
                    attnres_history.append(x_nodes)
            else:
                x_nodes = block(x_nodes, edge_index, edge_weight, edge_attr)

            # Apply global attention at computed positions.
            if (self.use_global_attn
                    and global_attn_idx < len(self.global_attn_positions)
                    and i == self.global_attn_positions[global_attn_idx]):
                x_nodes = self.global_attn_layers[global_attn_idx](
                    x_nodes, global_edge_index, global_edge_attr,
                )
                global_attn_idx += 1

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
                    tokens_out = tokens_out[:, self.register_tokens :]
                feat = self._project_output(
                    tokens_out,
                    inv_order,
                    batch_idx,
                    global_soft_mask,
                    global_labels,
                    H,
                    W,
                    h_s,
                    w_s,
                )
                outs.append(feat)

        if self.use_attnres and attnres_history is not None:
            self.last_attnres_history = [t.detach() for t in attnres_history]

        return tuple(outs)


# =====================================================================
# Builder
# =====================================================================


def create_gnn_vision(
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
    use_cpp: bool = False,
    downsample_factor: float = 1.0,
    knn_k: int = 4,
    gnn_conv: str = "gatv2",
    gnn_heads: int = 4,
    gnn_aggr: str = "mean",
    jk: str = "none",
    register_edge_weight_scale: float = 1.0,
    use_global_attn: bool = True,
    use_attnres: bool = False,
    attnres_gate_type: str = "bias",
    attnres_recency_bias_init: float = 10.0,
    use_jit: bool = False,
) -> torch.nn.Module:
    """Create a :class:`GNNVision` model (6-channel input).

    Mirrors :func:`create_vision_rwkv7`'s contract so the two backbones are
    drop-in comparable, with GNN-specific additions
    (``gnn_conv``, ``gnn_heads``, ``gnn_aggr``, ``jk``, ``knn_k``).
    """
    _model: torch.nn.Module = GNNVision(
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
        gnn_conv=gnn_conv,
        gnn_heads=gnn_heads,
        gnn_aggr=gnn_aggr,
        jk=jk,
        register_edge_weight_scale=register_edge_weight_scale,
        use_global_attn=use_global_attn,
        use_attnres=use_attnres,
        attnres_gate_type=attnres_gate_type,
        attnres_recency_bias_init=attnres_recency_bias_init,
    )
    return maybe_compile(_model, use_jit=use_jit)
