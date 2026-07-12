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
    GATConv,
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

from spixrwkv7.jit import maybe_compile
from spixrwkv7.layers.drop import DropPath
from spixrwkv7.models.spixrwkv7 import SuperpixelTokenizer, get_norm_layer

# Convs whose output head-multiplies features (need embed_dims % heads == 0).
_ATTENTION_CONVS = ("gat", "gatv2", "transformer")


# =====================================================================
# GNN conv construction + dispatch
# =====================================================================


def _make_aggr(aggr: str, dim: int):
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
        return SAGEConv(embed_dims, embed_dims, aggr=_make_aggr(aggr, embed_dims))
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
        return GATv2Conv(embed_dims, embed_dims // heads, heads=heads)
    if conv_type == "transformer":
        return TransformerConv(embed_dims, embed_dims // heads, heads=heads)
    if conv_type == "resgated":
        return ResGatedGraphConv(embed_dims, embed_dims)
    if conv_type == "gated":
        return GatedGraphConv(embed_dims, num_layers=2)
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

    Based on the PyG operator cheatsheet: GCN/Graph/SAGE/Gated/ResGated accept
    a 1-D ``edge_weight``; GAT/GATv2/Transformer accept multi-dim ``edge_attr``;
    GIN operates on node features alone.
    """
    if conv_type in ("gcn", "graphconv"):
        assert edge_weight is not None
        return conv(x, edge_index, edge_weight=edge_weight)
    # SAGE/GAT/GATv2/GIN/Gated/ResGated/Transformer learn their own message
    # weights; our per-edge distance features are not consumed in this PyG
    # version (would require edge_dim configuration at construction).
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
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        # LayerScale: zero-init suppresses feature blow-up at training start.
        if init_values is not None:
            self.gamma2 = nn.Parameter(init_values * torch.ones(embed_dims))
        else:
            self.gamma2 = None

    def forward(self, x: Tensor) -> Tensor:
        xn = self.norm(x)
        if self.act_layer == "relu2":
            k = F.relu(self.ffn_key(xn)).pow(2)
        elif self.act_layer == "gelu":
            k = F.gelu(self.ffn_key(xn))
        elif self.act_layer == "silu":
            k = F.silu(self.ffn_key(xn))
        elif self.act_layer == "swiglu":
            gate, val = self.ffn_key(xn).chunk(2, dim=-1)
            k = F.silu(gate) * val
        else:
            raise ValueError(f"Unknown activation layer: {self.act_layer}")
        out = self.ffn_value(k)
        out = self.ffn_ln(out)
        if self.gamma2 is not None:
            out = self.gamma2 * out
        return x + self.drop_path(out)


# =====================================================================
# GNN block = message-passing conv + feed-forward (both residual)
# =====================================================================


class GNNBlock(nn.Module):
    """Single GNN residual block: pre-norm conv message passing + FFN."""

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

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        edge_attr: Optional[Tensor],
    ) -> Tensor:
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
# GNNVision_RWKV7 backbone
# =====================================================================


class GNNVision_RWKV7(nn.Module):
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
        # GNN-specific configuration
        gnn_conv: str = "gatv2",
        gnn_heads: int = 4,
        gnn_aggr: str = "mean",
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

        if num_heads is None:
            assert (
                embed_dims % 64 == 0
            ), f"embed_dims={embed_dims} must be divisible by 64 if num_heads is not provided"
            num_heads = embed_dims // 64

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
        )
        # Public alias expected by benchmark scripts (tokenizer access).
        self.patch_embed = self.tokenizer.patch_embed

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        self.register_tokens = register_tokens
        if register_tokens > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, register_tokens, embed_dims))
        else:
            self.reg_token = None

        self.blocks = self._make_blocks(
            embed_dims,
            depth,
            drop_path_rate,
            init_values,
            norm_layer,
            act_layer,
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

    # ------------------------------------------------------------------
    # Edge construction from batched KNN neighbour indices
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edges(
        neighbors: Tensor, neighbor_dists: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert (B, N, k) KNN neighbours into a batched PyG edge set.

        Returns ``(edge_index (2, E), edge_weight (E,), edge_attr (E, 1))``
        with global node indices ``b*N + local`` so graphs for each batch
        item stay disjoint.  Edge weights are inverse distances.
        """
        B, N, k = neighbors.shape
        device = neighbors.device
        offsets = torch.arange(B, device=device).view(B, 1, 1)
        src = (
            offsets + torch.arange(N, device=device).view(1, N, 1)
        ).expand(B, N, k).reshape(-1)
        tgt = (offsets + neighbors).reshape(-1)
        edge_index = torch.stack([src, tgt], dim=0).long()
        edge_weight = 1.0 / (neighbor_dists.reshape(-1) + 1e-6)
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
        tokens = out["tokens"]
        neighbors = out["neighbors"]
        neighbor_dists = out["neighbor_dists"]
        inv_order = out["inv_order"]
        batch_idx = out["batch_idx"]
        global_soft_mask = out["global_soft_mask"]
        global_labels = out["global_labels"]
        h_s, w_s = out["h_s"], out["w_s"]
        sorted_mask = out["mask"]

        if self.with_cls_token:
            tokens = torch.cat((tokens, self.cls_token.expand(B, -1, -1)), dim=1)
        if self.register_tokens > 0:
            assert self.reg_token is not None
            tokens = torch.cat((self.reg_token.expand(B, -1, -1), tokens), dim=1)

        edge_index, edge_weight, edge_attr = self._build_edges(
            neighbors, neighbor_dists
        )

        # Optional token masking: zero masked-out superpixel features so they
        # neither emit nor aggregate meaningful messages.
        if sorted_mask is not None:
            tokens = tokens * sorted_mask.unsqueeze(-1)

        # Flatten to global node tensor for PyG message passing.
        x_nodes = tokens.reshape(B * tokens.shape[1], self.embed_dims)

        outs: List[Tensor] = []
        for i, block in enumerate(self.blocks):
            x_nodes = block(x_nodes, edge_index, edge_weight, edge_attr)
            if i == len(self.blocks) - 1 and self.final_norm:
                x_nodes = self.ln1(x_nodes)

            if i in self.out_indices:
                tokens_out = x_nodes.view(B, -1, self.embed_dims)
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

        return tuple(outs)


# =====================================================================
# Builder
# =====================================================================


def create_gnn_vision_rwkv7(
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
    gnn_conv: str = "gatv2",
    gnn_heads: int = 4,
    gnn_aggr: str = "mean",
    use_jit: bool = False,
) -> torch.nn.Module:
    """Create a :class:`GNNVision_RWKV7` ablation model (6-channel input).

    Mirrors :func:`create_vision_rwkv7`'s contract so the two backbones are
    drop-in comparable, with three GNN-specific additions
    (``gnn_conv``, ``gnn_heads``, ``gnn_aggr``).
    """
    _model: torch.nn.Module = GNNVision_RWKV7(
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
        gnn_conv=gnn_conv,
        gnn_heads=gnn_heads,
        gnn_aggr=gnn_aggr,
    )
    return maybe_compile(_model, use_jit=use_jit)
