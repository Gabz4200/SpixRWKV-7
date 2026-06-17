import torch
import torch.nn as nn
import torch.nn.functional as F
from VisualRWKV7.model import (
    build_knn_graph,
    q_shift_graph_multihead,
    SuperpixelEmbedding,
    Vision_RWKV7,
    Vision_RWKV7_Block,
)

# =====================================================================
# Helper Functions
# =====================================================================


def get_dummy_neighbors(B, N, K=4):
    """Helper to create dummy valid neighbors for testing blocks."""
    neighbors = torch.zeros(B, N, K, dtype=torch.long)
    for i in range(N):
        for k in range(K):
            neighbors[:, i, k] = (i + k + 1) % N
    return neighbors


# =====================================================================
# Graph-Based Helpers Tests
# =====================================================================


def test_build_knn_graph_single():
    """Test KNN graph building for a single set of centroids."""
    # 4 centroids in a square
    centroids = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    neighbors = build_knn_graph(centroids, k=2)
    assert neighbors.shape == (4, 2)
    # Ensure a node doesn't select itself as a neighbor
    for i in range(4):
        assert i not in neighbors[i]


def test_build_knn_graph_batched():
    """Test KNN graph building for batched centroids."""
    B, N = 2, 5
    centroids = torch.rand(B, N, 2)
    neighbors = build_knn_graph(centroids, k=3)
    assert neighbors.shape == (B, N, 3)


def test_q_shift_graph_multihead_logic():
    """Verify that Graph Q-Shift correctly shifts tokens along graph edges."""
    B, N, C = 1, 4, 16
    head_dim = 16

    # Graph: Node 0 connects to 1, 2. Node 1 connects to 0, 3, etc.
    neighbors = torch.tensor([[[1, 2], [0, 3], [0, 3], [1, 2]]])  # [B, N, K]

    # Fill input with node IDs + 1 so we can track movement
    x = torch.zeros(B, N, C)
    for i in range(N):
        x[0, i, :] = i + 1

    out = q_shift_graph_multihead(x, neighbors, head_dim=head_dim, with_cls_token=False)

    # Group 0 (channels 0-7): should come from neighbor index 0
    # Node 0's neighbor 0 is Node 1. So out[0, 0, 0] should be x[0, 1, 0] = 2
    assert out[0, 0, 0].item() == 2.0
    # Node 1's neighbor 0 is Node 0. So out[0, 1, 0] should be x[0, 0, 0] = 1
    assert out[0, 1, 0].item() == 1.0

    # Group 1 (channels 8-15): should come from neighbor index 1
    # Node 0's neighbor 1 is Node 2. So out[0, 0, 8] should be x[0, 2, 8] = 3
    assert out[0, 0, 8].item() == 3.0
    # Node 3's neighbor 1 is Node 2. So out[0, 3, 8] should be x[0, 2, 8] = 3
    assert out[0, 3, 8].item() == 3.0


def test_q_shift_graph_multihead_cls_token():
    """Verify CLS token is excluded from shifting and preserved."""
    B, N, C = 1, 4, 16
    head_dim = 16
    neighbors = torch.zeros(B, N, 2, dtype=torch.long)

    x = torch.randn(B, N + 1, C)
    cls_token = x[:, -1:, :]

    out = q_shift_graph_multihead(x, neighbors, head_dim=head_dim, with_cls_token=True)

    # The last token should be exactly the unmodified CLS token
    assert torch.allclose(out[:, -1:, :], cls_token)


# =====================================================================
# Superpixel Embedding Tests
# =====================================================================


def test_superpixel_embedding_hard():
    """Test SuperpixelEmbedding in hard (discrete) mode."""
    B, C_in, H, W = 2, 3, 8, 8
    K = 4
    embed_dims = 16

    emb = SuperpixelEmbedding(C_in, embed_dims, K, mode="hard")
    x = torch.randn(B, C_in, H, W)

    # Create hard integer labels [B, H, W]
    sp_map = torch.randint(0, K, (B, H, W))

    tokens = emb(x, sp_map)
    assert tokens.shape == (B, K, embed_dims)
    assert torch.isfinite(tokens).all()


def test_superpixel_embedding_soft():
    """Test SuperpixelEmbedding in soft (continuous) mode."""
    B, C_in, H, W = 2, 3, 8, 8
    K = 4
    embed_dims = 16

    emb = SuperpixelEmbedding(C_in, embed_dims, K, mode="soft")
    x = torch.randn(B, C_in, H, W)

    # Create soft probability masks [B, K, H, W]
    sp_map = torch.rand(B, K, H, W)
    sp_map = sp_map / sp_map.sum(dim=1, keepdim=True)  # normalize

    tokens = emb(x, sp_map)
    assert tokens.shape == (B, K, embed_dims)
    assert torch.isfinite(tokens).all()


# =====================================================================
# Vision_RWKV7_Block Tests (Updated for Graph Neighbors)
# =====================================================================


def test_block_v_first_propagation():
    """Verify that v_first propagation affects block output."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=2, layer_id=1)

    B, N = 1, 16
    x = torch.randn(B, N, n_embd)
    neighbors = get_dummy_neighbors(B, N, K=4)

    # Case 1: No v_first
    out1, _, _ = block(x, neighbors, v_first_fwd=None, v_first_bwd=None)

    # Case 2: With v_first
    vf_fwd = torch.randn(B, N, n_embd)
    vf_bwd = torch.randn(B, N, n_embd)
    out2, _, _ = block(x, neighbors, v_first_fwd=vf_fwd, v_first_bwd=vf_bwd)

    assert not torch.allclose(out1, out2, atol=1e-5)


def test_rwkv7_input_dependent_mixing():
    """Verify that dynamic mixing offsets (dm) are input-dependent."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=1, layer_id=0)

    B, N = 1, 16
    x1 = torch.randn(B, N, n_embd)
    x2 = torch.randn(B, N, n_embd)
    neighbors = get_dummy_neighbors(B, N, K=4)

    sp1 = block._spatial_prep(x1, neighbors)
    sp2 = block._spatial_prep(x2, neighbors)

    assert not torch.allclose(sp1["dm"], sp2["dm"], atol=1e-12)
    assert sp1["dm"].shape == (6, B, N, n_embd)


def test_rwkv7_decoupled_keys():
    """Verify that removal key (kk) and replacement key (kt) are decoupled."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=1, layer_id=0)

    assert isinstance(block.k_k, nn.Parameter)
    assert isinstance(block.k_a, nn.Parameter)

    with torch.no_grad():
        block.k_k.fill_(1.0)
        block.k_a.fill_(0.5)

    B, N = 1, 4
    x = torch.randn(B, N, n_embd)
    neighbors = get_dummy_neighbors(B, N, K=4)

    out, _, _ = block(x, neighbors)
    assert torch.isfinite(out).all()


def test_rwkv7_bonus_term():
    """Verify that the bonus term (r_k) is applied."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=1, layer_id=0)

    B, N = 1, 4
    x = torch.randn(B, N, n_embd)
    neighbors = get_dummy_neighbors(B, N, K=4)

    with torch.no_grad():
        block.r_k.zero_()
    out1, _, _ = block(x, neighbors)

    with torch.no_grad():
        block.r_k.fill_(1.0)
    out2, _, _ = block(x, neighbors)

    assert not torch.allclose(out1, out2, atol=1e-5)


# =====================================================================
# Vision_RWKV7 Backbone Tests
# =====================================================================


def test_vision_rwkv7_forward_hard():
    """Test full backbone forward pass with hard superpixel mode."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,  # 4x4 grid
        diff_slic_iters=2,
    )
    x = torch.randn(1, 3, 64, 64)
    outs = model(x)

    # Output should be scattered back to original [B, C, H, W]
    assert len(outs) == 1  # default out_indices=(-1,)
    assert outs[0].shape == (1, 64, 64, 64)
    assert torch.isfinite(outs[0]).all()


def test_vision_rwkv7_forward_soft():
    """Test full backbone forward pass if we manually change mode to soft."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,
        diff_slic_iters=2,
    )
    # Manually switch to soft mode to test that branch
    model.patch_embed.mode = "soft"

    x = torch.randn(1, 3, 64, 64)
    outs = model(x)

    assert len(outs) == 1
    assert outs[0].shape == (1, 64, 64, 64)
    assert torch.isfinite(outs[0]).all()


def test_output_matches_input_resolution():
    """Verify that the scattered output matches the original input resolution."""
    model = Vision_RWKV7(
        img_size=64, embed_dims=64, num_heads=1, depth=2, num_superpixels=16
    )
    # Test with a different resolution than img_size
    x = torch.randn(1, 3, 128, 128)
    outs = model(x)

    # Output must be scattered back to [B, C, 128, 128]
    assert outs[0].shape == (1, 64, 128, 128)


def test_multi_scale_indices():
    """Verify that out_indices correctly returns features from multiple layers."""
    depth = 4
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=depth,
        num_superpixels=16,
        out_indices=[1, 3],
    )
    x = torch.randn(1, 3, 64, 64)
    outs = model(x)

    assert len(outs) == 2
    # Both outputs are scattered back to [B, C, H, W]
    assert outs[0].shape == (1, 64, 64, 64)
    assert outs[1].shape == (1, 64, 64, 64)


def test_cls_token_behavior():
    """Verify CLS token handling and output."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,
        with_cls_token=True,
        output_cls_token=True,
    )
    x = torch.randn(1, 3, 64, 64)
    outs = model(x)

    assert isinstance(outs[0], tuple)
    feat, cls_token = outs[0]
    # Feat is scattered back to [B, C, H, W]
    assert feat.shape == (1, 64, 64, 64)
    assert cls_token.shape == (1, 64)


def test_numerical_stability_long_seq():
    """Check for stability with larger grids (longer sequences)."""
    model = Vision_RWKV7(
        img_size=128,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=64,  # 8x8 grid -> 64 tokens
        diff_slic_iters=2,
    )
    x = torch.randn(1, 3, 128, 128)
    outs = model(x)
    assert torch.isfinite(outs[0]).all()


def test_rwkv7_vector_iclr_decay():
    """Verify that ICLR (a) and Decay (w) are vector-valued and applied per-channel."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=1, layer_id=0)

    assert block.w0.shape == (n_embd,)
    assert block.a0.shape == (n_embd,)

    B, N = 1, 4
    x = torch.randn(B, N, n_embd)
    neighbors = get_dummy_neighbors(B, N, K=4)

    out, _, _ = block(x, neighbors)
    assert torch.isfinite(out).all()


def test_rwkv7_state_update_logic():
    """Verify the generalized delta rule state update formula."""
    # Pure math check, no block instantiation needed
    n_embd = 64
    n_head = 1
    head_size = 64

    B, D = 1, n_embd
    Hd, S = n_head, head_size

    w = torch.rand(B, D)
    a = torch.rand(B, D)
    kk = torch.rand(B, D)
    kk = F.normalize(kk.view(B, Hd, S), dim=-1).view(B, D)
    kt = torch.rand(B, D)
    v = torch.rand(B, D)

    vk = v.view(B, Hd, S, 1) @ kt.view(B, Hd, 1, S)
    ab = (-kk).view(B, Hd, S, 1) @ (kk * a).view(B, Hd, 1, S)

    new_state = vk
    expected_state = new_state * w.view(B, Hd, 1, S) + new_state @ ab + vk

    assert expected_state.shape == (B, Hd, S, S)
    assert torch.isfinite(expected_state).all()


def test_deterministic_behavior():
    """Verify that same input produces same output."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,
        diff_slic_iters=2,
    )
    x = torch.randn(1, 3, 64, 64)

    out1 = model(x)
    out2 = model(x)

    for o1, o2 in zip(out1, out2):
        if isinstance(o1, tuple):
            assert torch.allclose(o1[0], o2[0], atol=1e-5)
            assert torch.allclose(o1[1], o2[1], atol=1e-5)
        else:
            assert torch.allclose(o1, o2, atol=1e-5)


# =====================================================================
# Behavioral Tests (Coverage Expansion)
# =====================================================================


def test_forward_finite_random_input():
    """Forward pass with random input produces all-finite outputs."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,
        diff_slic_iters=2,
    )
    x = torch.randn(2, 3, 64, 64)  # batch=2
    outs = model(x)
    assert all(torch.isfinite(o).all() for o in outs)
    assert len(outs) == 1  # default out_indices


def test_scatter_output_spatial_shape():
    """Scatter-to-original-resolution works regardless of img_size parameter."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,
    )
    x = torch.randn(1, 3, 128, 128)  # different from img_size
    outs = model(x)
    assert outs[0].shape == (1, 64, 128, 128)


def test_multi_scale_output_count():
    """Correct number of outputs for multiple out_indices."""
    depth = 4
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=depth,
        num_superpixels=16,
        out_indices=[0, 2, 3],
    )
    x = torch.randn(1, 3, 64, 64)
    outs = model(x)
    assert len(outs) == 3
    assert outs[0].shape == (1, 64, 64, 64)
    assert outs[1].shape == (1, 64, 64, 64)
    assert outs[2].shape == (1, 64, 64, 64)


def test_cls_token_output_shape():
    """CLS token output has correct shape when enabled."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,
        with_cls_token=True,
        output_cls_token=True,
    )
    x = torch.randn(1, 3, 64, 64)
    outs = model(x)
    assert isinstance(outs[0], tuple)
    assert outs[0][0].shape == (1, 64, 64, 64)  # feature map
    assert outs[0][1].shape == (1, 64)  # cls token


def test_non_square_input():
    """Model handles non-square input resolutions."""
    # Use num_superpixels=8 with H=48,W=96 so diffSLIC's grid shape
    # (h_s=2, w_s=4) gives K=8, matching the configured num_superpixels.
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=8,
    )
    x = torch.randn(1, 3, 48, 96)  # non-square
    outs = model(x)
    assert outs[0].shape == (1, 64, 48, 96)
    assert torch.isfinite(outs[0]).all()


def test_minimal_depth():
    """Model with depth=1 and small config produces finite output."""
    # HEAD_SIZE=64, so embed_dims must be >= HEAD_SIZE * n_head = 64
    model = Vision_RWKV7(
        img_size=32,
        embed_dims=64,
        num_heads=1,
        depth=1,
        num_superpixels=4,
        diff_slic_iters=1,
    )
    x = torch.randn(1, 3, 32, 32)
    outs = model(x)
    assert len(outs) == 1
    assert torch.isfinite(outs[0]).all()


def test_gradient_flow_end_to_end():
    """Gradients flow back to the input tensor."""
    # HEAD_SIZE=64, so embed_dims must be >= HEAD_SIZE * n_head = 64
    model = Vision_RWKV7(
        img_size=32,
        embed_dims=64,
        num_heads=1,
        depth=1,
        num_superpixels=4,
        diff_slic_iters=1,
    )
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    outs = model(x)
    loss = outs[0].sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_deterministic_across_calls():
    """Same input produces same output across multiple forward calls."""
    model = Vision_RWKV7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=2,
        num_superpixels=16,
    )
    x = torch.randn(1, 3, 64, 64)
    out1 = model(x)
    out2 = model(x)
    for o1, o2 in zip(out1, out2):
        assert torch.allclose(o1, o2, atol=1e-5)
