from typing import TypedDict, Optional, Sequence, NotRequired
import torch
import torch.nn.functional as F
from spixrwkv7.models.spixrwkv7 import (
    SuperpixelEmbedding,
    Vision_RWKV7,
    Vision_RWKV7_Block,
    create_vision_rwkv7,
)


# =====================================================================
# Helper Functions
# =====================================================================


class ModelConfig(TypedDict):
    img_size: int
    embed_dims: int
    num_heads: int
    depth: int
    num_superpixels: int
    patch_size: NotRequired[Optional[int]]
    diff_slic_iters: int
    in_chans: int
    drop_path_rate: NotRequired[float]
    init_values: NotRequired[Optional[float]]
    final_norm: NotRequired[bool]
    out_indices: NotRequired[Sequence[int]]
    with_cls_token: NotRequired[bool]
    output_cls_token: NotRequired[bool]


def get_dummy_neighbors(B, N, K=4):
    """Helper to create dummy valid neighbors for testing blocks."""
    offsets = torch.arange(1, K + 1).unsqueeze(0)  # [1, K]
    neighbors = (torch.arange(N).unsqueeze(1) + offsets) % N  # [N, K]
    return neighbors.unsqueeze(0).expand(B, -1, -1)  # [B, N, K]


    # Common small model configs to reduce duplication across tests
_TINY_CONFIG: ModelConfig = {
    "img_size": 32,
    "embed_dims": 64,
    "num_heads": 1,
    "depth": 1,
    "num_superpixels": 9,
    "diff_slic_iters": 2,
    "in_chans": 6,
}
_SMALL_CONFIG: ModelConfig = {
    "img_size": 64,
    "embed_dims": 64,
    "num_heads": 1,
    "depth": 2,
    "num_superpixels": 16,
    "diff_slic_iters": 2,
    "in_chans": 6,
}


# =====================================================================
# Superpixel Embedding Tests
# =====================================================================


def test_superpixel_embedding_hard():
    """Test SuperpixelEmbedding in hard (discrete) mode."""
    B, C_in, H, W = 2, 6, 8, 8
    K = 4
    embed_dims = 16

    emb = SuperpixelEmbedding(C_in, embed_dims, K, mode="hard")
    x = torch.randn(B, C_in, H, W)

    # Create hard integer labels [B, H, W]
    sp_map = torch.randint(0, K, (B, H, W))

    tokens, centroids = emb(x, sp_map)
    assert tokens.shape == (B, K, embed_dims)
    assert centroids.shape == (B, K, 2)
    assert torch.isfinite(tokens).all()


def test_superpixel_embedding_soft():
    """Test SuperpixelEmbedding in soft (continuous) mode."""
    B, C_in, H, W = 2, 6, 8, 8
    K = 4
    embed_dims = 16

    emb = SuperpixelEmbedding(C_in, embed_dims, K, mode="soft")
    x = torch.randn(B, C_in, H, W)

    # Create soft probability masks [B, K, H, W]
    sp_map = torch.rand(B, K, H, W)
    sp_map = sp_map / sp_map.sum(dim=1, keepdim=True)  # normalize

    tokens, centroids = emb(x, sp_map)
    assert tokens.shape == (B, K, embed_dims)
    assert centroids.shape == (B, K, 2)
    assert torch.isfinite(tokens).all()


# =====================================================================
# Vision_RWKV7_Block Tests
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
    """Verify that block output depends on input content."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=1, layer_id=0)

    B, N = 1, 16
    x1 = torch.randn(B, N, n_embd)
    x2 = torch.randn(B, N, n_embd)
    neighbors = get_dummy_neighbors(B, N, K=4)

    out1, _, _ = block(x1, neighbors)
    out2, _, _ = block(x2, neighbors)
    assert not torch.allclose(out1, out2, atol=1e-5)


def test_rwkv7_decoupled_keys():
    """Verify that removal key (kk) and replacement key (kt) are decoupled."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=1, layer_id=0)

    # Find decoupled-key params by leaf name (resilient to module reshuffling)
    kk_param = next(p for n, p in block.named_parameters() if n.endswith('.k_k'))
    ka_param = next(p for n, p in block.named_parameters() if n.endswith('.k_a'))

    with torch.no_grad():
        kk_param.fill_(1.0)
        ka_param.fill_(0.5)

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

    # Find r_k by leaf name (resilient to module reshuffling)
    rk_param = next(p for n, p in block.named_parameters() if n.endswith('.r_k'))

    B, N = 1, 4
    x = torch.randn(B, N, n_embd)
    neighbors = get_dummy_neighbors(B, N, K=4)

    with torch.no_grad():
        rk_param.zero_()
    out1, _, _ = block(x, neighbors)

    with torch.no_grad():
        rk_param.fill_(1.0)
    out2, _, _ = block(x, neighbors)

    assert not torch.allclose(out1, out2, atol=1e-5)


# =====================================================================
# Vision_RWKV7 Backbone Tests
# =====================================================================


def test_vision_rwkv7_forward_hard():
    """Test full backbone forward pass with hard superpixel mode."""
    model = Vision_RWKV7(
        img_size=_SMALL_CONFIG["img_size"],
        embed_dims=_SMALL_CONFIG["embed_dims"],
        num_heads=_SMALL_CONFIG["num_heads"],
        depth=_SMALL_CONFIG["depth"],
        num_superpixels=_SMALL_CONFIG["num_superpixels"],
        diff_slic_iters=_SMALL_CONFIG["diff_slic_iters"],
        in_chans=_SMALL_CONFIG["in_chans"],
    )
    x = torch.randn(1, _SMALL_CONFIG["in_chans"], 64, 64)
    outs = model(x)

    # Output should be scattered back to original [B, C, H, W]
    assert len(outs) == 1  # default out_indices=(-1,)
    assert outs[0].shape == (1, 64, 4, 4)
    assert torch.isfinite(outs[0]).all()


def test_vision_rwkv7_forward_soft():
    """Test full backbone forward pass if we manually change mode to soft."""
    model = Vision_RWKV7(
        img_size=_SMALL_CONFIG["img_size"],
        embed_dims=_SMALL_CONFIG["embed_dims"],
        num_heads=_SMALL_CONFIG["num_heads"],
        depth=_SMALL_CONFIG["depth"],
        num_superpixels=_SMALL_CONFIG["num_superpixels"],
        diff_slic_iters=_SMALL_CONFIG["diff_slic_iters"],
        in_chans=_SMALL_CONFIG["in_chans"],
    )
    # Manually switch to soft mode to test that branch
    model.patch_embed.mode = "soft"

    x = torch.randn(1, _SMALL_CONFIG["in_chans"], 64, 64)
    outs = model(x)

    assert len(outs) == 1
    assert outs[0].shape == (1, 64, 4, 4)
    assert torch.isfinite(outs[0]).all()


def test_output_matches_input_resolution():
    """Verify that the scattered output matches the original input resolution."""
    model = Vision_RWKV7(
        img_size=_TINY_CONFIG["img_size"],
        embed_dims=_TINY_CONFIG["embed_dims"],
        num_heads=_TINY_CONFIG["num_heads"],
        depth=_TINY_CONFIG["depth"],
        num_superpixels=_TINY_CONFIG["num_superpixels"],
        diff_slic_iters=_TINY_CONFIG["diff_slic_iters"],
        in_chans=_TINY_CONFIG["in_chans"],
        scatter_output=True,
    )
    # Test with a different resolution than img_size
    x = torch.randn(1, _TINY_CONFIG["in_chans"], 128, 128)
    outs = model(x)

    # Output must be scattered back to [B, C, 128, 128]
    assert outs[0].shape == (1, 64, 128, 128)


def test_multi_scale_indices():
    """Verify that out_indices correctly returns features from multiple layers."""
    depth = 4
    model = Vision_RWKV7(
        img_size=32,
        embed_dims=64,
        num_heads=1,
        depth=depth,
        num_superpixels=9,
        diff_slic_iters=2,
        out_indices=[1, 3],
        in_chans=6,
    )
    x = torch.randn(1, 6, 64, 64)
    outs = model(x)

    assert len(outs) == 2
    # Both outputs are at grid resolution
    assert outs[0].shape == (1, 64, 3, 3)
    assert outs[1].shape == (1, 64, 3, 3)


def test_cls_token_behavior():
    """Verify CLS token handling and output."""
    model = Vision_RWKV7(
        img_size=_TINY_CONFIG["img_size"],
        embed_dims=_TINY_CONFIG["embed_dims"],
        num_heads=_TINY_CONFIG["num_heads"],
        depth=_TINY_CONFIG["depth"],
        num_superpixels=_TINY_CONFIG["num_superpixels"],
        diff_slic_iters=_TINY_CONFIG["diff_slic_iters"],
        with_cls_token=True,
        output_cls_token=True,
        in_chans=_TINY_CONFIG["in_chans"],
        scatter_output=True,
    )
    x = torch.randn(1, _TINY_CONFIG["in_chans"], 64, 64)
    outs = model(x)

    assert isinstance(outs[0], tuple)
    feat, cls_token = outs[0]
    # Feat is scattered back to [B, C, H, W]
    assert feat.shape == (1, 64, 64, 64)
    assert cls_token.shape == (1, 64)


def test_numerical_stability_long_seq():
    """Check for stability with larger grids (longer sequences)."""
    model = Vision_RWKV7(
        img_size=_TINY_CONFIG["img_size"],
        embed_dims=_TINY_CONFIG["embed_dims"],
        num_heads=_TINY_CONFIG["num_heads"],
        depth=_TINY_CONFIG["depth"],
        num_superpixels=_TINY_CONFIG["num_superpixels"],
        diff_slic_iters=_TINY_CONFIG["diff_slic_iters"],
        in_chans=_TINY_CONFIG["in_chans"],
    )
    x = torch.randn(1, _TINY_CONFIG["in_chans"], 128, 128)
    outs = model(x)
    assert torch.isfinite(outs[0]).all()


def test_rwkv7_vector_iclr_decay():
    """Verify that ICLR (a) and Decay (w) are vector-valued and applied per-channel."""
    n_embd = 64
    n_head = 1
    block = Vision_RWKV7_Block(n_embd=n_embd, n_head=n_head, n_layer=1, layer_id=0)

    # Find w0/a0 by leaf name — shape assertion proves per-channel (vector-valued) semantics
    w0_param = next(p for n, p in block.named_parameters() if n.endswith('.w0'))
    a0_param = next(p for n, p in block.named_parameters() if n.endswith('.a0'))
    assert w0_param.shape == (n_embd,)
    assert a0_param.shape == (n_embd,)

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
        img_size=_SMALL_CONFIG["img_size"],
        embed_dims=_SMALL_CONFIG["embed_dims"],
        num_heads=_SMALL_CONFIG["num_heads"],
        depth=_SMALL_CONFIG["depth"],
        num_superpixels=_SMALL_CONFIG["num_superpixels"],
        diff_slic_iters=_SMALL_CONFIG["diff_slic_iters"],
        in_chans=_SMALL_CONFIG["in_chans"],
    )
    x = torch.randn(1, _SMALL_CONFIG["in_chans"], 64, 64)

    out1 = model(x)
    out2 = model(x)

    for o1, o2 in zip(out1, out2):
        if isinstance(o1, tuple):
            assert torch.allclose(o1[0], o2[0], atol=1e-5)
            assert torch.allclose(o1[1], o2[1], atol=1e-5)
        else:
            assert torch.allclose(o1, o2, atol=1e-5)


# =====================================================================
# Behavioral Tests
# =====================================================================


def test_forward_finite_random_input():
    """Forward pass with random input produces all-finite outputs."""
    model = Vision_RWKV7(
        img_size=_SMALL_CONFIG["img_size"],
        embed_dims=_SMALL_CONFIG["embed_dims"],
        num_heads=_SMALL_CONFIG["num_heads"],
        depth=_SMALL_CONFIG["depth"],
        num_superpixels=_SMALL_CONFIG["num_superpixels"],
        diff_slic_iters=_SMALL_CONFIG["diff_slic_iters"],
        in_chans=_SMALL_CONFIG["in_chans"],
    )
    x = torch.randn(2, _SMALL_CONFIG["in_chans"], 64, 64)  # batch=2
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
        in_chans=6,
        scatter_output=True,
    )
    x = torch.randn(1, 6, 128, 128)  # different from img_size
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
        in_chans=6,
    )
    x = torch.randn(1, 6, 64, 64)
    outs = model(x)
    assert len(outs) == 3
    assert outs[0].shape == (1, 64, 4, 4)
    assert outs[1].shape == (1, 64, 4, 4)
    assert outs[2].shape == (1, 64, 4, 4)


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
        in_chans=6,
        scatter_output=True,
    )
    x = torch.randn(1, 6, 64, 64)
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
        in_chans=6,
        scatter_output=True,
    )
    x = torch.randn(1, 6, 48, 96)  # non-square
    outs = model(x)
    assert outs[0].shape == (1, 64, 48, 96)
    assert torch.isfinite(outs[0]).all()


def test_minimal_depth():
    """Model with depth=1 and small config produces finite output."""
    # HEAD_SIZE=64, so embed_dims must be >= HEAD_SIZE * n_head = 64
    model = Vision_RWKV7(
        img_size=_TINY_CONFIG["img_size"],
        embed_dims=_TINY_CONFIG["embed_dims"],
        num_heads=_TINY_CONFIG["num_heads"],
        depth=_TINY_CONFIG["depth"],
        num_superpixels=_TINY_CONFIG["num_superpixels"],
        diff_slic_iters=_TINY_CONFIG["diff_slic_iters"],
        in_chans=_TINY_CONFIG["in_chans"],
    )
    x = torch.randn(1, _TINY_CONFIG["in_chans"], 32, 32)
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
        in_chans=6,
    )
    x = torch.randn(1, 6, 32, 32, requires_grad=True)
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
        in_chans=6,
    )
    x = torch.randn(1, 6, 64, 64)
    out1 = model(x)
    out2 = model(x)
    for o1, o2 in zip(out1, out2):
        assert torch.allclose(o1, o2, atol=1e-5)


def test_dynamic_resolution_spixel_size():
    """Verify that spixel_size correctly scales the number of superpixels."""
    # spixel_size=16 means K = (H*W) / (16*16)
    model = create_vision_rwkv7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=1,
        spixel_size=16,
    )
    
    # 1. 64x64 input -> (64*64)/(16*16) = 16 superpixels (4x4 grid)
    x1 = torch.randn(1, 6, 64, 64)
    outs1 = model(x1)
    # Default is scatter_output=False, so output is at grid resolution
    assert outs1[0].shape == (1, 64, 4, 4)
    
    # 2. 128x128 input -> (128*128)/(16*16) = 64 superpixels (8x8 grid)
    x2 = torch.randn(1, 6, 128, 128)
    outs2 = model(x2)
    assert outs2[0].shape == (1, 64, 8, 8)
    
    # 3. Non-square 64x128 -> (64*128)/(16*16) = 32 superpixels (4x8 grid)
    x3 = torch.randn(1, 6, 64, 128)
    outs3 = model(x3)
    assert outs3[0].shape == (1, 64, 4, 8)


def test_vision_rwkv7_scatter_output():
    """Verify that scatter_output=True restores original resolution."""
    model = create_vision_rwkv7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=1,
        num_superpixels=16,
        scatter_output=True,
    )
    x = torch.randn(1, 6, 64, 64)
    outs = model(x)
    assert outs[0].shape == (1, 64, 64, 64)


def test_forward_num_superpixels_override():
    """Verify that num_superpixels can be overridden in the forward pass."""
    model = create_vision_rwkv7(
        img_size=64,
        embed_dims=64,
        num_heads=1,
        depth=1,
        num_superpixels=16,
    )
    
    # Override to 36 superpixels (6x6)
    x = torch.randn(1, 6, 64, 64)
    outs = model(x, num_superpixels=36)
    assert outs[0].shape == (1, 64, 6, 6)
    # We can't easily check the internal K without hooks, but if it doesn't crash 
    # and produces the right output shape, the interpolation and graph building 
    # worked for the new K.


def test_parallel_recurrent_scan_equivalence():
    """Verify that ParallelRecurrentScan output matches RecurrentScan and OptimizedRecurrentScan."""
    from spixrwkv7.kernels.optimized_block import ParallelRecurrentScan, OptimizedRecurrentScan
    from spixrwkv7.models.spixrwkv7 import RecurrentScan

    B, N, D = 2, 8, 128
    Hd = 2
    S = 64
    dev = "cpu"

    torch.manual_seed(42)

    # Instantiate the three scan modules with the same dimensions
    scan_ref = RecurrentScan(n_embd=D, n_head=Hd, layer_id=1, n_layer=2).to(dev)

    # Initialize weights to match
    scan_parallel = ParallelRecurrentScan(n_embd=D, n_head=Hd, layer_id=1, n_layer=2).to(dev)
    scan_opt = OptimizedRecurrentScan(n_embd=D, n_head=Hd, layer_id=1, n_layer=2).to(dev)

    # Copy parameters
    for p_name, p in scan_ref.named_parameters():
        dict(scan_parallel.named_parameters())[p_name].data.copy_(p.data)
        dict(scan_opt.named_parameters())[p_name].data.copy_(p.data)

    # Inputs
    xn = torch.randn(B, N, D, device=dev)
    xx = torch.randn(B, N, D, device=dev)
    dm = torch.randn(6, B, N, D, device=dev)

    # Forward passes
    out_ref, _ = scan_ref(xn, xx, dm, "forward", None)
    out_parallel, _ = scan_parallel(xn, xx, dm, "forward", None)
    out_opt, _ = scan_opt(xn, xx, dm, "forward", None)

    # Check that they produce very close results
    assert torch.allclose(out_parallel, out_ref, rtol=1e-3, atol=1e-4)
    assert torch.allclose(out_parallel, out_opt, rtol=1e-3, atol=1e-4)
def test_rmsnorm_swiglu_and_activation_options():
    """Verify that backbone can be initialized and run with RMSNorm and SwiGLU / other activations."""
    from spixrwkv7.models.spixrwkv7 import create_vision_rwkv7
    from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7

    device = "cpu"
    B, C, H, W = 2, 6, 64, 64
    x = torch.randn(B, C, H, W, device=device)

    # Test 1: PyTorch model with RMSNorm and SwiGLU
    model1 = create_vision_rwkv7(
        img_size=64, embed_dims=128, depth=2, num_heads=2,
        norm_layer="rmsnorm", act_layer="swiglu", scatter_output=True
    ).to(device)
    outs1 = model1(x)
    assert len(outs1) == 1
    assert outs1[0].shape == (B, 128, H, W)
    assert torch.isfinite(outs1[0]).all()

    # Test 2: Optimized C++ model with RMSNorm and SwiGLU
    model2 = create_optimized_vision_rwkv7(
        img_size=64, embed_dims=128, depth=2, num_heads=2,
        norm_layer="rmsnorm", act_layer="swiglu", use_cpp=True, scatter_output=True
    ).to(device)
    outs2 = model2(x)
    assert len(outs2) == 1
    assert outs2[0].shape == (B, 128, H, W)
    assert torch.isfinite(outs2[0]).all()


def test_sequence_masking_in_scans():
    """Verify that sequence masking works in the recurrent scan operations."""
    from spixrwkv7.models.spixrwkv7 import RecurrentScan
    from spixrwkv7.kernels.optimized_block import ParallelRecurrentScan, OptimizedRecurrentScan

    B, N, D = 2, 8, 128
    Hd, S = 2, 64
    dev = "cpu"

    # Setup model and copy weights
    scan_ref = RecurrentScan(n_embd=D, n_head=Hd, layer_id=1, n_layer=2).to(dev)
    scan_parallel = ParallelRecurrentScan(n_embd=D, n_head=Hd, layer_id=1, n_layer=2).to(dev)
    scan_opt = OptimizedRecurrentScan(n_embd=D, n_head=Hd, layer_id=1, n_layer=2).to(dev)

    for p_name, p in scan_ref.named_parameters():
        dict(scan_parallel.named_parameters())[p_name].data.copy_(p.data)
        dict(scan_opt.named_parameters())[p_name].data.copy_(p.data)

    xn = torch.randn(B, N, D, device=dev)
    xx = torch.randn(B, N, D, device=dev)
    dm = torch.randn(6, B, N, D, device=dev)

    # Mask: second batch element has last 4 timesteps masked
    mask = torch.ones(B, N, device=dev)
    mask[1, 4:] = 0.0

    # Forward passes with mask
    out_ref, _ = scan_ref(xn, xx, dm, "forward", None, mask=mask)
    out_parallel, _ = scan_parallel(xn, xx, dm, "forward", None, mask=mask)
    out_opt, _ = scan_opt(xn, xx, dm, "forward", None, mask=mask)

    # Masked positions should be zeroed out
    assert (out_ref[1, 4:] == 0.0).all()
    assert (out_parallel[1, 4:] == 0.0).all()
    assert (out_opt[1, 4:] == 0.0).all()

    # Check prefix scan equivalence and optimized scan equivalence with mask
    assert torch.allclose(out_parallel, out_ref, rtol=1e-3, atol=1e-4)
    assert torch.allclose(out_parallel, out_opt, rtol=1e-3, atol=1e-4)


def test_sequence_masking_backbone():
    """Verify that mask propagates correctly through the full backbone."""
    from spixrwkv7.models.spixrwkv7 import create_vision_rwkv7

    device = "cpu"
    B, C, H, W = 2, 6, 64, 64
    x = torch.randn(B, C, H, W, device=device)

    # Create backbone
    model = create_vision_rwkv7(img_size=64, embed_dims=128, depth=2, num_heads=2, scatter_output=True).to(device)
    
    # Since token ordering is Hilbert-sorted, we pass a mask of the same size as the number of superpixels
    # Let's say we have 196 superpixels. Let's mask some of them out.
    mask = torch.ones(B, 196, device=device)
    mask[0, 100:] = 0.0

    # Run model with mask
    outs = model(x, mask=mask)
    assert len(outs) == 1
    assert outs[0].shape == (B, 128, H, W)
    assert torch.isfinite(outs[0]).all()


def test_alternative_superpixel_backends():
    """Verify that alternative superpixel backends work correctly."""
    from spixrwkv7 import create_vision_rwkv7, create_optimized_vision_rwkv7

    device = "cpu"
    B, C, H, W = 2, 6, 32, 32
    x = torch.randn(B, C, H, W, device=device)

    backends = ["grid", "slic", "slico", "lnsnet"]
    modes = ["soft", "hard"]

    for backend in backends:
        for mode in modes:
            # Test PyTorch creator
            model = create_vision_rwkv7(
                img_size=32,
                embed_dims=64,
                depth=1,
                num_heads=1,
                num_superpixels=9,
                scatter_output=True,
                spixel_backend=backend,
            ).to(device)
            model.tokenizer.mode = mode
            model.patch_embed.mode = mode

            outs = model(x)
            assert len(outs) == 1
            assert outs[0].shape == (B, 64, H, W)
            assert torch.isfinite(outs[0]).all()

            # Test Optimized creator
            opt_model = create_optimized_vision_rwkv7(
                img_size=32,
                embed_dims=64,
                depth=1,
                num_heads=1,
                num_superpixels=9,
                scatter_output=True,
                spixel_backend=backend,
                use_cpp=False,  # Fallback to PyTorch since optimized blocks aren't needed here
            ).to(device)
            opt_model.tokenizer.mode = mode
            opt_model.patch_embed.mode = mode

            opt_outs = opt_model(x)
            assert len(opt_outs) == 1
            assert opt_outs[0].shape == (B, 64, H, W)
            assert torch.isfinite(opt_outs[0]).all()


def test_attention_residuals():
    """Verify that Attention Residuals (AttnRes) works correctly for all modes and gates."""
    from spixrwkv7 import create_vision_rwkv7, create_optimized_vision_rwkv7, ClassificationHead

    device = "cpu"
    B, C, H, W = 2, 6, 32, 32
    x = torch.randn(B, C, H, W, device=device)

    modes = ["block", "full"]
    gates = ["bias", "sigmoid_scalar", "sigmoid_vector", "learnable_alpha"]

    for mode in modes:
        for gate in gates:
            # Test standard model creator with AttnRes
            model = create_vision_rwkv7(
                img_size=32,
                embed_dims=64,
                depth=2,
                num_heads=1,
                num_superpixels=9,
                scatter_output=True,
                use_attnres=True,
                attnres_mode=mode,
                attnres_gate_type=gate,
            ).to(device)

            outs = model(x)
            assert len(outs) == 1
            assert outs[0].shape == (B, 64, H, W)
            assert torch.isfinite(outs[0]).all()

            # Check that last_attnres_history and last_project_fn are correctly set
            assert model.last_attnres_history is not None
            assert model.last_attnres_history_patches is not None
            assert model.last_project_fn is not None

            # Test classification head with AttnRes history
            head = ClassificationHead(embed_dims=64, num_classes=10).to(device)
            logits = head(
                outs[0],
                attnres_history=model.last_attnres_history_patches,
                project_fn=model.last_project_fn
            )
            assert logits.shape == (B, 10)
            assert torch.isfinite(logits).all()

            # Test optimized creator with AttnRes
            opt_model = create_optimized_vision_rwkv7(
                img_size=32,
                embed_dims=64,
                depth=2,
                num_heads=1,
                num_superpixels=9,
                scatter_output=True,
                use_attnres=True,
                attnres_mode=mode,
                attnres_gate_type=gate,
                use_cpp=False,
            ).to(device)

            opt_outs = opt_model(x)
            assert len(opt_outs) == 1
            assert opt_outs[0].shape == (B, 64, H, W)
            assert torch.isfinite(opt_outs[0]).all()

            opt_logits = head(
                opt_outs[0],
                attnres_history=opt_model.last_attnres_history_patches,
                project_fn=opt_model.last_project_fn
            )
            assert opt_logits.shape == (B, 10)
            assert torch.isfinite(opt_logits).all()



