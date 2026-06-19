"""Regression tests for numerical stability and edge cases in Vision-RWKV-7."""

import torch
from spixrwkv7 import DiffSLIC, Vision_RWKV7


def test_diffslic_all_black_no_nan():
    """Regression test for zero-norm division NaN bug with all-black pixels."""
    x = torch.zeros(2, 3, 32, 32)
    slic = DiffSLIC(
        n_spixels=8, n_iter=3, tau=0.01, candidate_radius=1, normalize=True, stable=True
    )
    with torch.no_grad():
        clst_feats, p2s_assign, _ = slic(x)
    assert torch.isfinite(clst_feats).all()
    assert torch.isfinite(p2s_assign).all()


def test_diffslic_uniform_image_no_nan():
    """Regression test for NaN from softmax over all -inf with uniform image."""
    x = torch.ones(1, 3, 32, 32) * 0.5
    slic = DiffSLIC(
        n_spixels=8, n_iter=3, tau=0.01, candidate_radius=1, normalize=True, stable=True
    )
    with torch.no_grad():
        clst_feats, p2s_assign, _ = slic(x)
    assert torch.isfinite(clst_feats).all()
    assert torch.isfinite(p2s_assign).all()


def test_model_all_black_finite():
    """Verify full model forward pass with all-black input produces finite outputs."""
    x = torch.zeros(1, 3, 32, 32)
    model = Vision_RWKV7(
        img_size=32,
        embed_dims=64,
        num_heads=1,
        depth=1,
        num_superpixels=9,
        diff_slic_iters=2,
    )
    outs = model(x)
    for o in outs:
        assert torch.isfinite(o).all()


def test_model_extreme_hyperparams():
    """Test with very few and very many superpixels to ensure stability."""
    configs = [
        (9, 1),  # few superpixels (3x3 grid), min iters
        (100, 10),  # many superpixels (10x10 grid), many iters
    ]
    for n_spx, iters in configs:
        model = Vision_RWKV7(
            img_size=32,
            embed_dims=64,
            num_heads=1,
            depth=1,
            num_superpixels=n_spx,
            diff_slic_iters=iters,
        )
        x = torch.randn(1, 3, 32, 32)
        outs = model(x)
        for o in outs:
            assert torch.isfinite(o).all()


def test_diffslic_zero_norm_single_channel():
    """Regression test for zero-norm division with single-channel input."""
    x = torch.zeros(1, 1, 32, 32)
    slic = DiffSLIC(n_spixels=8, n_iter=2, tau=0.01, candidate_radius=1, normalize=True)
    with torch.no_grad():
        clst_feats, p2s_assign, _ = slic(x)
    assert torch.isfinite(clst_feats).all()


def test_model_extreme_values():
    """Forward pass with extreme pixel values should not break."""
    x = torch.randn(1, 3, 32, 32) * 100
    model = Vision_RWKV7(
        img_size=32,
        embed_dims=64,
        num_heads=1,
        depth=1,
        num_superpixels=9,
        diff_slic_iters=2,
    )
    outs = model(x)
    for o in outs:
        assert torch.isfinite(o).all()


def test_model_batch_consistency():
    """Two identical single-item batches should give same result as one batch of 2."""
    model = Vision_RWKV7(
        img_size=32,
        embed_dims=64,
        num_heads=1,
        depth=1,
        num_superpixels=9,
        diff_slic_iters=2,
    )
    x_single = torch.randn(1, 3, 32, 32)
    x_batch = torch.cat([x_single, x_single], dim=0)
    out_single = model(x_single)
    out_batch = model(x_batch)
    assert torch.allclose(out_batch[0][:1], out_single[0], atol=1e-4), (
        "Batch and single outputs differ"
    )
    assert torch.allclose(out_batch[0][0], out_batch[0][1], atol=1e-4), (
        "Batch items differ"
    )
