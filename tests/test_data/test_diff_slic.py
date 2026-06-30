"""Complete test and debug of diffSLIC to ensure it works on any image."""

import pytest
import torch

from spixrwkv7 import DiffSLIC, spixel_downsampling, spixel_upsampling


def test_diffslic_nan_safety_with_black_pixels():
    """Verify diffSLIC doesn't produce NaN with all-zero (black) pixels."""
    # All-black image - would trigger 0/0 NaN with unguarded normalize
    x = torch.zeros(1, 3, 64, 64)
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=5,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    with torch.no_grad():
        clst_feats, p2s_assign, _ = diff_slic(x)
    assert not torch.isnan(clst_feats).any(), (
        "clst_feats has NaN from zero-norm division"
    )
    assert not torch.isnan(p2s_assign).any(), (
        "p2s_assign has NaN from zero-norm division"
    )
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"


@pytest.mark.parametrize(
    "n_spixels,n_iter,tau",
    [
        (49, 5, 0.01),
        (196, 10, 0.01),
        (400, 15, 0.005),
        (196, 20, 0.001),
    ],
)
def test_with_different_configs(n_spixels, n_iter, tau):
    """Run diffSLIC with different hyperparameter configurations."""
    test_img = torch.randn(1, 3, 224, 224) * 0.5
    diff_slic = DiffSLIC(
        n_spixels=n_spixels,
        n_iter=n_iter,
        tau=tau,
        candidate_radius=1,
        normalize=True,
    )
    with torch.no_grad():
        clst_feats, p2s_assign, _ = diff_slic(test_img)
    assert torch.isfinite(clst_feats).all()
    assert torch.isfinite(p2s_assign).all()


def test_diffslic_soft_assignment_probability():
    """Verify p2s_assign sums to 1 over candidate dimension (softmax probability)."""
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=5,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
        stable=True,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        _, p2s_assign, _ = diff_slic(x)
    row_sums = p2s_assign.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "p2s_assign rows should sum to 1 (softmax probability property)"
    )


def test_diffslic_output_shapes():
    """Check output tensor shapes for batch=2 with candidate_radius=1."""
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=5,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert clst_feats.shape[0] == 2 and clst_feats.shape[1] == 3, (
        "clst_feats batch or channel dim wrong"
    )
    assert clst_feats.shape[2] * clst_feats.shape[3] <= 16, (
        "clst_feats spatial product exceeds n_spixels"
    )
    assert p2s_assign.shape == (2, 9, 64, 64), (
        f"p2s_assign shape mismatch: {p2s_assign.shape}"
    )
    assert s2p_assign is not None, "s2p_assign should not be None with n_iter=5"


def test_diffslic_zero_iter():
    """Check s2p_assign is None and outputs are finite when n_iter=0."""
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=0,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert s2p_assign is None, "s2p_assign should be None when n_iter=0"
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"
    assert torch.isfinite(p2s_assign).all(), "p2s_assign has non-finite values"


def test_spixel_upsampling_shape():
    """Verify spixel_upsampling restores original spatial resolution."""
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=3,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, _ = diff_slic(x)
    upsampled = spixel_upsampling(clst_feats, p2s_assign, candidate_radius=1)
    assert upsampled.shape == (1, 3, 64, 64), (
        f"Expected (1, 3, 64, 64) but got {upsampled.shape}"
    )


def test_spixel_downsampling_shape():
    """Verify spixel_downsampling produces spixel-resolution output."""
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=3,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert s2p_assign is not None, "s2p_assign required for downsampling"
    downsampled = spixel_downsampling(x, s2p_assign, candidate_radius=1)
    assert downsampled.shape[-2:] == clst_feats.shape[-2:], (
        f"Expected spatial dims {clst_feats.shape[-2:]} but got {downsampled.shape[-2:]}"
    )


def test_diffslic_gradient_flow():
    """Verify gradients flow through the entire diffSLIC forward."""
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=3,
        tau=0.01,
        candidate_radius=1,
        normalize=False,
    )
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    clst_feats, p2s_assign, _ = diff_slic(x)
    loss = clst_feats.sum() + p2s_assign.sum()
    loss.backward()
    assert x.grad is not None, "x.grad is None — no gradient flowed"
    assert not x.grad.isnan().any(), "x.grad contains NaN"
    assert torch.isfinite(x.grad).all(), "x.grad has non-finite values"


def test_diffslic_single_superpixel():
    """Check behaviour with a single superpixel (n_spixels=1)."""
    diff_slic = DiffSLIC(
        n_spixels=1,
        n_iter=3,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        clst_feats, p2s_assign, _ = diff_slic(x)
    assert clst_feats.shape == (1, 3, 1, 1), (
        f"clst_feats shape mismatch: {clst_feats.shape}"
    )
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"
    assert torch.isfinite(p2s_assign).all(), "p2s_assign has non-finite values"


def test_diffslic_non_square_image():
    """Check diffSLIC works on non-square images (width != height)."""
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=3,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    x = torch.randn(1, 3, 32, 16)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"
    assert torch.isfinite(p2s_assign).all(), "p2s_assign has non-finite values"
    assert s2p_assign is None or torch.isfinite(s2p_assign).all(), (
        "s2p_assign has non-finite values"
    )
    assert clst_feats.shape[0] == 1 and clst_feats.shape[1] == 3, (
        "clst_feats batch or channel dim wrong"
    )
    assert p2s_assign.shape[0] == 1 and p2s_assign.shape[1] == 9, (
        "p2s_assign batch or candidate dim wrong"
    )
    assert p2s_assign.shape[-2:] == (32, 16), (
        f"p2s_assign spatial dims wrong: {p2s_assign.shape[-2:]} vs (32, 16)"
    )


def test_diffslic_cpp_backend_active():
    """Verify that the C++ backend is actually loaded and not silently falling back."""
    slic = DiffSLIC(n_spixels=100, use_cpp=True)
    assert slic._has_cpp is True
