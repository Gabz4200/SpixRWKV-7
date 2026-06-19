"""Tests for colors.py and gamut.py (Oklab and sRGB gamut clipping)."""

import torch
import pytest
from functools import partial

from spixrwkv7.data.colors import (
    _cbrt,
    from_srgb_to_linear_rgb,
    from_linear_rgb_to_srgb,
    from_linear_rgb_to_oklab,
    from_oklab_to_linear_rgb,
)
from spixrwkv7.data.gamut import (
    _safe_halley_denom,
    _compute_max_saturation,
    _find_cusp,
    _in_gamut_mask,
    gamut_clip_preserve_chroma,
    gamut_clip_project_to_0_5,
    gamut_clip_project_to_L_cusp,
    gamut_clip_adaptive_L0_0_5,
    gamut_clip_adaptive_L0_L_cusp,
)

# =========================================================================
# Helper Configurations
# =========================================================================
# Using functools.partial to bind the alpha argument for adaptive methods
CLIP_METHODS = [
    gamut_clip_preserve_chroma,
    gamut_clip_project_to_0_5,
    gamut_clip_project_to_L_cusp,
    partial(gamut_clip_adaptive_L0_0_5, alpha=0.05),
    partial(gamut_clip_adaptive_L0_L_cusp, alpha=0.05),
]


# =========================================================================
# colors.py Tests
# =========================================================================
def test_cbrt_correctness_and_gradients():
    """Verify _cbrt handles negatives correctly and maintains gradient flow."""
    x = torch.tensor([-8.0, -1.0, 0.0, 1.0, 8.0], requires_grad=True)
    y = _cbrt(x)
    expected = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    assert torch.allclose(y, expected, atol=1e-5)

    # Check gradients (standard torch.pow(x, 1/3) yields NaN gradients at x=0)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()


def test_srgb_linear_roundtrip():
    """Verify sRGB <-> Linear RGB conversion is perfectly invertible."""
    srgb = torch.rand(2, 3, 16, 16)
    linear = from_srgb_to_linear_rgb(srgb)
    srgb_recon = from_linear_rgb_to_srgb(linear)
    assert torch.allclose(srgb, srgb_recon, atol=1e-5)


def test_oklab_linear_roundtrip():
    """Verify Linear RGB <-> OkLAB roundtrip for both in-gamut and out-of-gamut values."""
    # In-gamut
    linear_in = torch.rand(2, 3, 16, 16)
    oklab = from_linear_rgb_to_oklab(linear_in)
    linear_recon = from_oklab_to_linear_rgb(oklab)
    assert torch.allclose(linear_in, linear_recon, atol=1e-4)

    # Out-of-gamut (negative values)
    linear_out = torch.randn(2, 3, 16, 16)
    oklab_out = from_linear_rgb_to_oklab(linear_out)
    linear_recon_out = from_oklab_to_linear_rgb(oklab_out)
    assert torch.allclose(linear_out, linear_recon_out, atol=1e-4)


def test_oklab_gradient_flow_negative():
    """Ensure gradients flow through OkLAB conversions even for negative RGB values."""
    linear = torch.tensor([[-1.0, 0.5, 0.2]], requires_grad=True).reshape(1, 3, 1, 1)
    linear.retain_grad()
    oklab = from_linear_rgb_to_oklab(linear)
    loss = oklab.sum()
    loss.backward()

    assert linear.grad is not None
    assert torch.isfinite(linear.grad).all()
    assert not torch.isnan(linear.grad).any()


# =========================================================================
# gamut.py Internal Helpers Tests
# =========================================================================
def test_safe_halley_denom():
    """Verify Halley's method denominator is clamped away from zero."""
    d = torch.tensor([0.0, 1e-40, -1e-40, 0.5, -0.5])
    safe_d = _safe_halley_denom(d)

    eps = torch.finfo(d.dtype).eps
    assert (safe_d.abs() >= eps).all()
    # Non-zero values should remain unchanged
    assert torch.allclose(safe_d[3:], d[3:])


def test_in_gamut_mask():
    """Verify the in-gamut boolean mask correctly identifies [0, 1] boundaries."""
    rgb = torch.zeros(1, 3, 2, 2)
    rgb[0, :, 0, 0] = 0.5  # In gamut
    rgb[0, :, 0, 1] = 1.5  # Out of gamut (>1)
    rgb[0, :, 1, 0] = -0.1  # Out of gamut (<0)
    rgb[0, :, 1, 1] = 0.8  # In gamut

    mask = _in_gamut_mask(rgb)
    expected = torch.tensor([[[[True, False], [False, True]]]])
    assert torch.equal(mask, expected)


def test_compute_max_saturation_and_cusp():
    """Verify max saturation and cusp calculations are finite and physically plausible."""
    # Normalized a, b across the hue circle
    theta = torch.linspace(0, 2 * 3.14159265, 100).view(1, 1, 100, 1)
    a = torch.cos(theta)
    b = torch.sin(theta)

    S = _compute_max_saturation(a, b)
    assert torch.isfinite(S).all()
    assert (S > 0).all()

    L_cusp, C_cusp = _find_cusp(a, b)
    assert torch.isfinite(L_cusp).all()
    assert torch.isfinite(C_cusp).all()
    assert (L_cusp >= -1e-4).all() and (L_cusp <= 1.0 + 1e-4).all()
    assert (C_cusp >= -1e-4).all()


# =========================================================================
# gamut.py Public API Tests
# =========================================================================
@pytest.mark.parametrize("clip_func", CLIP_METHODS)
def test_gamut_clip_in_gamut_unchanged(clip_func):
    """Verify that pixels already in the sRGB gamut are returned completely unchanged."""
    torch.manual_seed(42)
    # Strictly in-gamut linear RGB
    linear_rgb = torch.rand(2, 3, 16, 16) * 0.8 + 0.1
    clipped = clip_func(linear_rgb)
    assert torch.allclose(linear_rgb, clipped, atol=1e-5)


@pytest.mark.parametrize("clip_func", CLIP_METHODS)
def test_gamut_clip_out_of_gamut_clipped(clip_func):
    """Verify that out-of-gamut pixels are successfully projected into the [0, 1] range."""
    torch.manual_seed(42)
    linear_rgb = torch.randn(2, 3, 16, 16) * 2.0
    clipped = clip_func(linear_rgb)

    # Allow a tiny tolerance for floating point inaccuracies
    assert clipped.min() >= -1e-4, f"Min value {clipped.min()} is below 0"
    assert clipped.max() <= 1.0 + 1e-4, f"Max value {clipped.max()} is above 1"


def test_gamut_clip_gradient_flow():
    """Verify that gradients flow end-to-end through the gamut clipping process."""
    linear_rgb = (torch.randn(1, 3, 16, 16) * 2.0).requires_grad_(True)
    clipped = gamut_clip_preserve_chroma(linear_rgb)
    loss = clipped.sum()
    loss.backward()

    assert linear_rgb.grad is not None
    assert torch.isfinite(linear_rgb.grad).all()
    assert not torch.isnan(linear_rgb.grad).any()


def test_gamut_clip_extreme_values():
    """Ensure no NaNs or Infs are produced when processing extreme pixel values."""
    torch.manual_seed(123)
    linear_rgb = torch.randn(1, 3, 8, 8) * 100.0

    for clip_func in CLIP_METHODS:
        clipped = clip_func(linear_rgb)
        assert torch.isfinite(clipped).all(), (
            f"{clip_func.func.__name__} produced non-finite values"
        )


def test_gamut_clip_batch_consistency():
    """Verify that processing a batch yields the exact same results as processing items individually."""
    torch.manual_seed(0)
    linear_rgb = torch.randn(1, 3, 16, 16) * 2.0
    batch_rgb = torch.cat([linear_rgb, linear_rgb], dim=0)

    single_out = gamut_clip_preserve_chroma(linear_rgb)
    batch_out = gamut_clip_preserve_chroma(batch_rgb)

    assert torch.allclose(batch_out[0], single_out[0], atol=1e-5)
    assert torch.allclose(batch_out[1], single_out[0], atol=1e-5)


@pytest.mark.parametrize("clip_func", CLIP_METHODS)
def test_gamut_clip_rgba_support(clip_func):
    """Verify that gamut clipping correctly handles and preserves the alpha channel."""
    torch.manual_seed(42)
    rgba = torch.randn(2, 4, 16, 16) * 2.0
    # Ensure alpha is in [0, 1] for sanity, though clipping shouldn't touch it
    rgba[:, 3, :, :] = torch.rand(2, 16, 16)

    clipped = clip_func(rgba)

    assert clipped.shape == rgba.shape
    # Alpha should be identical
    assert torch.allclose(clipped[:, 3, :, :], rgba[:, 3, :, :], atol=1e-7)
    # RGB should be clipped to [0, 1]
    rgb_clipped = clipped[:, 0:3, :, :]
    assert rgb_clipped.min() >= -1e-3
    assert rgb_clipped.max() <= 1.0 + 1e-3
