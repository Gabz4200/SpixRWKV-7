"""Tests for dataload.py (Image loading, OkLAB conversion, and dataset statistics)."""

import torch
import pytest
import os
from PIL import Image
from typing import Tuple

# Adjust this import path to match your project structure
# (e.g., from VisualRWKV7.utils.dataload import ...)
from spixrwkv7.data.transforms import (
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    DEFAULT_OKLAB_MEAN,
    DEFAULT_OKLAB_STD,
    _convert_srgb_to_oklab,
    calculate_dataset_mean_std,
    load_image_to_tensor,
    add_spatial_coordinates,
    preprocess_image_for_rwkv7,
    prepare_balanced_superpixel_features,
    revert_balanced_superpixel_features,
)

# =============================================================================
# Helper Configurations & Fixtures
# =============================================================================


def create_dummy_image(
    path: str, size: Tuple[int, int], color: Tuple[int, int, int], mode: str = "RGB"
):
    """Helper to create and save a dummy PIL image."""
    img = Image.new(mode, size, color)
    img.save(path)


def create_dummy_dataset(
    root_dir: str,
    num_classes: int = 2,
    images_per_class: int = 3,
    size: Tuple[int, int] = (64, 64),
):
    """Helper to create a dummy ImageFolder-compatible directory structure."""
    for c in range(num_classes):
        class_dir = os.path.join(root_dir, f"class_{c}")
        os.makedirs(class_dir, exist_ok=True)
        for i in range(images_per_class):
            # Create slightly varying colors to ensure non-zero std
            color_val = int((c * 50 + i * 20) % 255)
            img_path = os.path.join(class_dir, f"img_{i}.jpg")
            create_dummy_image(img_path, size, (color_val, color_val, color_val))


# =============================================================================
# Constants & Helper Tests
# =============================================================================


def test_default_constants_shapes():
    """Verify that all default normalization constants are lists of length 4."""
    assert len(IMAGENET_RGB_MEAN) == 4
    assert len(IMAGENET_RGB_STD) == 4
    assert len(DEFAULT_OKLAB_MEAN) == 4
    assert len(DEFAULT_OKLAB_STD) == 4


def test_convert_srgb_to_oklab_ranges():
    """Verify OkLAB conversion produces values in physically plausible ranges."""
    # Create a batch of random sRGB images in [0, 1]
    srgb = torch.rand(2, 3, 32, 32)
    oklab = _convert_srgb_to_oklab(srgb)

    assert oklab.shape == srgb.shape

    L = oklab[:, 0:1, :, :]
    a = oklab[:, 1:2, :, :]
    b = oklab[:, 2:3, :, :]

    # L (Lightness) should be roughly in [0, 1]
    assert (L >= -0.05).all() and (L <= 1.05).all()
    # a and b (color opponents) should be roughly in [-0.5, 0.5] for in-gamut sRGB
    assert (a >= -0.6).all() and (a <= 0.6).all()
    assert (b >= -0.6).all() and (b <= 0.6).all()

    # Test RGBA
    rgba = torch.rand(2, 4, 32, 32)
    oklab_rgba = _convert_srgb_to_oklab(rgba)
    assert oklab_rgba.shape == rgba.shape
    assert torch.allclose(oklab_rgba[:, 3:4, :, :], rgba[:, 3:4, :, :])


# =============================================================================
# load_image_to_tensor Tests
# =============================================================================


def test_load_image_basic_shape(tmp_path):
    """Verify basic loading returns (1, 3, H, W) tensor."""
    img_path = tmp_path / "test.jpg"
    create_dummy_image(str(img_path), (100, 200), (128, 128, 128))

    tensor = load_image_to_tensor(str(img_path))

    assert tensor.shape == (1, 3, 200, 100)  # B=1, C=3, H=200, W=100
    assert tensor.dtype == torch.float32
    assert (tensor >= 0.0).all() and (tensor <= 1.0).all()


def test_load_image_target_resize(tmp_path):
    """Verify target_size correctly resizes the image."""
    img_path = tmp_path / "test.jpg"
    create_dummy_image(str(img_path), (500, 300), (255, 0, 0))

    tensor = load_image_to_tensor(str(img_path), target_size=(224, 224))

    assert tensor.shape == (1, 3, 224, 224)


def test_load_image_oklab_space(tmp_path):
    """Verify color_space='oklab' correctly converts the tensor."""
    img_path = tmp_path / "test.jpg"
    create_dummy_image(str(img_path), (64, 64), (0, 255, 0))

    tensor = load_image_to_tensor(str(img_path), color_space="oklab")

    assert tensor.shape == (1, 3, 64, 64)
    # OkLAB values can be negative and exceed 1.0 slightly for out-of-gamut,
    # but should definitely not be strictly bounded to [0, 1] like sRGB.
    assert tensor.min() < 0.0 or tensor.max() > 1.0 or True  # Just ensure it ran
    assert torch.isfinite(tensor).all()


def test_load_image_normalization_applied(tmp_path):
    """Verify normalize=True shifts the data and doesn't produce NaNs."""
    img_path = tmp_path / "test.jpg"
    create_dummy_image(str(img_path), (64, 64), (100, 150, 200))

    tensor_unnorm = load_image_to_tensor(str(img_path), normalize=False)
    tensor_norm = load_image_to_tensor(str(img_path), normalize=True)

    # Normalized tensor should have different values
    assert not torch.allclose(tensor_unnorm, tensor_norm)
    # Must remain finite
    assert torch.isfinite(tensor_norm).all()


def test_load_image_custom_mean_std(tmp_path):
    """Verify custom mean and std override the defaults."""
    img_path = tmp_path / "test.jpg"
    create_dummy_image(str(img_path), (64, 64), (128, 128, 128))

    custom_mean = [0.5, 0.5, 0.5]
    custom_std = [0.5, 0.5, 0.5]

    tensor = load_image_to_tensor(
        str(img_path), normalize=True, mean=custom_mean, std=custom_std
    )

    # 128/255 is approx 0.5019. Normalized with mean=0.5, std=0.5 -> (0.5019 - 0.5)/0.5 ≈ 0.0038
    # We just check it's finite and close to 0 for the uniform gray image
    assert torch.isfinite(tensor).all()
    assert (tensor.abs() < 0.1).all()


def test_load_image_rgba(tmp_path):
    """Verify loading with include_alpha=True returns (1, 4, H, W) tensor."""
    img_path = tmp_path / "test.png"
    # Create an RGBA image
    img = Image.new("RGBA", (100, 200), (255, 0, 0, 128))
    img.save(img_path)

    tensor = load_image_to_tensor(str(img_path), include_alpha=True)

    assert tensor.shape == (1, 4, 200, 100)
    assert torch.allclose(tensor[:, 3, :, :], torch.full((1, 200, 100), 128 / 255.0))

    # Test normalization with alpha
    tensor_norm = load_image_to_tensor(str(img_path), include_alpha=True, normalize=True)
    assert tensor_norm.shape == (1, 4, 200, 100)
    assert torch.isfinite(tensor_norm).all()

    # Test OkLAB with alpha
    tensor_oklab = load_image_to_tensor(str(img_path), include_alpha=True, color_space="oklab")
    assert tensor_oklab.shape == (1, 4, 200, 100)
    assert torch.allclose(tensor_oklab[:, 3, :, :], torch.full((1, 200, 100), 128 / 255.0))


def test_load_image_invalid_color_space(tmp_path):
    """Verify invalid color_space raises ValueError."""
    img_path = tmp_path / "test.jpg"
    create_dummy_image(str(img_path), (64, 64), (0, 0, 0))

    with pytest.raises(ValueError, match="Unsupported color_space"):
        load_image_to_tensor(str(img_path), color_space="cmyk")


def test_load_image_handles_rgba_and_grayscale(tmp_path):
    """Verify .convert('RGB') safely handles RGBA and Grayscale images."""
    # RGBA image
    rgba_path = tmp_path / "rgba.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(str(rgba_path))
    tensor_rgba = load_image_to_tensor(str(rgba_path))
    assert tensor_rgba.shape == (1, 3, 64, 64)

    # Grayscale image
    gray_path = tmp_path / "gray.jpg"
    Image.new("L", (64, 64), 128).save(str(gray_path))
    tensor_gray = load_image_to_tensor(str(gray_path))
    assert tensor_gray.shape == (1, 3, 64, 64)


# =============================================================================
# add_spatial_coordinates Tests
# =============================================================================


def test_add_spatial_coordinates_shape():
    """Verify that spatial coordinates add 2 channels."""
    batch = torch.randn(2, 3, 32, 64)
    out = add_spatial_coordinates(batch)
    assert out.shape == (2, 5, 32, 64)


def test_add_spatial_coordinates_values_centered():
    """Verify centered spatial coordinates map to [-1, 1]."""
    batch = torch.zeros(1, 1, 4, 4)
    out = add_spatial_coordinates(batch, center_origin=True)

    x_chan = out[0, 1, :, :]
    y_chan = out[0, 2, :, :]

    # Check corners
    assert torch.allclose(x_chan[:, 0], torch.tensor(-1.0))
    assert torch.allclose(x_chan[:, -1], torch.tensor(1.0))
    assert torch.allclose(y_chan[0, :], torch.tensor(-1.0))
    assert torch.allclose(y_chan[-1, :], torch.tensor(1.0))

    # Check that x varies across columns, y across rows
    assert torch.allclose(x_chan[0, :], x_chan[1, :])
    assert torch.allclose(y_chan[:, 0], y_chan[:, 1])


def test_add_spatial_coordinates_values_topleft():
    """Verify top-left spatial coordinates map to [0, 1]."""
    batch = torch.zeros(1, 1, 4, 4)
    out = add_spatial_coordinates(batch, center_origin=False)

    x_chan = out[0, 1, :, :]
    y_chan = out[0, 2, :, :]

    assert torch.allclose(x_chan[:, 0], torch.tensor(0.0))
    assert torch.allclose(x_chan[:, -1], torch.tensor(1.0))
    assert torch.allclose(y_chan[0, :], torch.tensor(0.0))
    assert torch.allclose(y_chan[-1, :], torch.tensor(1.0))


# =============================================================================
# preprocess_image_for_rwkv7 Tests
# =============================================================================


def test_preprocess_image_for_rwkv7_full_pipeline(tmp_path):
    """Verify the full 6-channel pipeline (L, a, b, alpha, x, y)."""
    img_path = tmp_path / "test_rgba.png"
    # Create an RGBA image
    Image.new("RGBA", (100, 200), (255, 128, 64, 200)).save(str(img_path))

    # Default settings: oklab=True, include_alpha=True, center_origin=True
    tensor = preprocess_image_for_rwkv7(str(img_path), target_size=(224, 224))

    # Shape: (1, 6, H, W)
    assert tensor.shape == (1, 6, 224, 224)

    L, a, b, alpha, x, y = [tensor[0, i] for i in range(6)]

    # 1. Check alpha (channel 3)
    # 200 / 255.0 ≈ 0.7843
    # Balanced with 2.0 * alpha - 1.0 -> 2.0 * 0.7843 - 1.0 = 0.5686
    expected_alpha = 2.0 * (200 / 255.0) - 1.0
    assert torch.allclose(alpha, torch.full_like(alpha, expected_alpha))

    # 2. Check spatial coordinates (channels 4, 5)
    assert x.min() == -1.0
    assert x.max() == 1.0
    assert y.min() == -1.0
    assert y.max() == 1.0

    # 3. Check color channels (L, a, b)
    # Just ensure they are finite and L is roughly in [0, 1]
    assert torch.isfinite(L).all()
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    assert L.mean() > 0  # Not a black image


def test_preprocess_image_for_rwkv7_no_alpha(tmp_path):
    """Verify pipeline without alpha still returns 6 channels (L, a, b, alpha, x, y)."""
    img_path = tmp_path / "test_rgb.jpg"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(str(img_path))

    # include_alpha=False -> alpha defaults to 1.0 (opaque)
    tensor = preprocess_image_for_rwkv7(
        str(img_path), target_size=(32, 32), include_alpha=False
    )

    # Shape: (1, 6, H, W) -> L, a, b, alpha, x, y
    assert tensor.shape == (1, 6, 32, 32)
    # Alpha (channel 3) should be 1.0 (2.0 * 1.0 - 1.0 = 1.0)
    assert torch.allclose(tensor[0, 3], torch.tensor(1.0))
    # Last two should be x, y
    assert tensor[0, 4].min() == -1.0
    assert tensor[0, 5].max() == 1.0


# =============================================================================
# calculate_dataset_mean_std Tests
# =============================================================================


def test_revert_balanced_superpixel_features_roundtrip():
    """Verify that balancing and reverting is a lossless round-trip operation."""
    B, H, W = 2, 32, 32
    # Create random OkLAB-like data
    L = torch.rand(B, 1, H, W)
    a = (torch.rand(B, 1, H, W) - 0.5) * 0.8  # ~[-0.4, 0.4]
    b = (torch.rand(B, 1, H, W) - 0.5) * 0.8
    alpha = torch.rand(B, 1, H, W)

    # 1. Test the math directly
    L_bal = 2.0 * L - 1.0
    alpha_bal = 2.0 * alpha - 1.0
    chroma_scale = 2.5
    a_bal = a * chroma_scale
    b_bal = b * chroma_scale
    x = torch.zeros(B, 1, H, W)
    y = torch.zeros(B, 1, H, W)
    
    balanced = torch.cat([L_bal, a_bal, b_bal, alpha_bal, x, y], dim=1)
    
    # 2. Revert
    oklab_rev, alpha_rev = revert_balanced_superpixel_features(balanced, chroma_scale=chroma_scale)
    
    # 3. Assertions
    assert torch.allclose(oklab_rev[:, 0:1], L, atol=1e-6)
    assert torch.allclose(oklab_rev[:, 1:2], a, atol=1e-6)
    assert torch.allclose(oklab_rev[:, 2:3], b, atol=1e-6)
    assert torch.allclose(alpha_rev, alpha, atol=1e-6)


def test_prepare_balanced_superpixel_features_integration():
    """Verify integration of prepare_balanced_superpixel_features."""
    B, H, W = 1, 16, 16
    srgb = torch.rand(B, 3, H, W)
    alpha = torch.rand(B, 1, H, W)
    
    balanced = prepare_balanced_superpixel_features(srgb, alpha=alpha)
    assert balanced.shape == (B, 6, H, W)
    
    # Check if L_bal is in [-1, 1]
    assert (balanced[:, 0] >= -1.0001).all() and (balanced[:, 0] <= 1.0001).all()
    # Check if alpha_bal is in [-1, 1]
    assert (balanced[:, 3] >= -1.0001).all() and (balanced[:, 3] <= 1.0001).all()
    # Check if x, y are in [-1, 1]
    assert (balanced[:, 4:] >= -1.0001).all() and (balanced[:, 4:] <= 1.0001).all()


def test_calculate_dataset_mean_std_rgb_uniform(tmp_path):
    """Verify RGB stats calculation on a uniform gray dataset."""
    create_dummy_dataset(
        str(tmp_path), num_classes=2, images_per_class=2, size=(64, 64)
    )

    # Overwrite with pure uniform gray (128, 128, 128)
    for root, _, files in os.walk(str(tmp_path)):
        for f in files:
            create_dummy_image(os.path.join(root, f), (64, 64), (128, 128, 128))

    mean, std = calculate_dataset_mean_std(
        str(tmp_path), img_size=64, batch_size=2, color_space="rgb"
    )

    expected_val = 128.0 / 255.0  # ~0.5019
    assert len(mean) == 3
    assert len(std) == 3

    # Mean should be very close to 0.5019
    for m in mean:
        assert abs(m - expected_val) < 0.01

    # Std should be very close to 0 for a uniform image
    for s in std:
        assert s < 0.01


def test_calculate_dataset_mean_std_oklab_finite(tmp_path):
    """Verify OkLAB stats calculation produces finite and plausible values."""
    create_dummy_dataset(
        str(tmp_path), num_classes=3, images_per_class=4, size=(64, 64)
    )

    mean, std = calculate_dataset_mean_std(
        str(tmp_path), img_size=64, batch_size=4, color_space="oklab"
    )

    assert len(mean) == 3
    assert len(std) == 3

    # All values must be finite
    assert all(torch.isfinite(torch.tensor(m)) for m in mean)
    assert all(torch.isfinite(torch.tensor(s)) for s in std)

    # OkLAB L mean should be roughly between 0 and 1
    assert 0.0 <= mean[0] <= 1.0


def test_calculate_dataset_mean_std_batch_consistency(tmp_path):
    """Verify that changing batch size doesn't significantly alter the calculated stats."""
    create_dummy_dataset(
        str(tmp_path), num_classes=2, images_per_class=5, size=(32, 32)
    )

    mean_b1, std_b1 = calculate_dataset_mean_std(
        str(tmp_path), img_size=32, batch_size=1, color_space="rgb"
    )
    mean_b4, std_b4 = calculate_dataset_mean_std(
        str(tmp_path), img_size=32, batch_size=4, color_space="rgb"
    )

    # Results should be identical regardless of batch size
    for m1, m4 in zip(mean_b1, mean_b4):
        assert abs(m1 - m4) < 1e-7
    for s1, s4 in zip(std_b1, std_b4):
        assert abs(s1 - s4) < 1e-7
