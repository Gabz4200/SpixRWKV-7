"""Image and feature transforms for SpixRWKV-7: preprocessing, color conversion, loading."""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from spixrwkv7.data.colors import from_linear_rgb_to_oklab, from_srgb_to_linear_rgb

# ==============================================================================
# Default Normalization Constants
# ==============================================================================

IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406, 0.5]
IMAGENET_RGB_STD = [0.229, 0.224, 0.225, 0.5]

DEFAULT_OKLAB_MEAN = [0.5, 0.0, 0.0, 0.5]
DEFAULT_OKLAB_STD = [0.2, 0.15, 0.15, 0.5]


def _convert_srgb_to_oklab(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert an sRGB tensor (B, 3, H, W) or (B, 4, H, W) to OkLAB (B, 3, H, W) or (B, 4, H, W)."""
    if srgb_tensor.shape[1] == 4:
        alpha = srgb_tensor[:, 3:4, :, :]
        rgb = srgb_tensor[:, 0:3, :, :]
    else:
        rgb = srgb_tensor
        alpha = None

    linear_rgb = from_srgb_to_linear_rgb(rgb)
    oklab = from_linear_rgb_to_oklab(linear_rgb)

    if alpha is not None:
        return torch.cat([oklab, alpha], dim=1)
    return oklab


def calculate_dataset_mean_std(
    dataloader: torch.utils.data.DataLoader,
    color_space: str = "rgb",
    device: Optional[torch.device] = None,
) -> tuple[list[float], list[float]]:
    """Calculate per-channel mean and std over an entire dataset.

    Uses variance-pooling formula: std = sqrt(E[x^2] - E[x]^2)
    across all batches, avoiding Bessel correction bias from per-batch
    weighted averaging.
    """
    total_sum: Optional[torch.Tensor] = None
    total_sqsum: Optional[torch.Tensor] = None
    total_count = 0

    for batch in dataloader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        if device is not None:
            images = images.to(device)
        B, C, H, W = images.shape

        if color_space.lower() == "rgb":
            pass
        elif color_space.lower() == "oklab":
            images = _convert_srgb_to_oklab(images)

        # Flatten spatial dims: (B, C, H*W)
        flat = images.view(B, C, -1)

        # Accumulate sum and sum-of-squares for correct variance pooling
        batch_sum = flat.sum(dim=(0, 2))  # (C,)
        batch_sqsum = (flat ** 2).sum(dim=(0, 2))  # (C,)

        if total_sum is None:
            total_sum = batch_sum
            total_sqsum = batch_sqsum
        else:
            assert total_sum is not None
            total_sum += batch_sum
            total_sqsum += batch_sqsum
        total_count += B * H * W  # total elements per channel

    # total_sum/total_sqsum are guaranteed assigned by this point (loop ran at least once)
    assert total_sum is not None and total_sqsum is not None, "dataloader must yield at least one batch"
    total_mean = total_sum / total_count
    total_var = total_sqsum / total_count - total_mean ** 2
    total_std = total_var.clamp(min=0.0).sqrt()
    return total_mean.tolist(), total_std.tolist()


def load_image_to_tensor(
    image_path: str,
    img_size: int = -1,
    target_size: Optional[Tuple[int, int]] = None,
    color_space: str = "rgb",
    normalize: bool = False,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
    include_alpha: bool = False,
) -> torch.Tensor:
    """Load an image from disk, convert to tensor, resize, normalize, and convert color space.
    
    Args:
        image_path: Path to image file.
        img_size: Target height (-1 for original size, otherwise scales proportionally).
        target_size: Legacy tuple (H, W) - overrides img_size if provided.
        color_space: 'rgb' or 'oklab'.
        normalize: Apply mean/std normalization.
        mean: Channel means (defaults IMAGENET_RGB_MEAN).
        std: Channel stds (defaults IMAGENET_RGB_STD).
        include_alpha: Include alpha channel from RGBA images.
    """
    if include_alpha:
        pil_image = Image.open(image_path).convert("RGBA")
    else:
        pil_image = Image.open(image_path).convert("RGB")

    if target_size is not None:
        pil_image = pil_image.resize(target_size, Image.Resampling.BILINEAR)
    elif img_size > 0:
        orig_w, orig_h = pil_image.size  # PIL uses (width, height)
        aspect = orig_w / orig_h
        new_h = img_size
        new_w = int(round(new_h * aspect))
        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    if include_alpha:
        arr = np.array(pil_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    else:
        arr = np.array(pil_image, dtype=np.float32).transpose(2, 0, 1) / 255.0

    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)

    if normalize:
        if mean is None:
            mean = IMAGENET_RGB_MEAN[: tensor.shape[1]]
        if std is None:
            std = IMAGENET_RGB_STD[: tensor.shape[1]]
        mean_t = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
        tensor = (tensor - mean_t) / std_t

    if color_space.lower() == "oklab":
        tensor = _convert_srgb_to_oklab(tensor)
    elif color_space.lower() != "rgb":
        raise ValueError(f"Unsupported color_space: '{color_space}'. Supported: 'rgb', 'oklab'.")

    return tensor


def add_spatial_coordinates(
    tensor: torch.Tensor, center_origin: bool = True
) -> torch.Tensor:
    """Add normalized x, y coordinate channels to a (B, C, H, W) tensor."""
    B, _, H, W = tensor.shape
    device = tensor.device

    if center_origin:
        y = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    else:
        y = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

    return torch.cat([tensor, x, y], dim=1)


def smart_resize(
    height: int, width: int, target_pixels: int = 224 * 224
) -> Tuple[int, int]:
    """Resize dimensions to approximately target_pixels while preserving aspect ratio."""
    aspect = width / height
    new_h = int(round((target_pixels / aspect) ** 0.5))
    new_w = int(round(new_h * aspect))
    return new_h, new_w


def preprocess_image_for_rwkv7(
    image_path: str,
    img_size: int = -1,
    target_size: Tuple[int, int] = (64, 64),
    include_alpha: bool = True,
) -> torch.Tensor:
    """Full preprocessing pipeline returning (1, 6, H, W) tensor: OkLAB + alpha + xy.
    
    Args:
        image_path: Path to image file.
        img_size: Target height (-1 for original size, otherwise scales proportionally).
        target_size: Legacy tuple (H, W) - overrides img_size if provided.
        include_alpha: Include alpha channel from RGBA images.
    """
    tensor = load_image_to_tensor(
        image_path,
        img_size=img_size,
        target_size=target_size,
        color_space="oklab",
        include_alpha=include_alpha,
    )
    # Always return 6 channels: OkLAB (3) + alpha (1) + xy (2)
    if not include_alpha:
        # Inject zero alpha channel when not provided (opaque)
        device = tensor.device
        _, _, H, W = tensor.shape
        alpha_channel = torch.ones(1, 1, H, W, device=device)
        tensor = torch.cat([tensor, alpha_channel], dim=1)
    tensor = add_spatial_coordinates(tensor, center_origin=True)
    return tensor  # (1, 6, H, W)


def prepare_balanced_superpixel_features(
    image_tensor: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    chroma_scale: float = 2.5,
) -> torch.Tensor:
    """Convert an sRGB tensor to balanced 6-channel features for the backbone.

    The output is (B, 6, H, W) with:
        [0]: Lightness L = 2.0 * OkLAB_L - 1.0 (balanced in [-1, 1])
        [1-2]: Chroma a, b = chroma_scale * OkLAB_a, chroma_scale * OkLAB_b
        [3]: Alpha (0 if not provided)
        [4-5]: Normalized xy coordinates (centered in [-1, 1])
    """
    if image_tensor.shape[1] == 4:
        rgb = image_tensor[:, :3, :, :]
        alpha = image_tensor[:, 3:4, :, :]
    elif image_tensor.shape[1] == 3:
        rgb = image_tensor[:, :3, :, :]
    else:
        raise ValueError(f"Expected 3 or 4 input channels, got {image_tensor.shape[1]}")

    linear_rgb = from_srgb_to_linear_rgb(rgb)
    oklab = from_linear_rgb_to_oklab(linear_rgb)

    L = 2.0 * oklab[:, 0:1, :, :] - 1.0
    a = chroma_scale * oklab[:, 1:2, :, :]
    b = chroma_scale * oklab[:, 2:3, :, :]

    if alpha is None:
        alpha = torch.zeros_like(L)

    features = add_spatial_coordinates(
        torch.cat([L, a, b, alpha], dim=1), center_origin=True
    )
    return features  # (B, 6, H, W)


def revert_balanced_superpixel_features(
    balanced: torch.Tensor, chroma_scale: float = 2.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reverse prepare_balanced_superpixel_features: extract OkLAB and alpha."""
    L_norm = balanced[:, 0:1, :, :]
    a_norm = balanced[:, 1:2, :, :]
    b_norm = balanced[:, 2:3, :, :]
    alpha = balanced[:, 3:4, :, :]

    L = (L_norm + 1.0) / 2.0  # back to [0, 1]
    a = a_norm / chroma_scale
    b = b_norm / chroma_scale

    oklab = torch.cat([L, a, b], dim=1)
    return oklab, alpha
