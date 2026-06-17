import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Optional, Tuple, cast

# Import color conversion utilities from your provided colors.py
from .colors import from_srgb_to_linear_rgb, from_linear_rgb_to_oklab

# ==============================================================================
# Default Normalization Constants
# ==============================================================================

# ImageNet RGB statistics.
# These were calculated by the original authors over the entire ImageNet
# training set after scaling pixel values to [0.0, 1.0].
IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD = [0.229, 0.224, 0.225]

# Heuristic OkLAB statistics for natural images.
# Since OkLAB perceptual color space is not standard on ML, there is no universally
# accepted "ImageNet equivalent" dataset stats yet. However, we can assume
# reasonable defaults based on the geometry of the OkLAB space:
# - L (Lightness) ranges roughly from 0 to 1. A mean of 0.5 and std of 0.2
#   is a good approximation for the distribution of natural images.
# - a and b (color opponents) range roughly from -0.5 to 0.5, centered at 0.
#   A std of 0.15 is a safe assumption for natural color variation.
DEFAULT_OKLAB_MEAN = [0.5, 0.0, 0.0]
DEFAULT_OKLAB_STD = [0.2, 0.15, 0.15]


def _convert_srgb_to_oklab(srgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Helper to convert an sRGB tensor [0, 1] to OkLAB.
    Uses the exact pipeline: sRGB -> Linear RGB -> OkLAB.
    """
    linear_rgb = from_srgb_to_linear_rgb(srgb_tensor)
    oklab = from_linear_rgb_to_oklab(linear_rgb)
    return oklab


def calculate_dataset_mean_std(
    data_dir: str, img_size: int = 224, batch_size: int = 64, color_space: str = "rgb"
):
    """
    Calculates the mean and std of a dataset in either RGB or OkLAB space.

    Args:
        data_dir (str): Path to the dataset folder.
        img_size (int): Target image size.
        batch_size (int): Batch size for DataLoader.
        color_space (str): 'rgb' or 'oklab'.
    """
    # 1. Transform to tensor ONLY. Do NOT normalize yet!
    # ToTensor automatically converts PIL images to sRGB tensors in [0.0, 1.0]
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    # 2. Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Initialize accumulators for 3 channels
    sum_pixels = torch.zeros(3)
    sum_sq_pixels = torch.zeros(3)
    total_pixels = 0

    print(
        f"Calculating mean and std in {color_space.upper()} space... (This may take a few minutes)"
    )

    # 4. Iterate through the dataset
    for images, _ in loader:
        # If OkLAB is requested, convert the sRGB batch to OkLAB before calculating stats
        if color_space == "oklab":
            images = _convert_srgb_to_oklab(images)

        # images shape: (Batch, Channels, Height, Width)
        # Flatten H and W to calculate stats per channel
        b, c, h, w = images.shape
        pixels = images.permute(0, 2, 3, 1).reshape(-1, c)  # Shape: (B*H*W, C)

        # Accumulate sums
        sum_pixels += pixels.sum(dim=0)
        sum_sq_pixels += (pixels**2).sum(dim=0)
        total_pixels += pixels.shape[0]

    # 5. Calculate final mean and std
    mean = sum_pixels / total_pixels
    var = (sum_sq_pixels / total_pixels) - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))

    return mean.tolist(), std.tolist()


def load_image_to_tensor(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    color_space: str = "rgb",
    mean: Optional[list] = None,
    std: Optional[list] = None,
):
    """
    Loads an image, optionally converts to OkLAB, and normalizes it.

    Args:
        image_path (str): Path to the image file.
        target_size (Tuple[int, int], optional): Target (Height, Width).
        normalize (bool): If True, normalizes the tensor.
        color_space (str): 'rgb' (default) or 'oklab'.
        mean (list, optional): Mean for normalization. If None, uses defaults.
        std (list, optional): Std for normalization. If None, uses defaults.

    Returns:
        torch.Tensor: Tensor of shape (1, 3, Height, Width).
    """

    # 1. Open image and force conversion to RGB to ensure C=3
    img = Image.open(image_path).convert("RGB")

    # 2. Build the transformation pipeline
    transform_list = []
    if target_size is not None:
        transform_list.append(transforms.Resize(target_size))

    # ToTensor converts PIL Image (H, W, C) in [0, 255] to sRGB Tensor (C, H, W) in [0.0, 1.0]
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # 3. Apply transforms (Resulting shape: C, H, W)
    img_tensor = cast(torch.Tensor, transform(img))

    # 4. Add the Batch dimension at the front (Resulting shape: B, C, H, W where B=1)
    img_tensor = img_tensor.unsqueeze(0)

    # 5. Convert to OkLAB if requested
    if color_space == "oklab":
        img_tensor = _convert_srgb_to_oklab(img_tensor)
    elif color_space != "rgb":
        raise ValueError(
            f"Unsupported color_space: '{color_space}'. Use 'rgb' or 'oklab'."
        )

    # 6. Normalize if requested
    if normalize:
        # Determine which default constants to use based on the color space
        if color_space == "rgb":
            norm_mean = mean if mean is not None else IMAGENET_RGB_MEAN
            norm_std = std if std is not None else IMAGENET_RGB_STD
        elif color_space == "oklab":
            norm_mean = mean if mean is not None else DEFAULT_OKLAB_MEAN
            norm_std = std if std is not None else DEFAULT_OKLAB_STD

        # Apply normalization manually using tensor broadcasting.
        # Equivalent to transforms.Normalize(mean=norm_mean, std=norm_std) but more explicit.
        # mean and std are reshaped to (1, C, 1, 1) to broadcast across (B, C, H, W)
        mean_t = torch.tensor(norm_mean, dtype=img_tensor.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(norm_std, dtype=img_tensor.dtype).view(1, -1, 1, 1)
        img_tensor = (img_tensor - mean_t) / std_t

    return img_tensor
