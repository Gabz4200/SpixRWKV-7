"""Shared image loading utilities for test/benchmark scripts.

Provides ``load_random_caltech101_image`` which loads a real image from
``data/caltech101_classification/`` and returns the 6-channel (OkLAB + alpha + xy)
tensor expected by SpixRWKV-7 models.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import torch

from spixrwkv7.data.transforms import (
    add_spatial_coordinates,
    load_image_to_tensor,
)

# Class name -> integer label mapping for caltech101_classification
CALTECH101_LABELS = {
    "butterfly": 0,
    "dalmatian": 1,
    "dolphin": 2,
}

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "caltech101_classification"
)


def _scan_images(data_dir: str) -> dict[str, list[str]]:
    """Scan data_dir for class subfolders and their .jpg images.

    Returns:
        dict mapping class_name -> list of absolute image paths.
    """
    images: dict[str, list[str]] = {}
    base = Path(data_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Data directory not found: {base}")
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        jpgs = sorted(str(p) for p in class_dir.glob("*.jpg"))
        if jpgs:
            images[class_name] = jpgs
    return images


# Module-level cache so we only scan the directory once per process
_IMAGE_CACHE: Optional[dict[str, list[str]]] = None


def load_random_caltech101_image(
    img_size: int = 512,
    data_dir: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, str, int]:
    """Load a random image from caltech101_classification.

    Args:
        img_size: Target height for the output tensor (width scales proportionally).
        data_dir: Path to the image directory. Defaults to ``data/caltech101_classification/``.
        seed: Optional random seed for reproducibility (uses a local Random instance).
        device: Optional device to place the tensor on.

    Returns:
        (tensor, class_name, label) where:
          - tensor is (1, 6, H, W) in OkLAB + alpha + xy layout
          - class_name is the folder name (e.g. "butterfly")
          - label is the integer class label
    """
    global _IMAGE_CACHE

    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    if _IMAGE_CACHE is None or not os.path.isdir(data_dir):
        _IMAGE_CACHE = _scan_images(data_dir)

    cache = _IMAGE_CACHE
    if not cache:
        raise RuntimeError(f"No image classes found in {data_dir}")

    rng = random.Random(seed)

    # Pick a random class, then a random image from that class
    class_name = rng.choice(list(cache.keys()))
    image_path = rng.choice(cache[class_name])
    label = CALTECH101_LABELS.get(class_name, -1)

    # Load as (1, 6, H, W) OkLAB + alpha + xy
    from spixrwkv7.data.transforms import preprocess_image_for_rwkv7

    tensor = preprocess_image_for_rwkv7(
        image_path, target_size=(img_size, img_size), include_alpha=True
    )

    if device is not None:
        tensor = tensor.to(device)

    return tensor, class_name, label


def load_random_caltech101_batch(
    batch_size: int,
    img_size: int = 512,
    num_classes: int = 3,
    data_dir: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of random caltech101 images with labels.

    Useful for replacing ``synth_batch()`` in training convergence tests.

    Args:
        batch_size: Number of images in the batch.
        img_size: Target height for each image.
        num_classes: Number of classes (for label range validation).
        data_dir: Path to image directory.
        seed: Base seed (each image uses seed+i for determinism).
        device: Target device.

    Returns:
        (x, y) where:
          - x is (batch_size, 6, H, W) OkLAB + alpha + xy
          - y is (batch_size,) integer labels
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    images = []
    labels = []
    for i in range(batch_size):
        img_seed = (seed + i) if seed is not None else None
        tensor, _, label = load_random_caltech101_image(
            img_size=img_size,
            data_dir=data_dir,
            seed=img_seed,
            device=device,
        )
        images.append(tensor.squeeze(0))  # (6, H, W)
        labels.append(label)

    x = torch.stack(images, dim=0)  # (B, 6, H, W)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    return x, y


def load_random_caltech101_rgb(
    img_size: int = 512,
    data_dir: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, str, int]:
    """Load a random image as a 3-channel RGB tensor (for ViT baselines).

    Returns:
        (tensor, class_name, label) where tensor is (1, 3, H, W) RGB [0, 1].
    """
    global _IMAGE_CACHE

    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    if _IMAGE_CACHE is None or not os.path.isdir(data_dir):
        _IMAGE_CACHE = _scan_images(data_dir)

    cache = _IMAGE_CACHE
    rng = random.Random(seed)

    class_name = rng.choice(list(cache.keys()))
    image_path = rng.choice(cache[class_name])
    label = CALTECH101_LABELS.get(class_name, -1)

    tensor = load_image_to_tensor(
        image_path,
        target_size=(img_size, img_size),
        color_space="rgb",
        include_alpha=False,
    )

    if device is not None:
        tensor = tensor.to(device)

    return tensor, class_name, label


def load_caltech101_rgb_batch(
    batch_size: int,
    img_size: int = 512,
    data_dir: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of RGB images for ViT baselines.

    Returns:
        (x, y) where x is (B, 3, H, W) RGB and y is (B,) labels.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    images = []
    labels = []
    for i in range(batch_size):
        img_seed = (seed + i) if seed is not None else None
        tensor, _, label = load_random_caltech101_rgb(
            img_size=img_size,
            data_dir=data_dir,
            seed=img_seed,
            device=device,
        )
        images.append(tensor.squeeze(0))
        labels.append(label)

    x = torch.stack(images, dim=0)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y
