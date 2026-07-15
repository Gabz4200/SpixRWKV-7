"""Shared test fixtures and helpers for SpixRWKV-7 tests."""

import pytest
import torch

from spixrwkv7 import DiffSLIC, Vision_RWKV7

# =====================================================================
# Shared model configs (reused across test_model.py and test_regression.py)
# =====================================================================

TINY_CONFIG = {
    "img_size": 32,
    "embed_dims": 64,
    "num_heads": 1,
    "depth": 1,
    "num_superpixels": 9,
    "diff_slic_iters": 2,
    "in_chans": 6,
}

SMALL_CONFIG = {
    "img_size": 64,
    "embed_dims": 64,
    "num_heads": 4,
    "depth": 2,
    "num_superpixels": 36,
    "diff_slic_iters": 3,
    "in_chans": 6,
}

VQ_TINY_CONFIG = {
    "img_size": 32,
    "embed_dims": 64,
    "depth": 2,
    "codebook_size": 64,
    "downsample_factor": 8,
    "latent_dim": 32,
    "in_chans": 6,
}


# =====================================================================
# Factory helpers
# =====================================================================


def make_vision_rwkv7(**overrides):
    """Create a tiny Vision_RWKV7 with sensible test defaults."""
    cfg = {**TINY_CONFIG, **overrides}
    return Vision_RWKV7(**cfg)


def make_diffslic(n_spixels=8, n_iter=3, **kwargs):
    """Create a DiffSLIC with sensible test defaults."""
    defaults = dict(tau=0.01, candidate_radius=1, normalize=True, stable=True)
    defaults.update(kwargs)
    return DiffSLIC(n_spixels=n_spixels, n_iter=n_iter, **defaults)


def get_dummy_neighbors(B, N, K=4):
    """Create dummy valid neighbors for testing blocks."""
    offsets = torch.arange(1, K + 1).unsqueeze(0)
    neighbors = (torch.arange(N).unsqueeze(1) + offsets) % N
    return neighbors.unsqueeze(0).expand(B, -1, -1)


# =====================================================================
# Assertion helpers
# =====================================================================


def assert_finite(tensor, msg=""):
    """Assert all values in tensor are finite."""
    assert torch.isfinite(tensor).all(), f"Not finite: {msg}"


def assert_all_finite(tensors, msg=""):
    """Assert all values in a sequence of tensors are finite."""
    for i, t in enumerate(tensors):
        assert torch.isfinite(t).all(), f"Tensor {i} not finite: {msg}"
