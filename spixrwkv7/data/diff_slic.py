"""Differentiable SLIC superpixel segmentation and helper functions."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Helper functions
# =====================================================================

# Filler value for masked positions in softmax (finite to avoid NaN when all positions are masked)
FILLER = -1e9


def _masked_softmax(
    similarities: torch.Tensor,
    tau: float = 0.01,
    dim: int = 1,
    stable: bool = False,
) -> torch.Tensor:
    """Apply softmax with temperature, masking zero-similarity positions."""
    similarities = torch.where(similarities == 0, FILLER, similarities)
    if stable:
        similarities = (
            similarities - similarities.max(dim, keepdim=True).values.detach()
        )
    return (similarities / tau).softmax(dim)


def compute_stride_and_padding(
    img_shape: Tuple[int, int],
    spixel_shape: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Compute stride and padding from image and superpixel grid shapes."""
    height, width = img_shape
    height_s, width_s = spixel_shape
    stride_h = (height + height_s - 1) // height_s
    stride_w = (width + width_s - 1) // width_s
    pad_y = (height_s - height % height_s) % height_s
    pad_x = (width_s - width % width_s) % width_s
    stride = (stride_h, stride_w)
    padding = (pad_x, pad_y)
    return stride, padding


def spixel_upsampling(
    x: torch.Tensor,
    assignments: torch.Tensor,
    stride: Optional[Tuple[int, int]] = None,
    candidate_radius: int = 1,
) -> torch.Tensor:
    r"""Upsampling a feature map based on superpixels."""
    batch_size, _, height, width = assignments.shape
    n_channels = x.shape[1]
    height_s, width_s = x.shape[-2:]
    n_spixels = height_s * width_s
    if stride is None:
        stride, padding = compute_stride_and_padding(
            (height, width), (height_s, width_s)
        )
    else:
        _, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    pad_x, pad_y = padding
    assignments = F.pad(assignments, (0, pad_x, 0, pad_y))
    height += pad_y
    width += pad_x
    neighbor_range = candidate_radius * 2 + 1
    candidate_clusters = F.unfold(
        x, kernel_size=neighbor_range, padding=candidate_radius
    )
    candidate_clusters = candidate_clusters.reshape(
        batch_size, n_channels, neighbor_range**2, n_spixels
    )
    assignments = F.unfold(assignments, kernel_size=stride, stride=stride)
    assignments = assignments.reshape(
        batch_size, neighbor_range**2, stride[0] * stride[1], n_spixels
    )
    upsampled_features = torch.einsum(
        "bkcn,bcpn->bkpn", (candidate_clusters, assignments)
    )
    upsampled_features = upsampled_features.contiguous().reshape(
        batch_size * n_channels, stride[0] * stride[1], -1
    )
    upsampled_features = F.fold(
        upsampled_features, (height, width), kernel_size=stride, stride=stride
    )
    upsampled_features = upsampled_features.reshape(
        batch_size, n_channels, height, width
    )
    if pad_y > 0:
        upsampled_features = upsampled_features[..., :-pad_y, :]
    if pad_x > 0:
        upsampled_features = upsampled_features[..., :-pad_x]
    return upsampled_features


def spixel_downsampling(
    x: torch.Tensor,
    assignments: torch.Tensor,
    stride: Optional[Tuple[int, int]] = None,
    candidate_radius: int = 1,
) -> torch.Tensor:
    r"""Downsampling a feature map based on superpixels."""
    batch, _, height_s, width_s = assignments.shape
    height, width = x.shape[-2:]
    channels = x.shape[1]
    if stride is None:
        stride, padding = compute_stride_and_padding(
            (height, width), (height_s, width_s)
        )
    else:
        _, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    pad_x, pad_y = padding
    x = F.pad(x, (0, pad_x, 0, pad_y))
    height += pad_y
    width += pad_x
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0] * neighbor_range, stride[1] * neighbor_range)
    padding = (stride[0] * candidate_radius, stride[1] * candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]
    unfold_elem_feats = F.unfold(x, kernel_size, stride=stride, padding=padding)
    unfold_elem_feats = unfold_elem_feats.reshape(
        batch, channels, n_candidate_pixels, height_s, width_s
    )
    downsampled_features = torch.einsum(
        "bphw,bcphw->bchw", (assignments, unfold_elem_feats)
    )
    return downsampled_features


def compute_elem_to_center_assignment(
    clst_feats: torch.Tensor,
    elem_feats: torch.Tensor,
    stride: Tuple[int, int],
    tau: float = 0.01,
    candidate_radius: int = 1,
    stable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute elem-to-center assignment with a local attention."""
    batch_size, channels, height, width = elem_feats.shape
    n_spixels = clst_feats.shape[2] * clst_feats.shape[3]
    neighbor_range = candidate_radius * 2 + 1
    candidate_clusters = F.unfold(
        clst_feats, kernel_size=neighbor_range, padding=candidate_radius
    )
    candidate_clusters = candidate_clusters.reshape(
        batch_size, channels, neighbor_range**2, n_spixels
    )
    unfold_elem_feats = F.unfold(elem_feats, kernel_size=stride, stride=stride)
    unfold_elem_feats = unfold_elem_feats.reshape(
        batch_size, channels, stride[0] * stride[1], n_spixels
    )
    similarities = torch.einsum(
        "bkcn,bkpn->bcpn", (candidate_clusters, unfold_elem_feats)
    )
    similarities = similarities.contiguous().reshape(
        batch_size * neighbor_range**2, -1, n_spixels
    )
    similarities = F.fold(
        similarities, (height, width), kernel_size=stride, stride=stride
    )
    similarities = similarities.reshape(batch_size, neighbor_range**2, height, width)
    soft_assignment = _masked_softmax(similarities, tau, dim=1, stable=stable)
    return soft_assignment, similarities


def compute_center_to_elem_assignment(
    clst_feats: torch.Tensor,
    elem_feats: torch.Tensor,
    stride: Tuple[int, int],
    tau: float = 0.01,
    candidate_radius: int = 1,
    stable: bool = False,
    return_unfold: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""Compute center-to-elem assignment with a local attention."""
    b, c, h, w = clst_feats.shape
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0] * neighbor_range, stride[1] * neighbor_range)
    padding = (stride[0] * candidate_radius, stride[1] * candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]
    unfold_elem_feats = F.unfold(
        elem_feats, kernel_size, padding=padding, stride=stride
    )
    unfold_elem_feats = unfold_elem_feats.reshape(b, c, n_candidate_pixels, h, w)
    similarities = torch.einsum("bcphw,bchw->bphw", (unfold_elem_feats, clst_feats))
    soft_assignment = _masked_softmax(similarities, tau, dim=1, stable=stable)
    if return_unfold:
        return soft_assignment, similarities, unfold_elem_feats
    return soft_assignment, similarities, None



def update_clst_feats(
    elem_feats: torch.Tensor,
    clst_feats: torch.Tensor,
    stride: Tuple[int, int],
    tau: float = 0.01,
    candidate_radius: int = 1,
    stable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Update cluster features with a local attention."""
    soft_assignment, similarities, unfold_elem_feats = compute_center_to_elem_assignment(
        clst_feats, elem_feats, stride, tau, candidate_radius, stable, return_unfold=True
    )
    new_clst_feats = torch.einsum(
        "bphw,bcphw->bchw", (soft_assignment, unfold_elem_feats)
    )
    return new_clst_feats, soft_assignment, similarities


# =====================================================================
# DiffSLIC — Differentiable superpixel segmentation
# =====================================================================


class DiffSLIC(nn.Module):
    r"""Differentiable SLIC superpixel segmentation.

    Args:
        n_spixels: target number of superpixels
        n_iter: optimization iterations
        tau: softmax temperature (→0 gives hard assignments)
        candidate_radius: local candidate region radius
        normalize: whether to L2-normalize features
        stable: stable softmax computation with temperature
    """

    def __init__(
        self,
        n_spixels: int,
        n_iter: int = 5,
        tau: float = 0.01,
        candidate_radius: int = 1,
        normalize: bool = True,
        stable: bool = False,
        use_cpp: bool = False,
        hard_mode: bool = False,
    ) -> None:
        """Initialize DiffSLIC.

        Args:
            n_spixels: target number of superpixels.
            n_iter: EM iterations.
            tau: softmax temperature (smaller → harder assignments).
            candidate_radius: local candidate region radius.
            normalize: L2-normalize features before similarity.
            stable: numerically stable softmax (subtract max before exp).
            use_cpp: use C++ AVX kernel (default True; fails fast if .so missing).
            hard_mode: replace soft assignments with argmax at inference time
                (non-differentiable; gradients stop through assignments).
                Equivalent to seeds-revised hard SLIC for inference speed.
        """
        super().__init__()
        self.n_spixels = n_spixels
        self.n_iter = n_iter
        self.tau = tau
        self.candidate_radius = candidate_radius
        self.normalize = normalize
        self.stable = stable
        self.hard_mode = hard_mode
        self.use_cpp = use_cpp

    @property
    def _has_cpp(self) -> bool:
        """Return True if the C++ backend is loaded and active."""
        if not self.use_cpp:
            return False
        try:
            from spixrwkv7.kernels.rwkv7_kernel import _C
            return hasattr(_C, "diff_slic_update_clusters") and hasattr(_C, "diff_slic_assign_pixels")
        except (ImportError, AttributeError):
            return False

    def forward(
        self, x: torch.Tensor, clst_feats: Optional[torch.Tensor] = None, n_spixels: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run diffSLIC on a batch of images.

        Returns:
            clst_feats — (B, C, h_s, w_s) cluster features
            p2s_assign — (B, (2*r+1)^2, H, W) pixel-to-superpixel assignments
            s2p_assign — (B, ..., h_s, w_s) superpixel-to-pixel assignments (None if n_iter=0)
        """
        height, width = x.shape[-2:]
        if clst_feats is None:
            n_sp = n_spixels if n_spixels is not None else self.n_spixels
            height_s = max(1, int(math.sqrt(n_sp * height / width)))
            width_s = max(1, int(math.sqrt(n_sp * width / height)))
            stride_h = (height + height_s - 1) // height_s
            stride_w = (width + width_s - 1) // width_s
            stride = (stride_h, stride_w)
            clst_feats = F.adaptive_avg_pool2d(x, (height_s, width_s))
        else:
            height_s, width_s = clst_feats.shape[-2:]
            stride = ((height + height_s) // height_s, (width + width_s) // width_s)

        if self.normalize:
            x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)

        pad_x = (width_s - width % width_s) % width_s
        pad_y = (height_s - height % height_s) % height_s
        x = F.pad(x, (0, pad_x, 0, pad_y))

        s2p_assign = None

        if self.use_cpp:
            from spixrwkv7.kernels.rwkv7_kernel import (
                diff_slic_assign_pixels,
                diff_slic_update_clusters,
            )
            for _ in range(self.n_iter):
                clst_feats = diff_slic_update_clusters(
                    x, clst_feats, stride, self.candidate_radius,
                    self.tau, self.normalize,
                )
            p2s_assign = diff_slic_assign_pixels(
                x, clst_feats, stride, self.candidate_radius, self.tau,
            )
        else:
            for _ in range(self.n_iter):
                clst_feats, s2p_assign, _ = update_clst_feats(
                    x, clst_feats, stride, self.tau, self.candidate_radius
                )
                if self.normalize:
                    clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
            p2s_assign, _ = compute_elem_to_center_assignment(
                clst_feats, x, stride, self.tau, self.candidate_radius
            )

        if pad_y > 0:
            p2s_assign = p2s_assign[..., :-pad_y, :]
        if pad_x > 0:
            p2s_assign = p2s_assign[..., :-pad_x]

        # Hard mode: argmax assignments (non-differentiable, inference-speed equivalent
        # to seeds-revised hard SLIC). Gradient stops through assignment indices.
        if self.hard_mode and not self.training:
            hard_idx = p2s_assign.argmax(dim=1, keepdim=True)  # (B, 1, H, W)
            p2s_assign = torch.zeros_like(p2s_assign).scatter_(1, hard_idx, 1.0)

        return clst_feats, p2s_assign, s2p_assign

    def extra_repr(self):
        return (
            f"n_spixels={self.n_spixels}, \n "
            f"n_iter={self.n_iter}, \n "
            f"tau={self.tau}, \n "
            f"candidate_radius={self.candidate_radius}, \n"
        )
