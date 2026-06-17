"""Helper functions for differentiable SLIC superpixel algorithm."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Filler value for masked positions in softmax (finite to avoid NaN when all positions are masked)
FILLER = -1e9


def _masked_softmax(
    similarities: torch.Tensor,
    tau: float = 0.01,
    dim: int = 1,
    stable: bool = False,
) -> torch.Tensor:
    """Apply softmax with temperature, masking zero-similarity positions.

    Uses FILLER (finite) instead of -inf to avoid NaN when ALL positions are masked.
    """
    similarities = torch.where(similarities == 0, FILLER, similarities)
    if stable:
        similarities = similarities - similarities.max(dim, keepdim=True).values.detach()
    return (similarities / tau).softmax(dim)


def compute_stride_and_padding(
    img_shape: Tuple[int, int],
    spixel_shape: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Args:
        img_shape (Tuple[int, int]): input image shape (height, width)
        spixel_shape (Tuple[int, int]): superpixel image shape (height, width)

    Returns:
        stride (Tuple[int, int]): (stride_h, stride_w)
        padding (Tuple[int, int]): (pad_x, pad_y)
    """
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
    r"""upsampling a feature map based on superpixels

    Args:
        x (torch.Tensor): a tensor of shape (batch, channels, height_s, width_s)
                          superpixel features
        assignments (torch.Tensor): a tensor of shape
                                    (batch, (2*candidate_radius + 1)**2, height, width)
                                    pixel-to-superpixel assignment
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_s * width_s grids
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled

    Returns:
        upsampled_features (torch.Tensor): a tensor of shape (batch, channels, height, width)
    """
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
    # padding an assignments so that its height and width are divisible by stride values
    pad_x, pad_y = padding
    assignments = F.pad(assignments, (0, pad_x, 0, pad_y))
    height += pad_y
    width += pad_x
    # get candidate clusters and corresponding assignments
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
    # unpad
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
    r"""downsampling a feature map based on superpixels

    Args:
        x (torch.Tensor): a tensor of shape (batch, channels, height, width)
                          pixel features
        assignments (torch.Tensor): a tensor of shape
                                    (batch, (2*candidate_radius + 1)**2, height_s, width_s)
                                    superpixel-to-pixel assignment
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_s * width_s grids
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled

    Returns:
        downsampled_features (torch.Tensor): a tensor of shape (batch, channels, height_s, width_s)
    """
    batch, _, height_s, width_s = assignments.shape
    height, width = x.shape[-2:]
    channels = x.shape[1]
    if stride is None:
        stride, padding = compute_stride_and_padding(
            (height, width), (height_s, width_s)
        )
    else:
        _, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    # padding an assignments so that its height and width are divisible by stride values
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
    r"""compute elem-to-center assignment with a local attention

    Args:
        clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        elem_feats (torch.Tensor): a tensor of shape (batch, channels, height, width)
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_c * width_c grids
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable computation of softmax with temperature

    Returns:
        soft_assignment (torch.Tensor): (batch, (2*candidate_radius + 1)**2, height, width)
        similarities (torch.Tensor): (batch, (2*candidate_radius + 1)**2, height, width)
    """
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""compute center-to-elem assignment with a local attention

    Args:
        clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        elem_feats (torch.Tensor): a tensor of shape (batch, channels, height, width)
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_c * width_c grids
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable computation of softmax with temperature

    Returns:
        soft_assignment (torch.Tensor): (batch, (2*candidate_radius + 1)**2, height_c, width_c)
        similarities (torch.Tensor): (batch, (2*candidate_radius + 1)**2, height, width)
    """
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
    return soft_assignment, similarities


def update_clst_feats(
    elem_feats: torch.Tensor,
    clst_feats: torch.Tensor,
    stride: Tuple[int, int],
    tau: float = 0.01,
    candidate_radius: int = 1,
    stable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""update cluster features with a local attention.

    Calls compute_center_to_elem_assignment then aggregates via einsum.

    Args:
        elem_feats (torch.Tensor): (batch, channels, height, width)
        clst_feats (torch.Tensor): (batch, channels, height_c, width_c)
        stride (Tuple[int, int]): grid size
        tau (float): temperature parameter.
        candidate_radius (int): radius for candidate sampling
        stable (bool): stable softmax computation

    Returns:
        new_clst_feats (torch.Tensor): (batch, channels, height_c, width_c)
        soft_assignment (torch.Tensor): (batch, ..., height_c, width_c)
        similarities (torch.Tensor): (batch, ..., height_c, width_c)
    """
    soft_assignment, similarities = compute_center_to_elem_assignment(
        clst_feats, elem_feats, stride, tau, candidate_radius, stable
    )
    b, c, h, w = clst_feats.shape
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0] * neighbor_range, stride[1] * neighbor_range)
    padding = (stride[0] * candidate_radius, stride[1] * candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]
    unfold_elem_feats = F.unfold(
        elem_feats, kernel_size, padding=padding, stride=stride
    )
    unfold_elem_feats = unfold_elem_feats.reshape(b, c, n_candidate_pixels, h, w)
    new_clst_feats = torch.einsum(
        "bphw,bcphw->bchw", (soft_assignment, unfold_elem_feats)
    )
    return new_clst_feats, soft_assignment, similarities
