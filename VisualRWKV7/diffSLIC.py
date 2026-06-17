from typing import Optional, Tuple

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from .diffSLIC_funcs import (
    compute_elem_to_center_assignment,
    update_clst_feats,
    spixel_upsampling,
    spixel_downsampling,
)


class DiffSLIC(nn.Module):
    r"""Differentiable SLIC

    Args:
        n_spixels (int): a number of superpixels
        n_iter (int): a number of iterations for updating cluster centers
        tau (float): a temperature parameter. when tau -> 0, assignemnt is deterministic
        normalize (bool): if True, pixel and superpixel features are normalized so that those l2 norm are 1
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable compuatation of softmax with temperature
                       `stable` should be True, when using extremely small tau for obtaining deterministic assignment.
    """

    def __init__(
        self,
        n_spixels: int,
        n_iter: int = 5,
        tau: float = 0.01,
        candidate_radius: int = 1,
        normalize: bool = True,
        stable: bool = False,
    ) -> None:
        super().__init__()
        self.n_spixels = n_spixels
        self.n_iter = n_iter
        self.tau = tau
        self.candidate_radius = candidate_radius
        self.normalize = normalize
        self.stable = stable

    def forward(
        self, x: torch.Tensor, clst_feats: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Args:
            x (torch.Tensor): a tensor of shape (batch, channels, height, width)
            clst_feats (Optional[torch.Tensor]): a tensor of shape (batch, channels, height_s, width_s)
                                                 initial cluster features. if clst_feats is None, it is
                                                 initialized by averaging pixels in a uniform grid

        Returns:
            clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_s, width_s)
                                       height_s * width_s <= self.n_spixels
            p2s_assign (torch.Tensor): a tensor of shape (batch, (2*candidate_radius + 1)**2, height, width)
                                       a pixel-to-superpixel assignemnt matrix
            s2p_assign (torch.Tensor): a tensor of shape
                                       (batch, stride_h * stride_w * (2*candidate_radius + 1)**2, height_s, width_s)
                                       a superpixel-to-pixel assignemnt matrix
                                       if n_iter is 0, s2p_assign is None
        """
        height, width = x.shape[-2:]
        # initialize cluster features
        if clst_feats is None:
            height_s = int(math.sqrt(self.n_spixels * height / width))
            width_s = int(math.sqrt(self.n_spixels * width / height))
            stride_h = (height + height_s - 1) // height_s
            stride_w = (width + width_s - 1) // width_s
            stride = (stride_h, stride_w)
            clst_feats = F.adaptive_avg_pool2d(x, (height_s, width_s))
        else:
            height_s, width_s = clst_feats.shape[-2:]
            stride = ((height + height_s) // height_s, (width + width_s) // width_s)
        # normalize feature vectors so that their l2-norm is 1
        if self.normalize:
            x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # padding an image feature so that its height and width are divisible by stride values
        pad_x = (width_s - width % width_s) % width_s
        pad_y = (height_s - height % height_s) % height_s
        x = F.pad(x, (0, pad_x, 0, pad_y))
        # update cluster features
        s2p_assign = None
        for _ in range(self.n_iter):
            clst_feats, s2p_assign, _ = update_clst_feats(
                x, clst_feats, stride, self.tau, self.candidate_radius
            )
            if self.normalize:
                clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # compute a pixel-to-superpixel assignment
        p2s_assign, _ = compute_elem_to_center_assignment(
            clst_feats, x, stride, self.tau, self.candidate_radius
        )
        # remove the padding region
        if pad_y > 0:
            p2s_assign = p2s_assign[..., :-pad_y, :]
        if pad_x > 0:
            p2s_assign = p2s_assign[..., :-pad_x]
        return clst_feats, p2s_assign, s2p_assign

    def extra_repr(self):
        return (
            f"n_spixels={self.n_spixels}, \n "
            f"n_iter={self.n_iter}, \n "
            f"tau={self.tau}, \n "
            f"candidate_radius={self.candidate_radius}, \n"
        )
