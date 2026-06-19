"""Graph-based helpers: KNN graph construction and multi-head Q-Shift for superpixel grids."""

from typing import Optional, Tuple

import torch

HEAD_SIZE = 64


def build_knn_graph(
    centroids: torch.Tensor, k: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds a K-Nearest Neighbors graph from superpixel centroids.
    Supports both single [N, 2] and batched [B, N, 2] centroids.
    Returns (neighbors, knn_dists).
    """
    squeeze = False
    if centroids.dim() == 2:
        centroids = centroids.unsqueeze(0)
        squeeze = True

    B, N, _ = centroids.shape
    centroids = centroids.float()
    dists = torch.cdist(centroids, centroids)  # [B, N, N]

    mask = (
        torch.eye(N, dtype=torch.bool, device=centroids.device)
        .unsqueeze(0)
        .expand(B, -1, -1)
    )
    dists = dists.masked_fill(mask, float("inf"))

    knn_dists, neighbors = torch.topk(
        dists, k, dim=2, largest=False
    )  # [B, N, k], [B, N, k]

    if squeeze:
        neighbors = neighbors.squeeze(0)
        knn_dists = knn_dists.squeeze(0)
    return neighbors, knn_dists


def q_shift_graph_multihead(
    input: torch.Tensor,
    neighbors: torch.Tensor,
    head_dim: int = HEAD_SIZE,
    with_cls_token: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Graph-based Q-Shift for superpixel or irregular grids.
    Supports batched graphs [B, N, K] for data-dependent topologies.

    Extra keyword arguments (dists, sigma) are accepted for call-site
    compatibility but NOT used in the pure gather-based Q-shift.
    """
    B, N_total, C = input.shape
    assert C % head_dim == 0, f"C={C} not divisible by head_dim={head_dim}"
    n_head = C // head_dim

    if neighbors.dim() == 2:
        neighbors = neighbors.unsqueeze(0).expand(B, -1, -1)

    K = neighbors.shape[2]
    assert head_dim % K == 0, f"head_dim={head_dim} must be divisible by K={K}"
    group_size = head_dim // K

    cls_tokens = None
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
        N = N_total - 1
    else:
        N = N_total

    assert neighbors.shape[1] == N, (
        f"neighbors length {neighbors.shape[1]} must match N={N}"
    )

    x = input.view(B, N, n_head, K, group_size)
    clamped_neighbors = neighbors.clamp(min=0)

    gather_idx = (
        clamped_neighbors.view(B, N, 1, K, 1)
        .expand(B, N, n_head, K, group_size)
    )

    output = torch.gather(x, 1, gather_idx)

    valid_mask = (neighbors != -1).view(B, N, 1, K, 1)
    output = output * valid_mask

    output = output.view(B, N, C)
    if with_cls_token:
        assert cls_tokens is not None
        output = torch.cat((output, cls_tokens), dim=1)
    return output
