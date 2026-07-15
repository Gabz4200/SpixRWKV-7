"""Graph-based helpers: KNN graph construction and multi-head Q-Shift for superpixel grids."""

from typing import Tuple

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

    # In-place diagonal fill avoids allocating an [B, N, N] bool mask.
    dists.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))

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
    num_extra_tokens: int = 0,
    num_prepend_tokens: int = 0,
) -> torch.Tensor:
    """Graph-based Q-Shift for superpixel or irregular grids.

    Supports batched graphs [B, N, K] for data-dependent topologies.
    Prepended tokens (e.g. register tokens) and appended tokens (e.g. CLS)
    are excluded from the graph shift and passed through unchanged.
    """
    B, N_total, C = input.shape
    assert C % head_dim == 0, f"C={C} not divisible by head_dim={head_dim}"
    n_head = C // head_dim

    if neighbors.dim() == 2:
        neighbors = neighbors.unsqueeze(0).expand(B, -1, -1)

    K = neighbors.shape[2]
    assert head_dim % K == 0, f"head_dim={head_dim} must be divisible by K={K}"
    group_size = head_dim // K

    # Strip prepend tokens (register tokens)
    prepend_tokens = None
    if num_prepend_tokens > 0:
        prepend_tokens = input[:, :num_prepend_tokens, :]
        input = input[:, num_prepend_tokens:, :]
        N_total -= num_prepend_tokens

    # Strip append tokens (CLS)
    if num_extra_tokens == 0 and with_cls_token:
        num_extra_tokens = 1

    extra_tokens = None
    if num_extra_tokens > 0:
        extra_tokens = input[:, -num_extra_tokens:, :]
        input = input[:, :-num_extra_tokens, :]
        N = N_total - num_extra_tokens
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

    # Re-assemble: prepend + shifted + append
    parts = []
    if prepend_tokens is not None:
        parts.append(prepend_tokens)
    parts.append(output)
    if extra_tokens is not None:
        parts.append(extra_tokens)
    return torch.cat(parts, dim=1)
