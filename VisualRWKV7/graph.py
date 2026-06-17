"""Graph-based helpers: KNN graph construction and multi-head Q-Shift for superpixel grids."""

import torch

HEAD_SIZE = 64


def build_knn_graph(centroids: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Builds a K-Nearest Neighbors graph from superpixel centroids.
    Supports both single [N, 2] and batched [B, N, 2] centroids.
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

    _, neighbors = torch.topk(dists, k, dim=2, largest=False)  # [B, N, k]

    if squeeze:
        neighbors = neighbors.squeeze(0)
    return neighbors


def q_shift_graph_multihead(
    input: torch.Tensor,
    neighbors: torch.Tensor,
    head_dim: int = HEAD_SIZE,
    with_cls_token: bool = False,
) -> torch.Tensor:
    """Graph-based Q-Shift for superpixel or irregular grids.
    Supports batched graphs [B, N, K] for data-dependent topologies.
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

    x = input.view(B, N, n_head, head_dim)
    output = torch.zeros_like(x)
    clamped_neighbors = neighbors.clamp(min=0)

    for k in range(K):
        neighbor_idx = clamped_neighbors[:, :, k]
        x_group = x[:, :, :, k * group_size : (k + 1) * group_size]
        idx = (
            neighbor_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_head, group_size)
        )
        gathered_group = torch.gather(x_group, 1, idx)
        valid_k = (neighbors[:, :, k] != -1).view(B, N, 1, 1)
        gathered_group = gathered_group * valid_k
        output[:, :, :, k * group_size : (k + 1) * group_size] = gathered_group

    output = output.view(B, N, C)
    if with_cls_token:
        assert cls_tokens is not None
        output = torch.cat((output, cls_tokens), dim=1)
    return output
