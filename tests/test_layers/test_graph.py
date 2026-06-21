import torch
from spixrwkv7.layers.graph import build_knn_graph, q_shift_graph_multihead


def test_build_knn_graph_single():
    """Test KNN graph building for a single set of centroids."""
    # 4 centroids in a square
    centroids = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    neighbors, _ = build_knn_graph(centroids, k=2)
    assert neighbors.shape == (4, 2)
    # Ensure a node doesn't select itself as a neighbor
    for i in range(4):
        assert i not in neighbors[i]


def test_build_knn_graph_batched():
    """Test KNN graph building for batched centroids."""
    B, N = 2, 5
    centroids = torch.rand(B, N, 2)
    neighbors, _ = build_knn_graph(centroids, k=3)
    assert neighbors.shape == (B, N, 3)


def test_q_shift_graph_multihead_logic():
    """Verify that Graph Q-Shift correctly shifts tokens along graph edges."""
    B, N, C = 1, 4, 16
    head_dim = 16

    # Graph: Node 0 connects to 1, 2. Node 1 connects to 0, 3, etc.
    neighbors = torch.tensor([[[1, 2], [0, 3], [0, 3], [1, 2]]])  # [B, N, K]

    # Fill input with node IDs + 1 so we can track movement
    x = torch.zeros(B, N, C)
    for i in range(N):
        x[0, i, :] = i + 1

    out = q_shift_graph_multihead(x, neighbors, head_dim=head_dim, with_cls_token=False)

    # Group 0 (channels 0-7): should come from neighbor index 0
    # Node 0's neighbor 0 is Node 1. So out[0, 0, 0] should be x[0, 1, 0] = 2
    assert out[0, 0, 0].item() == 2.0
    # Node 1's neighbor 0 is Node 0. So out[0, 1, 0] should be x[0, 0, 0] = 1
    assert out[0, 1, 0].item() == 1.0

    # Group 1 (channels 8-15): should come from neighbor index 1
    # Node 0's neighbor 1 is Node 2. So out[0, 0, 8] should be x[0, 2, 8] = 3
    assert out[0, 0, 8].item() == 3.0
    # Node 3's neighbor 1 is Node 2. So out[0, 3, 8] should be x[0, 2, 8] = 3
    assert out[0, 3, 8].item() == 3.0


def test_q_shift_graph_multihead_cls_token():
    """Verify CLS token is excluded from shifting and preserved."""
    B, N, C = 1, 4, 16
    head_dim = 16
    neighbors = torch.zeros(B, N, 2, dtype=torch.long)

    x = torch.randn(B, N + 1, C)
    cls_token = x[:, -1:, :]

    out = q_shift_graph_multihead(x, neighbors, head_dim=head_dim, with_cls_token=True)

    # The last token should be exactly the unmodified CLS token
    assert torch.allclose(out[:, -1:, :], cls_token)


def test_q_shift_graph_multihead_head_grouping():
    """Verify each head-group independently gathers from its designated neighbor.

    With C=16, head_dim=16 (1 head), K=2, group_size=8:
      Group 0 (channels 0-7)  ← neighbor 0
      Group 1 (channels 8-15) ← neighbor 1
    """
    B, N, C = 1, 4, 16
    head_dim = 16
    neighbors = torch.tensor([[[1, 2], [0, 3], [0, 3], [1, 2]]])

    x = torch.zeros(B, N, C)
    for i in range(N):
        x[0, i, :] = i + 1  # node k has value k+1 in all channels

    out = q_shift_graph_multihead(
        x, neighbors, head_dim=head_dim, with_cls_token=False
    )

    # Node 0, neighbor 0 = Node 1, neighbor 1 = Node 2
    # Group 0 (ch 0-7)  ← neighbor 0 value = 2
    assert (out[0, 0, :8] == 2.0).all()
    # Group 1 (ch 8-15) ← neighbor 1 value = 3
    assert (out[0, 0, 8:] == 3.0).all()

    # Node 1, neighbor 0 = Node 0, neighbor 1 = Node 3
    # Group 0 (ch 0-7)  ← neighbor 0 value = 1
    assert (out[0, 1, :8] == 1.0).all()
    # Group 1 (ch 8-15) ← neighbor 1 value = 4
    assert (out[0, 1, 8:] == 4.0).all()
