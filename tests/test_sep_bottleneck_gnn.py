import torch

from gnn_gym.models.sep_bottleneck_gnn import compute_structural_markers


def test_separator_markers_detect_bridge_and_articulation_endpoint() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0, 2, 3, 3, 4],
            [1, 0, 2, 1, 0, 2, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )

    markers = compute_structural_markers(edge_index, num_nodes=5)

    assert markers.articulation_nodes.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert markers.separator_edges.tolist() == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
