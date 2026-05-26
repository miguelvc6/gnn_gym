import torch

from gnn_gym.models.tree_pack_gnn import compute_tree_pack


def test_tree_pack_generates_separate_tree_views() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 4, 1, 3],
            [1, 0, 2, 1, 3, 2, 4, 3, 4, 0, 3, 1],
        ],
        dtype=torch.long,
    )

    pack = compute_tree_pack(edge_index, num_nodes=5, num_trees=4)

    assert pack.tree_edge_masks.shape == (4, edge_index.size(1))
    assert pack.source_depth.shape == (4, edge_index.size(1))
    assert pack.target_depth.shape == (4, edge_index.size(1))
    assert all(int(mask.sum().item()) == 8 for mask in pack.tree_edge_masks)
    assert any(
        not torch.equal(pack.tree_edge_masks[0], pack.tree_edge_masks[idx]) for idx in range(1, 4)
    )


def test_tree_pack_batched_matches_single_graph_first_component() -> None:
    graph_edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    second = graph_edge_index + 3
    batched_edge_index = torch.cat([graph_edge_index, second], dim=1)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    single = compute_tree_pack(graph_edge_index, num_nodes=3, num_trees=4)
    batched = compute_tree_pack(batched_edge_index, num_nodes=6, num_trees=4, batch=batch)

    first_edges = slice(0, graph_edge_index.size(1))
    assert torch.equal(single.tree_edge_masks, batched.tree_edge_masks[:, first_edges])
    assert torch.allclose(single.source_depth, batched.source_depth[:, first_edges])
    assert torch.allclose(single.target_depth, batched.target_depth[:, first_edges])


def test_tree_pack_start_index_selects_different_single_view() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 4, 1, 3],
            [1, 0, 2, 1, 3, 2, 4, 3, 4, 0, 3, 1],
        ],
        dtype=torch.long,
    )

    first = compute_tree_pack(edge_index, num_nodes=5, num_trees=1, tree_start_idx=0)
    third = compute_tree_pack(edge_index, num_nodes=5, num_trees=1, tree_start_idx=2)

    assert not torch.equal(first.tree_edge_masks, third.tree_edge_masks)
