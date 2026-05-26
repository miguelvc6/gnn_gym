import torch

from gnn_gym.data.loaders import load_dataset
from gnn_gym.models.normal_tree_backedge_gnn import (
    compute_tree_backedge_marker_sets,
    compute_tree_backedge_markers,
)


def test_tree_backedge_markers_split_tree_directions_and_back_edges() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 0, 3],
            [1, 0, 2, 1, 3, 2, 3, 0],
        ],
        dtype=torch.long,
    )

    markers = compute_tree_backedge_markers(edge_index, num_nodes=4)

    assert markers.down_edges.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert markers.up_edges.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert markers.back_edges.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert markers.span[-2:].tolist() == [1.0, 1.0]


def test_tree_backedge_marker_sets_generate_multiple_orders() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 0, 3],
            [1, 0, 2, 1, 3, 2, 3, 0],
        ],
        dtype=torch.long,
    )

    marker_sets = compute_tree_backedge_marker_sets(edge_index, num_nodes=4, num_orders=3)

    assert len(marker_sets) == 3
    assert all(markers.back_edges.shape == (8,) for markers in marker_sets)
    assert any(
        not torch.equal(marker_sets[0].down_edges, markers.down_edges)
        for markers in marker_sets[1:]
    )


def test_batched_tree_backedge_depths_are_normalized_per_graph() -> None:
    graph_edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    deeper_graph_edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )
    offset_deeper_graph = deeper_graph_edge_index + 3
    batched_edge_index = torch.cat([graph_edge_index, offset_deeper_graph], dim=1)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)

    single_graph_markers = compute_tree_backedge_marker_sets(
        graph_edge_index,
        num_nodes=3,
        num_orders=1,
    )[0]
    batched_markers = compute_tree_backedge_marker_sets(
        batched_edge_index,
        num_nodes=9,
        num_orders=1,
        batch=batch,
    )[0]

    first_graph_edges = slice(0, graph_edge_index.size(1))
    assert torch.equal(single_graph_markers.up_edges, batched_markers.up_edges[first_graph_edges])
    assert torch.equal(
        single_graph_markers.down_edges,
        batched_markers.down_edges[first_graph_edges],
    )
    assert torch.equal(
        single_graph_markers.back_edges,
        batched_markers.back_edges[first_graph_edges],
    )
    assert torch.allclose(
        single_graph_markers.source_depth,
        batched_markers.source_depth[first_graph_edges],
    )
    assert torch.allclose(
        single_graph_markers.target_depth,
        batched_markers.target_depth[first_graph_edges],
    )
    assert torch.allclose(single_graph_markers.span, batched_markers.span[first_graph_edges])


def test_tree_backedge_marker_role_modes_transform_roles() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 0, 3],
            [1, 0, 2, 1, 3, 2, 3, 0],
        ],
        dtype=torch.long,
    )

    true_markers = compute_tree_backedge_markers(edge_index, num_nodes=4)
    collapsed = compute_tree_backedge_markers(
        edge_index,
        num_nodes=4,
        edge_role_mode="collapsed",
    )
    tree_only = compute_tree_backedge_markers(
        edge_index,
        num_nodes=4,
        edge_role_mode="tree_only",
    )
    back_only = compute_tree_backedge_markers(
        edge_index,
        num_nodes=4,
        edge_role_mode="back_only",
    )
    shuffled = compute_tree_backedge_markers(
        edge_index,
        num_nodes=4,
        edge_role_mode="shuffled",
    )

    assert torch.equal(collapsed.up_edges, collapsed.down_edges)
    assert torch.equal(collapsed.down_edges, collapsed.back_edges)
    assert int(tree_only.back_edges.sum().item()) == 0
    assert int(back_only.up_edges.sum().item()) == 0
    assert int(back_only.down_edges.sum().item()) == 0
    assert int(back_only.back_edges.sum().item()) == int(true_markers.back_edges.sum().item())
    assert not torch.equal(shuffled.back_edges, true_markers.back_edges)


def test_tree_backedge_random_order_mode_changes_marker_choice() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5, 1, 4],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0, 4, 1],
        ],
        dtype=torch.long,
    )

    deterministic = compute_tree_backedge_markers(edge_index, num_nodes=6)
    random_order = compute_tree_backedge_markers(
        edge_index,
        num_nodes=6,
        dfs_order_mode="random",
    )

    assert not torch.equal(deterministic.down_edges, random_order.down_edges)


def test_normal_tree_backedge_dataset_smoke_loads() -> None:
    bundle = load_dataset(
        "normal-tree-backedge",
        {"dataset": {"name": "normal-tree-backedge", "num_graphs": 20}},
    )

    assert bundle.task == "graph_binary_classification"
    assert bundle.num_features == 1
    assert bundle.num_outputs == 1
    assert len(bundle.dataset) == 20
    assert set(float(graph.y.item()) for graph in bundle.dataset) == {0.0, 1.0}


def test_cycle_matching_variant_matches_obvious_graph_statistics() -> None:
    bundle = load_dataset(
        "normal-tree-backedge",
        {
            "dataset": {
                "name": "normal-tree-backedge",
                "num_graphs": 20,
                "variant": "cycle_matching_v4",
            }
        },
    )

    for graph in bundle.dataset:
        num_nodes = int(graph.num_nodes)
        undirected_edges = {
            tuple(sorted((int(src), int(dst))))
            for src, dst in graph.edge_index.t().tolist()
            if int(src) != int(dst)
        }
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for src, dst in undirected_edges:
            degrees[src] += 1
            degrees[dst] += 1

        assert num_nodes == 20
        assert len(undirected_edges) == 30
        assert degrees.tolist() == [3] * num_nodes
        assert graph.x.shape == (20, 1)
