from gnn_gym.registry import (
    DATASET_REGISTRY,
    EVALUATOR_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
    ensure_registrations,
)


def test_core_registries_are_populated() -> None:
    ensure_registrations()
    ensure_registrations()

    assert {
        "mlp",
        "gcn",
        "gat",
        "gin",
        "appnp_net",
        "gpr_gnn",
        "nb_belief_gnn",
        "normal_tree_backedge_gnn",
        "tree_pack_gnn",
        "sep_bottleneck_gnn",
    }.issubset(MODEL_REGISTRY)
    assert "toy-node" in DATASET_REGISTRY
    assert "normal-tree-backedge" in DATASET_REGISTRY
    assert "full_batch_node" in TRAINER_REGISTRY
    assert "accuracy" in EVALUATOR_REGISTRY
