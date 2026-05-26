import pytest
import torch

from gnn_gym.models import build_model


@pytest.mark.parametrize(
    "model_name",
    [
        "mlp",
        "gcn",
        "gat",
        "gin",
        "bethe_gnn",
        "confidence_appnp_net",
        "decimation_gnn",
        "dual_primal_gnn",
        "equilibrium_belief_gnn",
        "revision_gnn",
        "res_gin",
        "jk_gcn",
        "appnp_net",
        "gcn2_net",
        "gpr_gnn",
        "kikuchi_gnn",
        "loop_corrected_gnn",
        "gatv2",
        "gated_appnp_net",
        "cavity_gnn",
        "frustration_gnn",
        "nb_appnp_net",
        "nb_belief_gnn",
        "nb_light_gnn",
        "res_appnp_net",
        "region_collapse_gnn",
        "entropy_gated_gnn",
        "rign_gnn",
        "temp_ladder_gnn",
        "survey_gnn",
        "walk_belief_transformer",
    ],
)
def test_model_output_shape(model_name: str) -> None:
    x = torch.randn(12, 5)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
        ],
        dtype=torch.long,
    )
    config = {
        "model": {
            "name": model_name,
            "hidden_channels": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "activation": "elu" if model_name in {"gat", "gatv2"} else "relu",
            "norm": "none",
            "heads": 2,
            "attention_dropout": 0.0,
            "temperatures": [0.5, 1.0],
            "num_particles": 2,
            "residual_weight": 0.5,
            "nb_steps": 2,
            "propagation_steps": 3,
            "alpha": 0.1,
            "gate_init": -4.0,
            "max_steps": 2,
            "num_regions": 4,
            "max_triangles": 8,
            "num_rounds": 2,
            "walk_length": 3,
        }
    }
    model = build_model(
        model_name,
        in_channels=5,
        out_channels=3,
        task="node_classification",
        config=config,
    )
    model.eval()

    logits = model(x, edge_index)

    assert logits.shape == (12, 3)


def test_mean_max_add_graph_pooling_shape() -> None:
    x = torch.randn(12, 5)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
        ],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    config = {
        "model": {
            "name": "gcn",
            "hidden_channels": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "activation": "relu",
            "norm": "none",
            "pooling": "mean_max_add",
            "head_hidden_channels": 8,
        }
    }
    model = build_model(
        "gcn",
        in_channels=5,
        out_channels=2,
        task="graph_binary_classification",
        config=config,
    )
    model.eval()

    logits = model(x, edge_index, batch)

    assert logits.shape == (2, 2)
