import pytest
import torch

from gnn_gym.models import build_model


@pytest.mark.parametrize("model_name", ["mlp", "gcn", "gat", "gin"])
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
            "activation": "relu" if model_name != "gat" else "elu",
            "norm": "none",
            "heads": 2,
            "attention_dropout": 0.0,
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
