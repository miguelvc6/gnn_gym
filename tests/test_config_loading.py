from gnn_gym.utils.config import load_run_config


def test_load_run_config_with_override() -> None:
    config = load_run_config("gcn", "toy-node", ["training.lr=0.005", "model.hidden_channels=16"])

    assert config["model"]["name"] == "gcn"
    assert config["dataset"]["name"] == "toy-node"
    assert config["training"]["lr"] == 0.005
    assert config["model"]["hidden_channels"] == 16
