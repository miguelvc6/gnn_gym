import json
from pathlib import Path

from gnn_gym.cli import run_training


def test_toy_training_writes_artifacts(tmp_path: Path) -> None:
    run_dir = run_training(
        model_name="gcn",
        dataset_name="toy-node",
        seed=0,
        overrides=["training.max_epochs=2", "training.patience=2", "model.hidden_channels=8"],
        runs_dir=tmp_path,
        device_name="cpu",
    )

    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "final_metrics.json").exists()
    metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric_name"] == "accuracy"
    assert metrics["best_epoch"] >= 1


def test_toy_graph_training_writes_artifacts(tmp_path: Path) -> None:
    run_dir = run_training(
        model_name="gin",
        dataset_name="toy-graph",
        seed=0,
        overrides=[
            "training.max_epochs=2",
            "training.patience=2",
            "model.hidden_channels=8",
            "model.num_layers=2",
        ],
        runs_dir=tmp_path,
        device_name="cpu",
    )

    assert (run_dir / "final_metrics.json").exists()
    metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric_name"] == "average_precision"


def test_toy_neighbor_training_writes_artifacts(tmp_path: Path) -> None:
    run_dir = run_training(
        model_name="gcn",
        dataset_name="toy-node",
        seed=0,
        overrides=[
            "trainer.name=neighbor_node",
            "trainer.batch_size=8",
            "training.max_epochs=2",
            "training.patience=2",
            "model.hidden_channels=8",
            "model.num_layers=2",
        ],
        runs_dir=tmp_path,
        device_name="cpu",
    )

    assert (run_dir / "final_metrics.json").exists()
    metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric_name"] == "accuracy"


def test_toy_link_training_writes_artifacts(tmp_path: Path) -> None:
    run_dir = run_training(
        model_name="gcn",
        dataset_name="toy-link",
        seed=0,
        overrides=[
            "training.max_epochs=2",
            "training.patience=2",
            "model.hidden_channels=8",
            "model.num_layers=2",
        ],
        runs_dir=tmp_path,
        device_name="cpu",
    )

    assert (run_dir / "final_metrics.json").exists()
    metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert metrics["metric_name"] == "hits@50"
