import json
from pathlib import Path

import pytest
import torch

from gnn_gym.cli import run_sweep, run_training
from gnn_gym.experiments.sweep import expand_sweep
from gnn_gym.training.optimizers import build_optimizer
from gnn_gym.training.schedulers import build_scheduler
from gnn_gym.utils.config import load_yaml


def test_scheduler_creation() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(
        model,
        {"training": {"lr": 0.01, "weight_decay": 0.0, "scheduler": "cosine", "max_epochs": 3}},
    )

    scheduler = build_scheduler(optimizer, {"training": {"scheduler": "cosine", "max_epochs": 3}})

    assert scheduler is not None


def test_prediction_saving_and_resume(tmp_path: Path) -> None:
    run_dir = run_training(
        model_name="gcn",
        dataset_name="toy-node",
        seed=0,
        overrides=[
            "training.max_epochs=2",
            "training.patience=2",
            "model.hidden_channels=8",
            "artifacts.save_predictions=true",
        ],
        runs_dir=tmp_path,
        device_name="cpu",
    )

    assert (run_dir / "predictions.pt").exists()
    resumed_dir = run_training(
        model_name=None,
        dataset_name=None,
        seed=0,
        overrides=["training.max_epochs=3", "training.patience=3"],
        runs_dir=tmp_path,
        device_name="cpu",
        resume_dir=run_dir,
    )
    metrics = json.loads((resumed_dir / "final_metrics.json").read_text(encoding="utf-8"))

    assert resumed_dir == run_dir
    assert metrics["best_epoch"] >= 1
    assert (run_dir / "stdout.log").exists()
    assert (run_dir / "stderr.log").exists()


def test_failure_metadata_is_written(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        run_training(
            model_name="gcn",
            dataset_name="toy-node",
            seed=0,
            overrides=["trainer.name=missing_trainer"],
            runs_dir=tmp_path,
            device_name="cpu",
        )
    run_dirs = list(tmp_path.iterdir())

    assert len(run_dirs) == 1
    metadata = json.loads((run_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert (run_dirs[0] / "final_metrics.json").exists()


def test_expand_sweep() -> None:
    overrides = expand_sweep(
        {
            "sweep": {
                "model.hidden_channels": [8, 16],
                "training.lr": [0.01],
            }
        }
    )

    assert overrides == [
        ["model.hidden_channels=8", "training.lr=0.01"],
        ["model.hidden_channels=16", "training.lr=0.01"],
    ]


def test_run_sweep_writes_runs_and_research_results(tmp_path: Path) -> None:
    config_path = tmp_path / "sweep.yaml"
    research_results = tmp_path / "research_results.tsv"
    config_path.write_text(
        f"""
experiment:
  name: test_sweep
  seeds: [0]
  write_research_results: true
  research_results_path: {research_results}
models: [gcn]
datasets: [toy-node]
training:
  max_epochs: 1
  patience: 1
sweep:
  model.hidden_channels: [8, 16]
""",
        encoding="utf-8",
    )

    run_sweep(config_path, runs_dir=tmp_path / "runs", device="cpu")

    assert len(list((tmp_path / "runs").iterdir())) == 2
    assert len(research_results.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert load_yaml(config_path)["experiment"]["name"] == "test_sweep"
