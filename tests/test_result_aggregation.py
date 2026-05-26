import json
from pathlib import Path

import yaml

from gnn_gym.evaluation.aggregate import aggregate_runs, summarize_runs
from gnn_gym.utils.hashing import architecture_config_hash


def _write_fake_run(
    root: Path,
    name: str,
    seed: int,
    hidden_channels: int,
    val_metric: float,
) -> None:
    run_dir = root / name
    run_dir.mkdir()
    config = {
        "model": {"name": "gcn", "hidden_channels": hidden_channels},
        "dataset": {"name": "toy-node", "task": "node_classification", "metric": "accuracy"},
        "trainer": {"name": "full_batch_node"},
        "training": {"seed": seed, "lr": 0.01, "weight_decay": 0.0005},
    }
    digest = architecture_config_hash(config)
    config["config_hash"] = f"run-{name}"
    config["architecture_config_hash"] = digest
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "experiment_name": "test",
                "model": "gcn",
                "dataset": "toy-node",
                "task": "node_classification",
                "seed": seed,
                "best_epoch": 2,
                "device": "cpu",
                "git_commit": None,
                "config_hash": f"run-{name}",
                "architecture_config_hash": digest,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "metric_name": "accuracy",
                "best_val_metric": val_metric,
                "test_metric": val_metric - 0.1,
                "train_time_seconds": 1.5,
                "num_parameters": 42,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_fake_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "experiment_name": "test",
                "model": "gcn",
                "dataset": "toy-node",
                "task": "node_classification",
                "seed": 0,
                "best_epoch": 2,
                "device": "cpu",
                "git_commit": None,
                "config_hash": "abc123",
                "architecture_config_hash": "cfg123",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "metric_name": "accuracy",
                "best_val_metric": 0.7,
                "test_metric": 0.6,
                "train_time_seconds": 1.5,
                "num_parameters": 42,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "summary.csv"

    table = aggregate_runs(tmp_path, out)

    assert out.exists()
    assert len(table) == 1
    assert table.loc[0, "model"] == "gcn"
    assert table.loc[0, "test_metric"] == 0.6
    assert table.loc[0, "config_hash"] == "abc123"
    assert table.loc[0, "architecture_config_hash"] == "cfg123"


def test_architecture_config_hash_is_seed_independent() -> None:
    config_a = {
        "model": {"name": "gcn", "hidden_channels": 16},
        "dataset": {"name": "toy-node"},
        "trainer": {"name": "full_batch_node"},
        "training": {"seed": 0, "lr": 0.01, "weight_decay": 0.0005},
    }
    config_b = {
        "model": {"name": "gcn", "hidden_channels": 16},
        "dataset": {"name": "toy-node"},
        "trainer": {"name": "full_batch_node"},
        "training": {"seed": 1, "lr": 0.01, "weight_decay": 0.0005},
    }

    assert architecture_config_hash(config_a) == architecture_config_hash(config_b)


def test_architecture_config_hash_changes_for_model_hyperparameters() -> None:
    config_a = {
        "model": {"name": "gcn", "hidden_channels": 16},
        "dataset": {"name": "toy-node"},
        "trainer": {"name": "full_batch_node"},
        "training": {"seed": 0, "lr": 0.01, "weight_decay": 0.0005},
    }
    config_b = {
        "model": {"name": "gcn", "hidden_channels": 32},
        "dataset": {"name": "toy-node"},
        "trainer": {"name": "full_batch_node"},
        "training": {"seed": 0, "lr": 0.01, "weight_decay": 0.0005},
    }

    assert architecture_config_hash(config_a) != architecture_config_hash(config_b)


def test_summarize_runs_groups_by_architecture_config_hash(tmp_path: Path) -> None:
    _write_fake_run(tmp_path, "run-a", seed=0, hidden_channels=16, val_metric=0.7)
    _write_fake_run(tmp_path, "run-b", seed=1, hidden_channels=16, val_metric=0.8)
    _write_fake_run(tmp_path, "run-c", seed=0, hidden_channels=32, val_metric=0.9)

    table = aggregate_runs(tmp_path)
    summary = summarize_runs(table)

    assert len(summary) == 2
    assert sorted(summary["seeds"].tolist()) == [1, 2]
