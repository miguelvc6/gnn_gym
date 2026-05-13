import json
from pathlib import Path

from gnn_gym.evaluation.aggregate import aggregate_runs


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
