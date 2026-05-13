from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def collect_runs(runs_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for final_metrics_path in sorted(Path(runs_dir).glob("*/final_metrics.json")):
        run_dir = final_metrics_path.parent
        metadata_path = run_dir / "metadata.json"
        with final_metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        rows.append(
            {
                "run_id": metadata.get("run_id", run_dir.name),
                "experiment_name": metadata.get("experiment_name", "manual"),
                "model": metadata.get("model"),
                "dataset": metadata.get("dataset"),
                "task": metadata.get("task"),
                "metric_name": metrics.get("metric_name"),
                "seed": metadata.get("seed"),
                "best_epoch": metadata.get("best_epoch", metrics.get("best_epoch")),
                "val_metric": metrics.get("best_val_metric"),
                "test_metric": metrics.get("test_metric"),
                "train_time_seconds": metrics.get("train_time_seconds"),
                "num_parameters": metrics.get("num_parameters"),
                "device": metadata.get("device"),
                "git_commit": metadata.get("git_commit"),
                "config_hash": metadata.get("config_hash"),
                "status": metadata.get("status", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_runs(runs_dir: str | Path, out: str | Path | None = None) -> pd.DataFrame:
    table = collect_runs(runs_dir)
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            table.to_parquet(out_path, index=False)
        else:
            table.to_csv(out_path, index=False)
    return table
