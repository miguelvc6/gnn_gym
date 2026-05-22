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


def summarize_runs(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    grouped = table.groupby(["task", "dataset", "model", "metric_name"], dropna=False)
    return grouped.agg(
        seeds=("seed", "count"),
        val_mean=("val_metric", "mean"),
        val_std=("val_metric", "std"),
        test_mean=("test_metric", "mean"),
        test_std=("test_metric", "std"),
        train_time_seconds_mean=("train_time_seconds", "mean"),
        num_parameters_mean=("num_parameters", "mean"),
    ).reset_index()


def export_markdown(table: pd.DataFrame, out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_latex(table: pd.DataFrame, out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(table.to_latex(index=False), encoding="utf-8")
