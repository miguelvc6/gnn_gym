from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gnn_gym.utils.config import Config


def expand_sweep(config: Config) -> list[list[str]]:
    sweep = config.get("sweep", {})
    if not sweep:
        return [[]]
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a mapping of dotted keys to value lists")
    keys = list(sweep)
    values: list[list[Any]] = []
    for key in keys:
        raw_values = sweep[key]
        if not isinstance(raw_values, list):
            raise ValueError(f"sweep.{key} must be a list")
        values.append(raw_values)
    overrides: list[list[str]] = []
    for combo in itertools.product(*values):
        overrides.append(
            [f"{key}={json.dumps(value)}" for key, value in zip(keys, combo, strict=True)]
        )
    return overrides


def append_research_result(
    path: str | Path,
    run_dir: str | Path,
    overrides: list[str],
    status: str = "completed",
    notes: str = "",
) -> None:
    path = Path(path)
    path_exists = path.exists()
    run_dir = Path(run_dir)
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (run_dir / "final_metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    def value_or_empty(value: Any) -> Any:
        return "" if value is None else value

    if not path_exists:
        path.write_text(
            "timestamp\trun_id\tcommit\tmodel\tdataset\tseed\tmetric\tval_metric\t"
            "test_metric\tbest_epoch\ttrain_time_seconds\tstatus\toverrides\tnotes\n",
            encoding="utf-8",
        )
    row = [
        datetime.now(UTC).isoformat(),
        value_or_empty(metadata.get("run_id", run_dir.name)),
        value_or_empty(metadata.get("git_commit")),
        value_or_empty(metadata.get("model")),
        value_or_empty(metadata.get("dataset")),
        value_or_empty(metadata.get("seed")),
        value_or_empty(metrics.get("metric_name")),
        value_or_empty(metrics.get("best_val_metric")),
        value_or_empty(metrics.get("test_metric")),
        value_or_empty(metrics.get("best_epoch")),
        value_or_empty(metrics.get("train_time_seconds")),
        status,
        " ".join(overrides),
        notes,
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(value).replace("\t", " ") for value in row) + "\n")
