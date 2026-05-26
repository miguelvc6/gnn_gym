from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from gnn_gym.data.loaders import load_dataset
from gnn_gym.evaluation.aggregate import (
    aggregate_runs,
    export_latex,
    export_markdown,
    summarize_runs,
    summarize_runs_by_model,
)
from gnn_gym.experiments.sweep import append_research_result, expand_sweep
from gnn_gym.models import build_model
from gnn_gym.registry import TRAINER_REGISTRY, ensure_registrations
from gnn_gym.training.trainer import git_commit
from gnn_gym.utils.config import deep_merge, load_run_config, load_yaml, parse_override, set_dotted
from gnn_gym.utils.device import get_device
from gnn_gym.utils.hashing import architecture_config_hash, config_hash
from gnn_gym.utils.paths import make_run_dir
from gnn_gym.utils.seed import set_seed

app = typer.Typer(no_args_is_help=True)
console = Console()


class Tee:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_run_logs(run_dir: Path) -> object:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with stdout_path.open("a", encoding="utf-8") as stdout_file:
        with stderr_path.open("a", encoding="utf-8") as stderr_file:
            sys.stdout = Tee(original_stdout, stdout_file)  # type: ignore[assignment]
            sys.stderr = Tee(original_stderr, stderr_file)  # type: ignore[assignment]
            try:
                yield
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr


@app.command()
def train(
    model: Annotated[str | None, typer.Option("--model")] = None,
    dataset: Annotated[str | None, typer.Option("--dataset")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    override: Annotated[list[str] | None, typer.Option("--override")] = None,
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("results/runs"),
    device: Annotated[str, typer.Option("--device")] = "auto",
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
) -> Path:
    run_dir = run_training(
        model_name=model,
        dataset_name=dataset,
        seed=seed,
        overrides=override or [],
        runs_dir=runs_dir,
        device_name=device,
        resume_dir=resume,
    )
    console.print(str(run_dir))
    return run_dir


def run_training(
    model_name: str | None,
    dataset_name: str | None,
    seed: int,
    overrides: list[str] | None = None,
    runs_dir: str | Path = "results/runs",
    device_name: str = "auto",
    base_config: dict[str, object] | None = None,
    resume_dir: str | Path | None = None,
    capture_logs: bool = True,
) -> Path:
    ensure_registrations()
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            config_path = run_dir / "config.yaml"
        config = load_yaml(config_path)
        model_name = str(model_name or config["model"]["name"])
        dataset_name = str(dataset_name or config["dataset"]["name"])
        if overrides:
            for raw in overrides:
                key, value = parse_override(raw)
                set_dotted(config, key, value)
    else:
        if model_name is None or dataset_name is None:
            raise typer.BadParameter("--model and --dataset are required unless --resume is used")
        config = load_run_config(model_name, dataset_name, overrides)
        run_dir = None
    if base_config:
        config = deep_merge(config, base_config)
    config.setdefault("training", {})["seed"] = seed
    digest = config_hash(config)
    config["config_hash"] = digest
    config["architecture_config_hash"] = architecture_config_hash(config)

    set_seed(seed)
    if run_dir is None:
        run_dir = make_run_dir(runs_dir, str(model_name), str(dataset_name), seed, digest)

    try:
        context = tee_run_logs(run_dir) if capture_logs else nullcontext()
        with context:
            dataset_bundle = load_dataset(str(dataset_name), config)
            model = build_model(
                name=str(model_name),
                in_channels=dataset_bundle.num_features,
                out_channels=dataset_bundle.num_outputs,
                task=dataset_bundle.task,
                config=config,
            )
            trainer_name = str(config.get("trainer", {}).get("name") or dataset_bundle.trainer)
            if trainer_name == "full_batch_node" and dataset_bundle.trainer != "full_batch_node":
                trainer_name = dataset_bundle.trainer
                config.setdefault("trainer", {})["name"] = trainer_name
            if trainer_name not in TRAINER_REGISTRY:
                raise KeyError(f"Unknown trainer: {trainer_name}")

            trainer = TRAINER_REGISTRY[trainer_name](
                model=model,
                dataset=dataset_bundle,
                config=config,
                run_dir=run_dir,
                device=get_device(device_name),
                resume_from=resume_dir,
            )
            trainer.run()
    except Exception as error:
        write_failed_run(run_dir, config, str(model_name), str(dataset_name), seed, error)
        raise
    return run_dir


@contextmanager
def nullcontext() -> object:
    yield


def write_failed_run(
    run_dir: Path,
    config: dict[str, object],
    model_name: str,
    dataset_name: str,
    seed: int,
    error: BaseException,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "resolved_config.yaml"):
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=True)
    metadata = {
        "run_id": run_dir.name,
        "experiment_name": config.get("experiment", {}).get("name", "manual"),
        "model": model_name,
        "dataset": dataset_name,
        "task": config.get("dataset", {}).get("task"),
        "seed": seed,
        "git_commit": git_commit(),
        "config_hash": config.get("config_hash"),
        "architecture_config_hash": config.get("architecture_config_hash"),
        "device": device_name_from_config(config),
        "best_epoch": 0,
        "status": "failed",
    }
    final_metrics = {
        "metric_name": config.get("dataset", {}).get("metric"),
        "best_epoch": 0,
        "best_val_metric": None,
        "test_metric": None,
        "error": f"{type(error).__name__}: {error}",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )
    with (run_dir / "stderr.log").open("a", encoding="utf-8") as handle:
        handle.write("".join(traceback.format_exception(error)))


def device_name_from_config(config: dict[str, object]) -> str:
    return str(config.get("device", "unknown"))


@app.command("run-experiment")
def run_experiment(
    config: Annotated[Path, typer.Option("--config")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("results/runs"),
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    experiment_config = load_yaml(config)
    models = experiment_config.get("models", [])
    datasets = experiment_config.get("datasets", [])
    seeds = experiment_config.get("experiment", {}).get("seeds", [0])
    shared = {
        key: value
        for key, value in experiment_config.items()
        if key not in {"models", "datasets"}
    }
    for model_name in models:
        for dataset_name in datasets:
            for seed in seeds:
                run_dir = run_training(
                    model_name=str(model_name),
                    dataset_name=str(dataset_name),
                    seed=int(seed),
                    runs_dir=runs_dir,
                    device_name=device,
                    base_config=shared,
                )
                console.print(str(run_dir))


@app.command("run-sweep")
def run_sweep(
    config: Annotated[Path, typer.Option("--config")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("results/runs"),
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    sweep_config = load_yaml(config)
    models = sweep_config.get("models", [])
    datasets = sweep_config.get("datasets", [])
    seeds = sweep_config.get("experiment", {}).get("seeds", [0])
    override_grid = expand_sweep(sweep_config)
    shared = {
        key: value
        for key, value in sweep_config.items()
        if key not in {"models", "datasets", "sweep"}
    }
    experiment = sweep_config.get("experiment", {})
    write_research_results = bool(experiment.get("write_research_results", False))
    research_results_path = Path(experiment.get("research_results_path", "research_results.tsv"))
    for model_name in models:
        for dataset_name in datasets:
            for seed in seeds:
                for sweep_overrides in override_grid:
                    try:
                        run_dir = run_training(
                            model_name=str(model_name),
                            dataset_name=str(dataset_name),
                            seed=int(seed),
                            overrides=sweep_overrides,
                            runs_dir=runs_dir,
                            device_name=device,
                            base_config=shared,
                        )
                        if write_research_results:
                            append_research_result(
                                research_results_path,
                                run_dir,
                                sweep_overrides,
                                status="completed",
                            )
                        console.print(str(run_dir))
                    except Exception as error:
                        console.print(
                            f"FAILED {model_name}/{dataset_name}/seed-{seed}: {error}",
                            style="red",
                        )


@app.command()
def aggregate(
    runs: Annotated[Path, typer.Option("--runs")] = Path("results/runs"),
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    if out is None:
        out = Path("results/tables/all_runs.csv")
    table = aggregate_runs(runs, out)
    console.print(f"Wrote {len(table)} rows to {out}")


@app.command()
def evaluate(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
) -> None:
    final_metrics = run_dir / "final_metrics.json"
    if not final_metrics.exists():
        raise typer.BadParameter(f"No final_metrics.json found in {run_dir}")
    console.print(final_metrics.read_text(encoding="utf-8"))


@app.command("export-tables")
def export_tables(
    input: Annotated[Path, typer.Option("--input")],
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("results/tables"),
) -> None:
    if not input.exists():
        raise typer.BadParameter(f"Input does not exist: {input}")
    import pandas as pd

    table = pd.read_parquet(input) if input.suffix == ".parquet" else pd.read_csv(input)
    summary = summarize_runs(table)
    model_summary = summarize_runs_by_model(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"{input.stem}_by_config_mean_std.csv"
    summary_md = out_dir / f"{input.stem}_by_config_table.md"
    summary_tex = out_dir / f"{input.stem}_by_config_table.tex"
    model_summary_csv = out_dir / f"{input.stem}_by_model_mean_std.csv"
    model_summary_md = out_dir / f"{input.stem}_by_model_table.md"
    model_summary_tex = out_dir / f"{input.stem}_by_model_table.tex"
    summary.to_csv(summary_csv, index=False)
    export_markdown(summary, summary_md)
    export_latex(summary, summary_tex)
    model_summary.to_csv(model_summary_csv, index=False)
    export_markdown(model_summary, model_summary_md)
    export_latex(model_summary, model_summary_tex)
    console.print(
        "Wrote "
        f"{summary_csv}, {summary_md}, {summary_tex}, "
        f"{model_summary_csv}, {model_summary_md}, and {model_summary_tex}"
    )
