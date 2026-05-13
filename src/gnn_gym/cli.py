from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from gnn_gym.data.loaders import load_dataset
from gnn_gym.evaluation.aggregate import aggregate_runs
from gnn_gym.models import build_model
from gnn_gym.registry import TRAINER_REGISTRY, ensure_registrations
from gnn_gym.utils.config import deep_merge, load_run_config, load_yaml
from gnn_gym.utils.device import get_device
from gnn_gym.utils.hashing import config_hash
from gnn_gym.utils.paths import make_run_dir
from gnn_gym.utils.seed import set_seed

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def train(
    model: Annotated[str, typer.Option("--model")],
    dataset: Annotated[str, typer.Option("--dataset")],
    seed: Annotated[int, typer.Option("--seed")] = 0,
    override: Annotated[list[str] | None, typer.Option("--override")] = None,
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path("results/runs"),
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> Path:
    run_dir = run_training(
        model_name=model,
        dataset_name=dataset,
        seed=seed,
        overrides=override or [],
        runs_dir=runs_dir,
        device_name=device,
    )
    console.print(str(run_dir))
    return run_dir


def run_training(
    model_name: str,
    dataset_name: str,
    seed: int,
    overrides: list[str] | None = None,
    runs_dir: str | Path = "results/runs",
    device_name: str = "auto",
    base_config: dict[str, object] | None = None,
) -> Path:
    ensure_registrations()
    config = load_run_config(model_name, dataset_name, overrides)
    if base_config:
        config = deep_merge(config, base_config)
    config.setdefault("training", {})["seed"] = seed
    digest = config_hash(config)
    config["config_hash"] = digest

    set_seed(seed)
    dataset_bundle = load_dataset(dataset_name, config)
    model = build_model(
        name=model_name,
        in_channels=dataset_bundle.num_features,
        out_channels=dataset_bundle.num_outputs,
        task=dataset_bundle.task,
        config=config,
    )
    trainer_name = str(config.get("trainer", {}).get("name") or dataset_bundle.trainer)
    if trainer_name not in TRAINER_REGISTRY:
        raise KeyError(f"Unknown trainer: {trainer_name}")

    run_dir = make_run_dir(runs_dir, model_name, dataset_name, seed, digest)
    trainer = TRAINER_REGISTRY[trainer_name](
        model=model,
        dataset=dataset_bundle,
        config=config,
        run_dir=run_dir,
        device=get_device(device_name),
    )
    trainer.run()
    return run_dir


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
) -> None:
    if not input.exists():
        raise typer.BadParameter(f"Input does not exist: {input}")
    console.print("Table export formats will be expanded after benchmark summaries are defined.")
