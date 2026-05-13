from __future__ import annotations

import json
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from gnn_gym.data.adapters import DatasetBundle
from gnn_gym.utils.checkpointing import save_checkpoint
from gnn_gym.utils.logging import JsonlLogger


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataset: DatasetBundle,
        config: dict[str, Any],
        run_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.dataset = dataset
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = device
        self.logger = JsonlLogger(self.run_dir / "metrics.jsonl")
        self.best_epoch = 0
        self.best_val_metric = float("-inf")
        self.best_test_metric = float("-inf")

    @abstractmethod
    def train_epoch(self, epoch: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError

    def save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        save_checkpoint(
            self.run_dir / name,
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
        )

    def write_config_files(self) -> None:
        for name in ("config.yaml", "resolved_config.yaml"):
            with (self.run_dir / name).open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.config, handle, sort_keys=True)

    def write_metadata(self, final_metrics: dict[str, Any], status: str) -> None:
        metadata = {
            "run_id": self.run_dir.name,
            "experiment_name": self.config.get("experiment", {}).get("name", "manual"),
            "model": self.config["model"]["name"],
            "dataset": self.dataset.name,
            "task": self.dataset.task,
            "seed": self.config["training"]["seed"],
            "git_commit": git_commit(),
            "config_hash": self.config.get("config_hash"),
            "device": str(self.device),
            "best_epoch": self.best_epoch,
            "status": status,
        }
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        with (self.run_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(final_metrics, handle, indent=2, sort_keys=True)

    def run(self) -> dict[str, Any]:
        self.write_config_files()
        start = time.perf_counter()
        training = self.config["training"]
        max_epochs = int(training.get("max_epochs", 100))
        patience = int(training.get("patience", max_epochs))

        last_metrics: dict[str, float] = {}
        for epoch in range(1, max_epochs + 1):
            loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            last_metrics = metrics
            val_metric = metrics["val_metric"]
            improved = val_metric > self.best_val_metric
            if improved:
                self.best_val_metric = val_metric
                self.best_test_metric = metrics["test_metric"]
                self.best_epoch = epoch
                self.save_checkpoint("checkpoint_best.pt", epoch, metrics)
            self.logger.log({"epoch": epoch, "train_loss": loss, **metrics})
            if epoch - self.best_epoch >= patience:
                break

        self.save_checkpoint("checkpoint_last.pt", epoch, last_metrics)
        final_metrics = {
            "metric_name": self.dataset.metric,
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "test_metric": self.best_test_metric,
            "last_val_metric": last_metrics.get("val_metric"),
            "last_test_metric": last_metrics.get("test_metric"),
            "train_time_seconds": time.perf_counter() - start,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }
        self.write_metadata(final_metrics, "completed")
        return final_metrics


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()
