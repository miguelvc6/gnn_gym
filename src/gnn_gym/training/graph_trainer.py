from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from gnn_gym.evaluation.metrics import average_precision, binary_roc_auc, mean_absolute_error
from gnn_gym.evaluation.ogb_eval import OgbEvaluatorAdapter
from gnn_gym.registry import register_trainer
from gnn_gym.training.optimizers import build_optimizer
from gnn_gym.training.schedulers import build_scheduler
from gnn_gym.training.trainer import BaseTrainer


@register_trainer("graph_prediction")
class GraphPredictionTrainer(BaseTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        trainer_config = self.config.get("trainer", {})
        batch_size = int(
            trainer_config.get("batch_size", self.config["training"].get("batch_size", 128))
        )
        self.optimizer = build_optimizer(self.model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.criterion = self._build_loss()
        self.loaders = self._build_loaders(batch_size)
        self.ogb_evaluator = (
            OgbEvaluatorAdapter(self.dataset.name) if self.dataset.evaluator == "ogb" else None
        )

    def _build_loaders(self, batch_size: int) -> dict[str, DataLoader]:
        if isinstance(self.dataset.dataset, dict):
            return {
                split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
                for split, ds in self.dataset.dataset.items()
            }
        if self.dataset.dataset is None or self.dataset.split_idx is None:
            raise ValueError("GraphPredictionTrainer requires a dataset and split indices")
        train_dataset = self._select_split("train")
        val_dataset = self._select_split("valid")
        test_dataset = self._select_split("test")
        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        }

    def _select_split(self, split: str) -> Any:
        idx = self.dataset.split_idx[split]
        if split == "valid" and idx is None:
            idx = self.dataset.split_idx["val"]
        if isinstance(self.dataset.dataset, list):
            return [self.dataset.dataset[int(i)] for i in idx]
        return self.dataset.dataset[idx]

    def _build_loss(self) -> nn.Module:
        if self.dataset.task == "graph_regression":
            return nn.L1Loss()
        if self.dataset.task in {"graph_binary_classification", "graph_multilabel_classification"}:
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _forward_batch(self, batch: Any) -> torch.Tensor:
        x = batch.x.float()
        return self.model(
            x,
            batch.edge_index,
            batch=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
        )

    def _loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.dataset.task == "graph_regression":
            return self.criterion(pred, y.float())
        if self.dataset.task in {"graph_binary_classification", "graph_multilabel_classification"}:
            y = y.float()
            mask = ~torch.isnan(y)
            return self.criterion(pred[mask], y[mask])
        return self.criterion(pred, y.view(-1).long())

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in self.loaders["train"]:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self._forward_batch(batch)
            loss = self._loss(pred, batch.y)
            loss.backward()
            grad_clip = self.config["training"].get("grad_clip_norm")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
            self.optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
            total_graphs += int(batch.num_graphs)
        return total_loss / max(total_graphs, 1)

    @torch.no_grad()
    def _predict(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        preds: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for batch in self.loaders[split]:
            batch = batch.to(self.device)
            preds.append(self._forward_batch(batch).detach().cpu())
            ys.append(batch.y.detach().cpu())
        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

    def _score(self, pred: torch.Tensor, y: torch.Tensor) -> float:
        if self.ogb_evaluator is not None:
            result = self.ogb_evaluator.eval({"y_pred": pred, "y_true": y})
            return float(next(iter(result.values())))
        if self.dataset.metric == "rocauc":
            return binary_roc_auc(pred, y)
        if self.dataset.metric in {"average_precision", "ap"}:
            return average_precision(pred, y)
        if self.dataset.metric == "mae":
            return mean_absolute_error(pred, y)
        raise ValueError(f"Unsupported graph metric: {self.dataset.metric}")

    def evaluate(self) -> dict[str, float]:
        train_pred, train_y = self._predict("train")
        val_pred, val_y = self._predict("val")
        test_pred, test_y = self._predict("test")
        return {
            "train_metric": self._score(train_pred, train_y),
            "val_metric": self._score(val_pred, val_y),
            "test_metric": self._score(test_pred, test_y),
        }

    @torch.no_grad()
    def predict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            split: {"pred": pred, "y": y}
            for split in ("train", "val", "test")
            for pred, y in [self._predict(split)]
        }
