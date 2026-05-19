import torch
from torch import nn
from torch_geometric.utils import negative_sampling

from gnn_gym.evaluation.metrics import hits_at_k
from gnn_gym.evaluation.ogb_eval import OgbEvaluatorAdapter
from gnn_gym.registry import register_trainer
from gnn_gym.training.optimizers import build_optimizer
from gnn_gym.training.trainer import BaseTrainer


@register_trainer("link_prediction")
class LinkPredictionTrainer(BaseTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if self.dataset.data is None or self.dataset.split_idx is None:
            raise ValueError("LinkPredictionTrainer requires graph data and edge splits")
        self.data = self.dataset.data.to(self.device)
        self.split_edge = self.dataset.split_idx
        self.optimizer = build_optimizer(self.model, self.config)
        self.criterion = nn.BCEWithLogitsLoss()
        self.ogb_evaluator = (
            OgbEvaluatorAdapter(self.dataset.name) if self.dataset.evaluator == "ogb" else None
        )

    def _encode(self) -> torch.Tensor:
        return self.model(self.data.x.float(), self.data.edge_index)

    @staticmethod
    def _decode(z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        if edge.shape[0] != 2:
            edge = edge.t()
        return (z[edge[0]] * z[edge[1]]).sum(dim=-1)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        z = self._encode()
        pos_edge = self.split_edge["train"]["edge"].to(self.device)
        neg_edge = negative_sampling(
            edge_index=self.data.edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge.shape[0],
            method="sparse",
        )
        pos_pred = self._decode(z, pos_edge)
        neg_pred = self._decode(z, neg_edge)
        pred = torch.cat([pos_pred, neg_pred])
        target = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        loss = self.criterion(pred, target)
        loss.backward()
        grad_clip = self.config["training"].get("grad_clip_norm")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def _split_score(self, z: torch.Tensor, split: str) -> float:
        pos_edge = self.split_edge[split]["edge"].to(self.device)
        neg_key = "edge_neg" if "edge_neg" in self.split_edge[split] else "edge_neg"
        neg_edge = self.split_edge[split].get(neg_key)
        if neg_edge is None:
            neg_edge = negative_sampling(
                edge_index=self.data.edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=pos_edge.shape[0],
                method="sparse",
            )
        neg_edge = neg_edge.to(self.device)
        pos_pred = self._decode(z, pos_edge).detach().cpu()
        neg_pred = self._decode(z, neg_edge).detach().cpu()
        if self.ogb_evaluator is not None:
            result = self.ogb_evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})
            return float(result.get("hits@50", next(iter(result.values()))))
        return hits_at_k(pos_pred, neg_pred, 50)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        z = self._encode()
        return {
            "train_metric": self._split_score(z, "train"),
            "val_metric": self._split_score(z, "valid"),
            "test_metric": self._split_score(z, "test"),
        }
