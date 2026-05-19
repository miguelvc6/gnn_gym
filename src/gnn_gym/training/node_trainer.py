import torch
from torch_geometric.utils import k_hop_subgraph

from gnn_gym.evaluation.metrics import accuracy
from gnn_gym.evaluation.ogb_eval import OgbEvaluatorAdapter
from gnn_gym.registry import register_trainer
from gnn_gym.training.losses import cross_entropy_loss
from gnn_gym.training.optimizers import build_optimizer
from gnn_gym.training.trainer import BaseTrainer


@register_trainer("full_batch_node")
class FullBatchNodeTrainer(BaseTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.data = self.dataset.data.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)
        self.ogb_evaluator = (
            OgbEvaluatorAdapter(self.dataset.name) if self.dataset.evaluator == "ogb" else None
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.data.x, self.data.edge_index)
        loss = cross_entropy_loss(logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        grad_clip = self.config["training"].get("grad_clip_norm")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        if self.ogb_evaluator is not None:
            y_pred = logits.argmax(dim=-1, keepdim=True)
            y_true = self.data.y.view(-1, 1)
            return {
                "train_metric": self.ogb_evaluator.eval(
                    {
                        "y_true": y_true[self.data.train_mask],
                        "y_pred": y_pred[self.data.train_mask],
                    }
                )["acc"],
                "val_metric": self.ogb_evaluator.eval(
                    {"y_true": y_true[self.data.val_mask], "y_pred": y_pred[self.data.val_mask]}
                )["acc"],
                "test_metric": self.ogb_evaluator.eval(
                    {"y_true": y_true[self.data.test_mask], "y_pred": y_pred[self.data.test_mask]}
                )["acc"],
            }
        return {
            "train_metric": accuracy(
                logits[self.data.train_mask],
                self.data.y[self.data.train_mask],
            ),
            "val_metric": accuracy(logits[self.data.val_mask], self.data.y[self.data.val_mask]),
            "test_metric": accuracy(logits[self.data.test_mask], self.data.y[self.data.test_mask]),
        }


@register_trainer("neighbor_node")
class NeighborNodeTrainer(BaseTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.data = self.dataset.data
        if self.data is None:
            raise ValueError("NeighborNodeTrainer requires a single graph Data object")
        self.data = self.data.to(self.device)
        trainer_config = self.config.get("trainer", {})
        self.batch_size = int(trainer_config.get("batch_size", 1024))
        self.num_hops = len(list(trainer_config.get("num_neighbors", [15, 10, 5])))
        self.train_nodes = self.data.train_mask.nonzero(as_tuple=False).view(-1)
        self.optimizer = build_optimizer(self.model, self.config)
        self.ogb_evaluator = (
            OgbEvaluatorAdapter(self.dataset.name) if self.dataset.evaluator == "ogb" else None
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        permutation = self.train_nodes[torch.randperm(self.train_nodes.numel(), device=self.device)]
        for start in range(0, permutation.numel(), self.batch_size):
            batch_nodes = permutation[start : start + self.batch_size]
            subset, edge_index, mapping, _ = k_hop_subgraph(
                batch_nodes,
                num_hops=self.num_hops,
                edge_index=self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes,
            )
            self.optimizer.zero_grad()
            logits = self.model(self.data.x[subset], edge_index)
            loss = cross_entropy_loss(logits[mapping], self.data.y[batch_nodes].view(-1))
            loss.backward()
            grad_clip = self.config["training"].get("grad_clip_norm")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
            self.optimizer.step()
            total_loss += float(loss.item()) * batch_nodes.numel()
            total_examples += int(batch_nodes.numel())
        return total_loss / max(total_examples, 1)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        if self.ogb_evaluator is not None:
            y_pred = logits.argmax(dim=-1, keepdim=True)
            y_true = self.data.y.view(-1, 1)
            return {
                "train_metric": self.ogb_evaluator.eval(
                    {
                        "y_true": y_true[self.data.train_mask],
                        "y_pred": y_pred[self.data.train_mask],
                    }
                )["acc"],
                "val_metric": self.ogb_evaluator.eval(
                    {"y_true": y_true[self.data.val_mask], "y_pred": y_pred[self.data.val_mask]}
                )["acc"],
                "test_metric": self.ogb_evaluator.eval(
                    {"y_true": y_true[self.data.test_mask], "y_pred": y_pred[self.data.test_mask]}
                )["acc"],
            }
        return {
            "train_metric": accuracy(
                logits[self.data.train_mask],
                self.data.y[self.data.train_mask],
            ),
            "val_metric": accuracy(logits[self.data.val_mask], self.data.y[self.data.val_mask]),
            "test_metric": accuracy(logits[self.data.test_mask], self.data.y[self.data.test_mask]),
        }
