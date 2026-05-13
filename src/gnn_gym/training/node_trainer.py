import torch

from gnn_gym.evaluation.metrics import accuracy
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
        return {
            "train_metric": accuracy(
                logits[self.data.train_mask],
                self.data.y[self.data.train_mask],
            ),
            "val_metric": accuracy(logits[self.data.val_mask], self.data.y[self.data.val_mask]),
            "test_metric": accuracy(logits[self.data.test_mask], self.data.y[self.data.test_mask]),
        }
