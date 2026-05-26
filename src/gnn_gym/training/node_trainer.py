import torch
from torch_geometric.utils import k_hop_subgraph

from gnn_gym.evaluation.metrics import accuracy
from gnn_gym.evaluation.ogb_eval import OgbEvaluatorAdapter
from gnn_gym.registry import register_trainer
from gnn_gym.training.losses import cross_entropy_loss
from gnn_gym.training.optimizers import build_optimizer
from gnn_gym.training.schedulers import build_scheduler
from gnn_gym.training.trainer import BaseTrainer

try:
    from torch_geometric.loader import NeighborLoader
except ImportError:  # pragma: no cover - depends on optional PyG extras.
    NeighborLoader = None  # type: ignore[assignment]


@register_trainer("full_batch_node")
class FullBatchNodeTrainer(BaseTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.data = self.dataset.data.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.ogb_evaluator = (
            OgbEvaluatorAdapter(self.dataset.name) if self.dataset.evaluator == "ogb" else None
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(
            self.data.x,
            self.data.edge_index,
            edge_attr=getattr(self.data, "edge_attr", None),
        )
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
        logits = self.model(
            self.data.x,
            self.data.edge_index,
            edge_attr=getattr(self.data, "edge_attr", None),
        )
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

    @torch.no_grad()
    def predict(self) -> dict[str, torch.Tensor]:
        self.model.eval()
        logits = self.model(
            self.data.x,
            self.data.edge_index,
            edge_attr=getattr(self.data, "edge_attr", None),
        )
        return {
            "logits": logits.detach().cpu(),
            "y": self.data.y.detach().cpu(),
            "train_mask": self.data.train_mask.detach().cpu(),
            "val_mask": self.data.val_mask.detach().cpu(),
            "test_mask": self.data.test_mask.detach().cpu(),
        }


@register_trainer("neighbor_node")
class NeighborNodeTrainer(BaseTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.data = self.dataset.data
        if self.data is None:
            raise ValueError("NeighborNodeTrainer requires a single graph Data object")
        trainer_config = self.config.get("trainer", {})
        self.batch_size = int(trainer_config.get("batch_size", 1024))
        self.eval_batch_size = int(trainer_config.get("eval_batch_size", 4096))
        self.num_neighbors = [
            int(value) for value in trainer_config.get("num_neighbors", [15, 10, 5])
        ]
        self.num_hops = len(self.num_neighbors)
        self.train_nodes = self.data.train_mask.nonzero(as_tuple=False).view(-1)
        self.use_neighbor_loader = NeighborLoader is not None
        self._fallback_device_data = None
        self.optimizer = build_optimizer(self.model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.ogb_evaluator = (
            OgbEvaluatorAdapter(self.dataset.name) if self.dataset.evaluator == "ogb" else None
        )

    def train_epoch(self, epoch: int) -> float:
        if self.use_neighbor_loader:
            try:
                return self._train_epoch_neighbor_loader()
            except (ImportError, RuntimeError) as error:
                if not self._is_neighbor_loader_backend_error(error):
                    raise
                self.use_neighbor_loader = False
        return self._train_epoch_fallback()

    def _train_epoch_neighbor_loader(self) -> float:
        loader = self._neighbor_loader(self.data.train_mask, self.batch_size, shuffle=True)
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(
                batch.x,
                batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
            )
            root_count = int(batch.batch_size)
            loss = cross_entropy_loss(logits[:root_count], batch.y[:root_count].view(-1))
            loss.backward()
            grad_clip = self.config["training"].get("grad_clip_norm")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
            self.optimizer.step()
            total_loss += float(loss.item()) * root_count
            total_examples += root_count
        return total_loss / max(total_examples, 1)

    def _train_epoch_fallback(self) -> float:
        self.model.train()
        data = self._fallback_data()
        total_loss = 0.0
        total_examples = 0
        train_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
        edge_attr = getattr(data, "edge_attr", None)
        permutation = train_nodes[torch.randperm(train_nodes.numel(), device=self.device)]
        for start in range(0, permutation.numel(), self.batch_size):
            batch_nodes = permutation[start : start + self.batch_size]
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                batch_nodes,
                num_hops=self.num_hops,
                edge_index=data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )
            self.optimizer.zero_grad()
            batch_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
            logits = self.model(data.x[subset], edge_index, edge_attr=batch_edge_attr)
            loss = cross_entropy_loss(logits[mapping], data.y[batch_nodes].view(-1))
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
        if self.use_neighbor_loader:
            try:
                return {
                    "train_metric": self._evaluate_mask(self.data.train_mask),
                    "val_metric": self._evaluate_mask(self.data.val_mask),
                    "test_metric": self._evaluate_mask(self.data.test_mask),
                }
            except (ImportError, RuntimeError) as error:
                if not self._is_neighbor_loader_backend_error(error):
                    raise
                self.use_neighbor_loader = False
        return self._evaluate_full_graph()

    @torch.no_grad()
    def _evaluate_mask(self, mask: torch.Tensor) -> float:
        self.model.eval()
        preds: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        loader = self._neighbor_loader(mask, self.eval_batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(
                batch.x,
                batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
            )
            root_count = int(batch.batch_size)
            preds.append(logits[:root_count].detach().cpu())
            ys.append(batch.y[:root_count].detach().cpu())
        logits = torch.cat(preds, dim=0)
        y = torch.cat(ys, dim=0)
        if self.ogb_evaluator is not None:
            return float(
                self.ogb_evaluator.eval(
                    {"y_true": y.view(-1, 1), "y_pred": logits.argmax(dim=-1, keepdim=True)}
                )["acc"]
            )
        return accuracy(logits, y)

    @torch.no_grad()
    def _evaluate_full_graph(self) -> dict[str, float]:
        self.model.eval()
        data = self._fallback_data()
        logits = self.model(
            data.x,
            data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
        )
        if self.ogb_evaluator is not None:
            y_pred = logits.argmax(dim=-1, keepdim=True)
            y_true = data.y.view(-1, 1)
            return {
                "train_metric": self.ogb_evaluator.eval(
                    {
                        "y_true": y_true[data.train_mask],
                        "y_pred": y_pred[data.train_mask],
                    }
                )["acc"],
                "val_metric": self.ogb_evaluator.eval(
                    {"y_true": y_true[data.val_mask], "y_pred": y_pred[data.val_mask]}
                )["acc"],
                "test_metric": self.ogb_evaluator.eval(
                    {"y_true": y_true[data.test_mask], "y_pred": y_pred[data.test_mask]}
                )["acc"],
            }
        return {
            "train_metric": accuracy(
                logits[data.train_mask],
                data.y[data.train_mask],
            ),
            "val_metric": accuracy(logits[data.val_mask], data.y[data.val_mask]),
            "test_metric": accuracy(logits[data.test_mask], data.y[data.test_mask]),
        }

    @torch.no_grad()
    def predict(self) -> dict[str, torch.Tensor]:
        self.model.eval()
        data = self._fallback_data()
        logits = self.model(
            data.x,
            data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
        )
        return {
            "logits": logits.detach().cpu(),
            "y": data.y.detach().cpu(),
            "train_mask": data.train_mask.detach().cpu(),
            "val_mask": data.val_mask.detach().cpu(),
            "test_mask": data.test_mask.detach().cpu(),
        }

    def _neighbor_loader(
        self,
        input_nodes: torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ) -> object:
        if NeighborLoader is None:
            raise ImportError("torch_geometric.loader.NeighborLoader is unavailable")
        return NeighborLoader(
            self.data,
            input_nodes=input_nodes,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _is_neighbor_loader_backend_error(self, error: BaseException) -> bool:
        message = str(error).lower()
        return "neighborloader" in message or "pyg-lib" in message or "torch-sparse" in message

    def _fallback_data(self) -> object:
        if self._fallback_device_data is None:
            self._fallback_device_data = self.data.to(self.device)
        return self._fallback_device_data
