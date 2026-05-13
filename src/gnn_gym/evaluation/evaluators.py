import torch

from gnn_gym.evaluation.metrics import accuracy
from gnn_gym.registry import register_evaluator


@register_evaluator("accuracy")
class AccuracyEvaluator:
    higher_is_better = True
    metric_name = "accuracy"

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        return accuracy(logits, y)
