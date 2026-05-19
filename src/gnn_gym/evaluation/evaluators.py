import torch

from gnn_gym.evaluation.metrics import (
    accuracy,
    average_precision,
    binary_roc_auc,
    mean_absolute_error,
)
from gnn_gym.registry import register_evaluator


@register_evaluator("accuracy")
class AccuracyEvaluator:
    higher_is_better = True
    metric_name = "accuracy"

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        return accuracy(logits, y)


@register_evaluator("rocauc")
class RocAucEvaluator:
    higher_is_better = True
    metric_name = "rocauc"

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        return binary_roc_auc(logits, y)


@register_evaluator("average_precision")
class AveragePrecisionEvaluator:
    higher_is_better = True
    metric_name = "average_precision"

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        return average_precision(logits, y)


@register_evaluator("mae")
class MeanAbsoluteErrorEvaluator:
    higher_is_better = False
    metric_name = "mae"

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        return mean_absolute_error(logits, y)
