from torch import nn
from torch.optim import AdamW, Optimizer


def build_optimizer(model: nn.Module, config: dict[str, object]) -> Optimizer:
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    return AdamW(
        model.parameters(),
        lr=float(training.get("lr", 0.01)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
