import torch
from torch import nn


def cross_entropy_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, y)
