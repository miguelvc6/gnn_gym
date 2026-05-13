from typing import Any

import torch
from torch import nn

from gnn_gym.registry import MODEL_REGISTRY, ensure_registrations


class NodeModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def activation(name: str) -> nn.Module:
    match name:
        case "elu":
            return nn.ELU()
        case "gelu":
            return nn.GELU()
        case "relu":
            return nn.ReLU()
        case _:
            raise ValueError(f"Unsupported activation: {name}")


def norm_layer(name: str | None, channels: int) -> nn.Module:
    if name in (None, "none"):
        return nn.Identity()
    if name == "batchnorm":
        return nn.BatchNorm1d(channels)
    if name == "layernorm":
        return nn.LayerNorm(channels)
    raise ValueError(f"Unsupported norm: {name}")


def build_model(
    name: str,
    in_channels: int,
    out_channels: int,
    task: str,
    config: dict[str, Any],
) -> nn.Module:
    ensure_registrations()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    model_config = config.get("model", config)
    return MODEL_REGISTRY[name](
        in_channels=in_channels,
        out_channels=out_channels,
        task=task,
        **model_config,
    )
