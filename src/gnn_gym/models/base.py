import inspect
from typing import Any

import torch
from torch import nn

from gnn_gym.models.heads import (
    DotProductLinkPredictionHead,
    GraphClassificationHead,
    GraphRegressionHead,
    MLPLinkPredictionHead,
    NodeClassificationHead,
)
from gnn_gym.registry import MODEL_REGISTRY, ensure_registrations


class NodeEncoder(nn.Module):
    output_channels: int

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


NodeModel = NodeEncoder


class TaskModel(nn.Module):
    def __init__(
        self,
        encoder: NodeEncoder,
        out_channels: int,
        task: str,
        pooling: str = "mean",
        link_decoder: str = "dot",
        head_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.task = task
        self.out_channels = out_channels
        in_channels = encoder.output_channels
        if task == "node_classification":
            self.head: nn.Module = NodeClassificationHead(in_channels, out_channels)
        elif task in {"graph_binary_classification", "graph_multilabel_classification"}:
            self.head = GraphClassificationHead(
                in_channels,
                out_channels,
                pooling=pooling,
                hidden_channels=head_hidden_channels,
            )
        elif task == "graph_regression":
            self.head = GraphRegressionHead(
                in_channels,
                out_channels,
                pooling=pooling,
                hidden_channels=head_hidden_channels,
            )
        elif task == "link_prediction":
            self.head = (
                MLPLinkPredictionHead(in_channels, head_hidden_channels)
                if link_decoder == "mlp"
                else DotProductLinkPredictionHead()
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return call_encoder(self.encoder, x, edge_index, batch, edge_attr)

    def decode_links(self, z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        if edge.shape[0] != 2:
            edge = edge.t()
        return self.head(z[edge[0]], z[edge[1]])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index, batch, edge_attr)
        if self.task == "link_prediction":
            return z
        if self.task.startswith("graph_"):
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            return self.head(z, batch)
        return self.head(z)


def call_encoder(
    encoder: NodeEncoder,
    x: torch.Tensor,
    edge_index: torch.Tensor | None,
    batch: torch.Tensor | None,
    edge_attr: torch.Tensor | None,
) -> torch.Tensor:
    if encoder_accepts_edge_attr(encoder):
        return encoder(x, edge_index, batch, edge_attr=edge_attr)
    return encoder(x, edge_index, batch)


def encoder_accepts_edge_attr(encoder: NodeEncoder) -> bool:
    signature = inspect.signature(encoder.forward)
    return "edge_attr" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


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
    encoder = MODEL_REGISTRY[name](
        in_channels=in_channels,
        out_channels=out_channels,
        task=task,
        **model_config,
    )
    return TaskModel(
        encoder=encoder,
        out_channels=out_channels,
        task=task,
        pooling=str(model_config.get("pooling", "mean")),
        link_decoder=str(model_config.get("link_decoder", "dot")),
        head_hidden_channels=model_config.get("head_hidden_channels"),
    )
