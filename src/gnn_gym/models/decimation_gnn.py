import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("decimation_gnn")
class DecimationGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_rounds: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.update = nn.GRUCell(hidden_channels, hidden_channels)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.norm = norm_layer(norm, hidden_channels)
        self.num_rounds = num_rounds
        self.dropout = dropout
        self.activation = activation
        self.task = task

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("DecimationGNN requires edge_index")
        h = activation_layer(self.activation)(self.input_proj(x))
        clamped = torch.zeros_like(h)
        for _ in range(self.num_rounds):
            confidence = self.confidence(h)
            clamped = confidence * h + (1.0 - confidence) * clamped
            message_source = h + clamped.detach()
            message = self.conv(self.norm(message_source), edge_index)
            message = activation_layer(self.activation)(message)
            message = nn.functional.dropout(message, p=self.dropout, training=self.training)
            h = self.update(message, h)
        return h


def activation_layer(name: str) -> nn.Module:
    return activation(name)
