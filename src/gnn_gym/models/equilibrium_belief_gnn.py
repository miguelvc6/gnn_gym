import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("equilibrium_belief_gnn")
class EquilibriumBeliefGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        max_steps: int = 8,
        tolerance: float = 1e-3,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.update = nn.GRUCell(hidden_channels, hidden_channels)
        self.norm = norm_layer(norm, hidden_channels)
        self.max_steps = max_steps
        self.tolerance = tolerance
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
            raise ValueError("EquilibriumBeliefGNN requires edge_index")
        h = activation_layer(self.activation)(self.input_proj(x))
        for _ in range(self.max_steps):
            previous = h
            message = self.conv(self.norm(h), edge_index)
            message = activation_layer(self.activation)(message)
            message = nn.functional.dropout(message, p=self.dropout, training=self.training)
            h = self.update(message, h)
            if not self.training:
                delta = (h - previous).norm() / h.numel()
                if float(delta.detach().cpu()) < self.tolerance:
                    break
        return h


def activation_layer(name: str) -> nn.Module:
    return activation(name)
