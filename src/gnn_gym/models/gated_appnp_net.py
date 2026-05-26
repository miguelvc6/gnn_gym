import torch
from torch import nn
from torch_geometric.nn import APPNP

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("gated_appnp_net")
class GatedAPPNPNet(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 2,
        propagation_steps: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        gate_hidden_channels: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Linear(current, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
            current = hidden_channels
        gate_width = gate_hidden_channels or hidden_channels
        self.propagation = APPNP(K=propagation_steps, alpha=alpha)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_channels, gate_width),
            activation_layer(activation),
            nn.Linear(gate_width, hidden_channels),
            nn.Sigmoid(),
        )
        self.output_mix = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation_layer(activation),
        )
        self.activation = activation
        self.dropout = dropout
        self.task = task

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("GatedAPPNPNet requires edge_index")
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[idx](x)
            x = activation_layer(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        propagated = self.propagation(x, edge_index)
        gate = self.gate(torch.cat([x, propagated], dim=-1))
        mixed = gate * propagated + (1.0 - gate) * x
        return self.output_mix(mixed)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
