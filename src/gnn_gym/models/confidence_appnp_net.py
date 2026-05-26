import torch
from torch import nn
from torch_geometric.nn import APPNP

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("confidence_appnp_net")
class ConfidenceAPPNPNet(NodeModel):
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
        self.propagation = APPNP(K=propagation_steps, alpha=alpha)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation_layer(activation),
        )
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
            raise ValueError("ConfidenceAPPNPNet requires edge_index")
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[idx](x)
            x = activation_layer(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        propagated = self.propagation(x, edge_index)
        keep_feature = self.confidence(x)
        mixed = keep_feature * x + (1.0 - keep_feature) * propagated
        return self.output_proj(mixed)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
