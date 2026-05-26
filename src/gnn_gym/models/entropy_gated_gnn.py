import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


class EntropyGatedLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float,
        activation_name: str,
        norm: str | None,
    ) -> None:
        super().__init__()
        self.norm = norm_layer(norm, channels)
        self.uncertainty = nn.Sequential(
            nn.Linear(channels, channels),
            activation(activation_name),
            nn.Linear(channels, 1),
        )
        self.message = nn.Linear(channels, channels)
        self.update = nn.Sequential(
            nn.Linear(2 * channels, channels),
            activation(activation_name),
            nn.Linear(channels, channels),
        )
        self.activation_name = activation_name
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        src, dst = edge_index
        uncertainty = torch.sigmoid(self.uncertainty(h))
        weights = (1.0 - uncertainty[src]) * uncertainty[dst]
        messages = weights * self.message(h[src])
        aggregated = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        delta = self.update(torch.cat([h, aggregated], dim=-1))
        delta = activation(self.activation_name)(delta)
        delta = nn.functional.dropout(delta, p=self.dropout, training=self.training)
        return x + delta


@register_model("entropy_gated_gnn")
class EntropyGatedGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [
                EntropyGatedLayer(
                    channels=hidden_channels,
                    dropout=dropout,
                    activation_name=activation,
                    norm=norm,
                )
                for _ in range(num_layers)
            ]
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
            raise ValueError("EntropyGatedGNN requires edge_index")
        x = self.input_proj(x)
        x = activation(self.activation)(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
