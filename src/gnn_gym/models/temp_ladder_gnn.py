import torch
from torch import nn
from torch_geometric.utils import scatter, softmax

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


class TemperatureAttentionStream(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temperature: float) -> None:
        super().__init__()
        self.value = nn.Linear(in_channels, out_channels)
        self.score = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, 1),
        )
        self.temperature = temperature

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h = self.value(x)
        logits = self.score(torch.cat([h[src], h[dst]], dim=-1)).view(-1)
        weights = softmax(logits / self.temperature, dst)
        messages = weights.unsqueeze(-1) * h[src]
        return scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")


class TemperatureLadderLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        temperatures: list[float],
        dropout: float,
        activation_name: str,
        norm: str | None,
    ) -> None:
        super().__init__()
        self.streams = nn.ModuleList(
            [
                TemperatureAttentionStream(in_channels, hidden_channels, temperature)
                for temperature in temperatures
            ]
        )
        self.output_channels = hidden_channels * len(temperatures)
        self.norm = norm_layer(norm, self.output_channels)
        self.activation_name = activation_name
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.cat([stream(x, edge_index) for stream in self.streams], dim=-1)
        x = self.norm(x)
        x = activation(self.activation_name)(x)
        return nn.functional.dropout(x, p=self.dropout, training=self.training)


@register_model("temp_ladder_gnn")
class TemperatureLadderGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 16,
        num_layers: int = 2,
        temperatures: list[float] | None = None,
        dropout: float = 0.2,
        activation: str = "elu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        temperatures = temperatures or [0.5, 1.0, 2.0]
        if not temperatures:
            raise ValueError("temperatures must not be empty")
        self.layers = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            layer = TemperatureLadderLayer(
                in_channels=current,
                hidden_channels=hidden_channels,
                temperatures=temperatures,
                dropout=dropout,
                activation_name=activation,
                norm=norm,
            )
            self.layers.append(layer)
            current = layer.output_channels
        self.output_channels = current
        self.task = task

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("TemperatureLadderGNN requires edge_index")
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
