import torch
from torch import nn

from gnn_gym.models.base import NodeModel, norm_layer
from gnn_gym.models.base import activation as make_activation
from gnn_gym.registry import register_model


@register_model("mlp")
class MLP(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        layers: list[nn.Module] = []
        current = in_channels
        for _layer in range(max(0, num_layers)):
            layers.extend(
                [
                    nn.Linear(current, hidden_channels),
                    norm_layer(norm, hidden_channels),
                    make_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_channels
        self.net = nn.Sequential(*layers)
        self.task = task

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(x)
