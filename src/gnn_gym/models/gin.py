import torch
from torch import nn
from torch_geometric.nn import GINConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("gin")
class GIN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.5,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        eps_trainable: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(current, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=eps_trainable))
            self.norms.append(norm_layer(norm, hidden_channels))
            current = hidden_channels
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
            raise ValueError("GIN requires edge_index")
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[idx](x)
            x = activation(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x
