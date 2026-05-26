import torch
from torch import nn
from torch_geometric.nn import GCN2Conv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("gcn2_net")
class GCN2Net(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 8,
        dropout: float = 0.5,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        alpha: float = 0.1,
        theta: float = 0.5,
        shared_weights: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [
                GCN2Conv(
                    hidden_channels,
                    alpha=alpha,
                    theta=theta,
                    layer=idx + 1,
                    shared_weights=shared_weights,
                )
                for idx in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([norm_layer(norm, hidden_channels) for _ in range(num_layers)])
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
            raise ValueError("GCN2Net requires edge_index")
        x = self.input_proj(x)
        x = activation(self.activation)(x)
        x0 = x
        for idx, conv in enumerate(self.convs):
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x0, edge_index)
            x = self.norms[idx](x)
            x = activation(self.activation)(x)
        return x
