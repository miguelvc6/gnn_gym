import torch
from torch import nn
from torch_geometric.nn import GATConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("gat")
class GAT(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 32,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        attention_dropout: float = 0.2,
        activation: str = "elu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, dropout=attention_dropout)
            )
        else:
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, dropout=attention_dropout)
            )
            current = hidden_channels * heads
            self.norms.append(norm_layer(norm, current))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(current, hidden_channels, heads=heads, dropout=attention_dropout)
                )
                current = hidden_channels * heads
                self.norms.append(norm_layer(norm, current))
            self.convs.append(GATConv(current, out_channels, heads=1, concat=False))
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
            raise ValueError("GAT requires edge_index")
        for idx, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[idx](x)
            x = activation(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
