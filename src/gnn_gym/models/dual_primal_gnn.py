import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("dual_primal_gnn")
class DualPrimalGNN(NodeModel):
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
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_proj = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(5 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.node_update = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norms = nn.ModuleList([norm_layer(norm, hidden_channels) for _ in range(num_layers)])
        self.num_layers = num_layers
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
            raise ValueError("DualPrimalGNN requires edge_index")
        src, dst = edge_index
        h = activation_layer(self.activation)(self.node_proj(x))
        edge_state = self.edge_proj(torch.cat([h[src], h[dst]], dim=-1))
        for idx in range(self.num_layers):
            node_state = self.norms[idx](h)
            edge_to_src = scatter(edge_state, src, dim=0, dim_size=x.size(0), reduce="mean")
            edge_to_dst = scatter(edge_state, dst, dim=0, dim_size=x.size(0), reduce="mean")
            edge_delta = self.edge_update(
                torch.cat(
                    [
                        edge_state,
                        node_state[src],
                        node_state[dst],
                        edge_to_src[src],
                        edge_to_dst[dst],
                    ],
                    dim=-1,
                )
            )
            edge_delta = activation_layer(self.activation)(edge_delta)
            edge_delta = nn.functional.dropout(edge_delta, p=self.dropout, training=self.training)
            edge_state = edge_state + edge_delta
            incoming = scatter(edge_state, dst, dim=0, dim_size=x.size(0), reduce="mean")
            outgoing = scatter(edge_state, src, dim=0, dim_size=x.size(0), reduce="mean")
            node_delta = self.node_update(torch.cat([node_state, incoming, outgoing], dim=-1))
            node_delta = activation_layer(self.activation)(node_delta)
            node_delta = nn.functional.dropout(node_delta, p=self.dropout, training=self.training)
            h = h + node_delta
        return h


def activation_layer(name: str) -> nn.Module:
    return activation(name)
