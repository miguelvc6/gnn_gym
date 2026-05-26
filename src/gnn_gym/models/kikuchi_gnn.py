import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("kikuchi_gnn")
class KikuchiGNN(NodeModel):
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
        self.region_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.node_mlp = nn.Sequential(
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
            raise ValueError("KikuchiGNN requires edge_index")
        src, dst = edge_index
        h = activation_layer(self.activation)(self.input_proj(x))
        for idx in range(self.num_layers):
            node_state = self.norms[idx](h)
            neighbor_mean = scatter(node_state[src], dst, dim=0, dim_size=x.size(0), reduce="mean")
            ego_region = self.region_mlp(torch.cat([node_state, neighbor_mean], dim=-1))
            region_to_node = scatter(ego_region[src], dst, dim=0, dim_size=x.size(0), reduce="mean")
            delta = self.node_mlp(torch.cat([node_state, ego_region, region_to_node], dim=-1))
            delta = activation_layer(self.activation)(delta)
            delta = nn.functional.dropout(delta, p=self.dropout, training=self.training)
            h = h + delta
        return h


def activation_layer(name: str) -> nn.Module:
    return activation(name)
