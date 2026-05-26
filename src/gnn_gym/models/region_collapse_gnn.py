import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("region_collapse_gnn")
class RegionCollapseGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_regions: int = 16,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_regions < 1:
            raise ValueError("num_regions must be >= 1")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([norm_layer(norm, hidden_channels) for _ in range(num_layers)])
        self.assign = nn.Linear(hidden_channels, num_regions)
        self.region_update = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
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
            raise ValueError("RegionCollapseGNN requires edge_index")
        h = activation_layer(self.activation)(self.input_proj(x))
        for idx, conv in enumerate(self.convs):
            local = conv(self.norms[idx](h), edge_index)
            local = activation_layer(self.activation)(local)
            assign = torch.softmax(self.assign(h), dim=-1)
            denom = assign.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
            regions = assign.t() @ h / denom
            regions = self.region_update(regions)
            region_context = assign @ regions
            delta = self.node_update(torch.cat([local, region_context], dim=-1))
            delta = nn.functional.dropout(delta, p=self.dropout, training=self.training)
            h = h + delta
        return h


def activation_layer(name: str) -> nn.Module:
    return activation(name)
