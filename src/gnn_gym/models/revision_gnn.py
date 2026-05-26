import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("revision_gnn")
class RevisionGNN(NodeModel):
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
        self.feature_belief = nn.Linear(in_channels, hidden_channels)
        self.feature_context = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.delta_mlps = nn.ModuleList()
        self.gate_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
            self.delta_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_channels * 3, hidden_channels),
                    activation_layer(activation),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            )
            self.gate_mlps.append(nn.Linear(hidden_channels * 3, hidden_channels))
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
            raise ValueError("RevisionGNN requires edge_index")
        belief = activation(self.activation)(self.feature_belief(x))
        context = activation(self.activation)(self.feature_context(x))
        for idx, conv in enumerate(self.convs):
            evidence = conv(belief, edge_index)
            evidence = self.norms[idx](evidence)
            evidence = activation(self.activation)(evidence)
            evidence = nn.functional.dropout(evidence, p=self.dropout, training=self.training)
            joint = torch.cat([belief, evidence, context], dim=-1)
            delta = self.delta_mlps[idx](joint)
            gate = torch.sigmoid(self.gate_mlps[idx](joint))
            delta = nn.functional.dropout(delta, p=self.dropout, training=self.training)
            belief = belief + gate * delta
        return belief


def activation_layer(name: str) -> nn.Module:
    return activation(name)
