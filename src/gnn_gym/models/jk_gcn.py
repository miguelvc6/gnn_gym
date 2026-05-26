import torch
from torch import nn
from torch_geometric.nn import GCNConv, JumpingKnowledge

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("jk_gcn")
class JKGCN(NodeModel):
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
        jk_mode: str = "cat",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if jk_mode not in {"cat", "max", "lstm"}:
            raise ValueError("jk_mode must be one of: cat, max, lstm")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            self.convs.append(GCNConv(current, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
            current = hidden_channels
        self.jk = JumpingKnowledge(jk_mode, channels=hidden_channels, num_layers=num_layers)
        self.output_channels = hidden_channels * num_layers if jk_mode == "cat" else hidden_channels
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
            raise ValueError("JKGCN requires edge_index")
        states = []
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[idx](x)
            x = activation(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            states.append(x)
        return self.jk(states)
