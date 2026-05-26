import torch
from torch import nn
from torch_geometric.nn import APPNP

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("appnp_net")
class APPNPNet(NodeModel):
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
        propagation_steps: int = 10,
        alpha: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Linear(current, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
            current = hidden_channels
        self.propagation = APPNP(K=propagation_steps, alpha=alpha)
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
            raise ValueError("APPNPNet requires edge_index")
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[idx](x)
            x = activation(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.propagation(x, edge_index)
