import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("rign_gnn")
class RIGNGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        macro_steps: int = 3,
        micro_steps: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "layernorm",
        **_: object,
    ) -> None:
        super().__init__()
        if macro_steps < 1:
            raise ValueError("macro_steps must be >= 1")
        if micro_steps < 1:
            raise ValueError("micro_steps must be >= 1")
        self.output_channels = hidden_channels
        self.y_init = nn.Linear(in_channels, hidden_channels)
        self.z_init = nn.Linear(in_channels, hidden_channels)
        self.x_proj = nn.Linear(in_channels, hidden_channels)
        self.latent_conv = GCNConv(hidden_channels, hidden_channels)
        self.z_input = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.z_update = nn.GRUCell(hidden_channels, hidden_channels)
        self.delta_y = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.gate_y = nn.Linear(3 * hidden_channels, hidden_channels)
        self.norm_y = norm_layer(norm, hidden_channels)
        self.norm_z = norm_layer(norm, hidden_channels)
        self.macro_steps = macro_steps
        self.micro_steps = micro_steps
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
            raise ValueError("RIGNGNN requires edge_index")
        context = activation_layer(self.activation)(self.x_proj(x))
        y = activation_layer(self.activation)(self.y_init(x))
        z = activation_layer(self.activation)(self.z_init(x))
        for _ in range(self.macro_steps):
            for _ in range(self.micro_steps):
                msg = self.latent_conv(z, edge_index)
                msg = activation_layer(self.activation)(msg)
                z_in = self.z_input(torch.cat([self.norm_z(z), self.norm_y(y), msg], dim=-1))
                z = self.z_update(z_in, z)
                z = nn.functional.dropout(z, p=self.dropout, training=self.training)
            yz = torch.cat([self.norm_y(y), self.norm_z(z), context], dim=-1)
            delta = self.delta_y(yz)
            gate = torch.sigmoid(self.gate_y(yz))
            y = y + gate * delta
            y = nn.functional.dropout(y, p=self.dropout, training=self.training)
        return self.norm_y(y)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
