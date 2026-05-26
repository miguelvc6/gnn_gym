import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.models.nb_belief_gnn import reverse_edge_lookup
from gnn_gym.registry import register_model


@register_model("cavity_gnn")
class CavityGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_steps: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        residual_weight: float = 0.5,
        **_: object,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError("residual_weight must be in [0, 1]")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.message_init = nn.Linear(hidden_channels, hidden_channels)
        self.message_input = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.message_gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = norm_layer(norm, hidden_channels)
        self.num_steps = num_steps
        self.dropout = dropout
        self.activation = activation
        self.residual_weight = residual_weight
        self.task = task

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("CavityGNN requires edge_index")
        src, dst = edge_index
        reverse = reverse_edge_lookup(edge_index)
        has_reverse = reverse >= 0

        h = activation_layer(self.activation)(self.input_proj(x))
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        messages = self.message_init(h[src])

        for _ in range(self.num_steps):
            incoming = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
            cavity = incoming[src].clone()
            cavity[has_reverse] = cavity[has_reverse] - messages[reverse[has_reverse]]
            update_input = self.message_input(torch.cat([h[src], h[dst], cavity], dim=-1))
            update_input = nn.functional.dropout(
                update_input,
                p=self.dropout,
                training=self.training,
            )
            updated = self.message_gru(update_input, messages)
            messages = self.residual_weight * messages + (1.0 - self.residual_weight) * updated

        incoming = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        out = self.node_update(torch.cat([self.norm(h), incoming], dim=-1))
        out = activation_layer(self.activation)(out)
        return nn.functional.dropout(out, p=self.dropout, training=self.training)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
