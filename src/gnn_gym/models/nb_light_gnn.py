import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation
from gnn_gym.registry import register_model


def vectorized_reverse_edge_lookup(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    keys = src * num_nodes + dst
    reverse_keys = dst * num_nodes + src
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    positions = torch.searchsorted(sorted_keys, reverse_keys)
    valid = positions < sorted_keys.numel()
    valid_positions = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
    valid = valid & (sorted_keys[valid_positions] == reverse_keys)
    reverse = torch.full_like(src, -1)
    reverse[valid] = order[valid_positions[valid]]
    return reverse


@register_model("nb_light_gnn")
class NonBacktrackingLightGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_steps: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        **_: object,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.message_init = nn.Linear(hidden_channels, hidden_channels)
        self.message_update = nn.Linear(hidden_channels, hidden_channels)
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.num_steps = num_steps
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
            raise ValueError("NonBacktrackingLightGNN requires edge_index")
        src, dst = edge_index
        reverse = vectorized_reverse_edge_lookup(edge_index, x.size(0))
        has_reverse = reverse >= 0
        h = activation_layer(self.activation)(self.input_proj(x))
        messages = self.message_init(h[src])
        for _ in range(self.num_steps):
            incoming = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
            cavity = incoming[src]
            if has_reverse.any():
                cavity = cavity.clone()
                cavity[has_reverse] = cavity[has_reverse] - messages[reverse[has_reverse]]
            updated = self.message_update(cavity)
            updated = activation_layer(self.activation)(updated)
            updated = nn.functional.dropout(updated, p=self.dropout, training=self.training)
            messages = 0.5 * messages + 0.5 * updated
        incoming = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        out = self.node_update(torch.cat([h, incoming], dim=-1))
        return activation_layer(self.activation)(out)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
