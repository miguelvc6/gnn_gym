import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("walk_belief_transformer")
class WalkBeliefTransformer(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        walk_length: int = 4,
        num_layers: int = 1,
        heads: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if walk_length < 2:
            raise ValueError("walk_length must be >= 2")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=heads,
            dim_feedforward=2 * hidden_channels,
            dropout=dropout,
            batch_first=True,
            activation=activation,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = norm_layer(norm, hidden_channels)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.walk_length = walk_length
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
            raise ValueError("WalkBeliefTransformer requires edge_index")
        h = activation_layer(self.activation)(self.input_proj(x))
        successor = deterministic_successor(edge_index, h.size(0))
        walk_indices = build_walk_indices(successor, self.walk_length)
        tokens = self.norm(h)[walk_indices]
        walk_state = self.transformer(tokens).mean(dim=1)
        out = self.out(torch.cat([h, walk_state], dim=-1))
        return nn.functional.dropout(out, p=self.dropout, training=self.training)


def deterministic_successor(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    successor = scatter(dst, src, dim=0, dim_size=num_nodes, reduce="min")
    has_edge = scatter(torch.ones_like(src), src, dim=0, dim_size=num_nodes, reduce="sum") > 0
    node_ids = torch.arange(num_nodes, device=edge_index.device)
    successor = torch.where(has_edge, successor, node_ids)
    return successor.clamp(min=0, max=num_nodes - 1)


def build_walk_indices(successor: torch.Tensor, walk_length: int) -> torch.Tensor:
    current = torch.arange(successor.numel(), device=successor.device)
    walks = [current]
    for _ in range(walk_length - 1):
        current = successor[current]
        walks.append(current)
    return torch.stack(walks, dim=1)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
