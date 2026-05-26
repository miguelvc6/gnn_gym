import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("loop_corrected_gnn")
class LoopCorrectedGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 2,
        max_triangles: int = 512,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(norm_layer(norm, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
        self.loop_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.max_triangles = max_triangles
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
            raise ValueError("LoopCorrectedGNN requires edge_index")
        triangles = find_triangles(edge_index, x.size(0), self.max_triangles)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[idx](x)
            x = activation_layer(self.activation)(x)
            if triangles.numel() > 0:
                triangles = triangles.to(x.device)
                tri_state = x[triangles].mean(dim=1)
                tri_index = triangles.reshape(-1)
                tri_messages = tri_state.repeat_interleave(3, dim=0)
                loop_context = scatter(
                    tri_messages,
                    tri_index,
                    dim=0,
                    dim_size=x.size(0),
                    reduce="mean",
                )
                x = x + self.loop_update(torch.cat([x, loop_context], dim=-1))
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


def find_triangles(edge_index: torch.Tensor, num_nodes: int, max_triangles: int) -> torch.Tensor:
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for left, right in zip(src, dst, strict=True):
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    triangles: list[tuple[int, int, int]] = []
    for left in range(num_nodes):
        for middle in adjacency[left]:
            if middle <= left:
                continue
            common = adjacency[left].intersection(adjacency[middle])
            for right in common:
                if right <= middle:
                    continue
                triangles.append((left, middle, right))
                if len(triangles) >= max_triangles:
                    return torch.tensor(triangles, dtype=torch.long)
    if not triangles:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.tensor(triangles, dtype=torch.long)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
