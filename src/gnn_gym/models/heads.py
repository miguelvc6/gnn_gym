import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class NodeClassificationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GraphClassificationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pooling: str = "mean") -> None:
        super().__init__()
        self.pooling = pooling
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "add":
            pooled = global_add_pool(x, batch)
        elif self.pooling == "max":
            pooled = global_max_pool(x, batch)
        else:
            pooled = global_mean_pool(x, batch)
        return self.linear(pooled)


class DotProductLinkPredictionHead(nn.Module):
    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return (src * dst).sum(dim=-1)
