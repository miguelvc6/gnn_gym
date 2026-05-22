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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: str = "mean",
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        if hidden_channels is None:
            self.net = nn.Linear(in_channels, out_channels)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "add":
            pooled = global_add_pool(x, batch)
        elif self.pooling == "max":
            pooled = global_max_pool(x, batch)
        else:
            pooled = global_mean_pool(x, batch)
        return self.net(pooled)


class GraphRegressionHead(GraphClassificationHead):
    pass


class DotProductLinkPredictionHead(nn.Module):
    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return (src * dst).sum(dim=-1)


class MLPLinkPredictionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or in_channels
        self.net = nn.Sequential(
            nn.Linear(2 * in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([src, dst], dim=-1)).view(-1)
