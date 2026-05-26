import torch
from torch import nn
from torch_geometric.nn import GINConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


class ResidualGINBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float,
        activation_name: str,
        norm: str | None,
        eps_trainable: bool,
    ) -> None:
        super().__init__()
        self.norm = norm_layer(norm, channels)
        self.conv = GINConv(
            nn.Sequential(
                nn.Linear(channels, channels),
                activation(activation_name),
                nn.Linear(channels, channels),
            ),
            train_eps=eps_trainable,
        )
        self.activation_name = activation_name
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        update = self.norm(x)
        update = self.conv(update, edge_index)
        update = activation(self.activation_name)(update)
        update = nn.functional.dropout(update, p=self.dropout, training=self.training)
        return x + update


@register_model("res_gin")
class ResidualGIN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        eps_trainable: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                ResidualGINBlock(
                    channels=hidden_channels,
                    dropout=dropout,
                    activation_name=activation,
                    norm=norm,
                    eps_trainable=eps_trainable,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = norm_layer(norm, hidden_channels)
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
            raise ValueError("ResidualGIN requires edge_index")
        x = self.input_proj(x)
        x = activation(self.activation)(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for block in self.blocks:
            x = block(x, edge_index)
        x = self.final_norm(x)
        return activation(self.activation)(x)
