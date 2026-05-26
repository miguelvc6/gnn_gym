import torch
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("gpr_gnn")
class GPRGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 2,
        propagation_steps: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if propagation_steps < 1:
            raise ValueError("propagation_steps must be >= 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.output_channels = hidden_channels
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Linear(current, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
            current = hidden_channels
        coeffs = [alpha * (1.0 - alpha) ** step for step in range(propagation_steps)]
        coeffs.append((1.0 - alpha) ** propagation_steps)
        self.hop_logits = nn.Parameter(torch.tensor(coeffs, dtype=torch.float))
        self.propagation_steps = propagation_steps
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
            raise ValueError("GPRGNN requires edge_index")
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[idx](x)
            x = activation_layer(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=x.size(0),
            add_self_loops=True,
        )
        states = [x]
        propagated = x
        for _ in range(self.propagation_steps):
            propagated = propagate_normalized(propagated, edge_index, edge_weight)
            states.append(propagated)
        weights = self.hop_logits.view(-1, 1, 1)
        stacked = torch.stack(states, dim=0)
        return (weights * stacked).sum(dim=0)


def propagate_normalized(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    messages = edge_weight.view(-1, 1) * x[src]
    return scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")


def activation_layer(name: str) -> nn.Module:
    return activation(name)
