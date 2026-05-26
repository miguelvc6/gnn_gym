import torch
from torch import nn
from torch_geometric.nn import APPNP
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.models.nb_belief_gnn import reverse_edge_lookup
from gnn_gym.registry import register_model


@register_model("nb_appnp_net")
class NBAPPNPNet(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 2,
        nb_steps: int = 3,
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
        if nb_steps < 1:
            raise ValueError("nb_steps must be >= 1")
        self.output_channels = hidden_channels
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Linear(current, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
            current = hidden_channels

        self.appnp = APPNP(K=propagation_steps, alpha=alpha)
        self.message_init = nn.Linear(hidden_channels, hidden_channels)
        self.message_update = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.nb_readout = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.fusion_out = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            activation_layer(activation),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.nb_steps = nb_steps
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
            raise ValueError("NBAPPNPNet requires edge_index")
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[idx](x)
            x = activation_layer(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        appnp_state = self.appnp(x, edge_index)
        nb_state = self._non_backtracking_state(x, edge_index)
        gate = self.fusion_gate(torch.cat([x, appnp_state, nb_state], dim=-1))
        mixed = gate * appnp_state + (1.0 - gate) * nb_state
        out = self.fusion_out(torch.cat([x, mixed], dim=-1))
        return nn.functional.dropout(out, p=self.dropout, training=self.training)

    def _non_backtracking_state(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        reverse = reverse_edge_lookup(edge_index)
        has_reverse = reverse >= 0
        messages = self.message_init(x[src])
        for _ in range(self.nb_steps):
            incoming = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
            cavity = incoming[src].clone()
            cavity[has_reverse] = cavity[has_reverse] - messages[reverse[has_reverse]]
            updated = self.message_update(torch.cat([x[src], cavity], dim=-1))
            updated = activation_layer(self.activation)(updated)
            updated = nn.functional.dropout(updated, p=self.dropout, training=self.training)
            messages = 0.5 * messages + 0.5 * updated
        incoming = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        return self.nb_readout(torch.cat([x, incoming], dim=-1))


def activation_layer(name: str) -> nn.Module:
    return activation(name)
