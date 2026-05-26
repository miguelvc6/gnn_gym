import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("survey_gnn")
class SurveyGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_particles: int = 4,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_particles < 1:
            raise ValueError("num_particles must be >= 1")
        self.output_channels = hidden_channels
        self.particle_proj = nn.ModuleList(
            [nn.Linear(in_channels, hidden_channels) for _ in range(num_particles)]
        )
        self.convs = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([norm_layer(norm, hidden_channels) for _ in range(num_layers)])
        self.exchange = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_channels, hidden_channels),
                    activation_layer(activation),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                for _ in range(num_layers)
            ]
        )
        self.particle_score = nn.Linear(hidden_channels, 1)
        self.num_particles = num_particles
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
            raise ValueError("SurveyGNN requires edge_index")
        particles = torch.stack(
            [activation_layer(self.activation)(proj(x)) for proj in self.particle_proj],
            dim=1,
        )
        for idx, conv in enumerate(self.convs):
            updated_particles = []
            consensus = particles.mean(dim=1)
            for particle_idx in range(self.num_particles):
                particle = particles[:, particle_idx, :]
                update = conv(particle, edge_index)
                update = self.norms[idx](update)
                update = activation_layer(self.activation)(update)
                exchanged = self.exchange[idx](torch.cat([update, consensus], dim=-1))
                exchanged = nn.functional.dropout(
                    exchanged,
                    p=self.dropout,
                    training=self.training,
                )
                updated_particles.append(particle + exchanged)
            particles = torch.stack(updated_particles, dim=1)
        scores = self.particle_score(particles).softmax(dim=1)
        return (scores * particles).sum(dim=1)


def activation_layer(name: str) -> nn.Module:
    return activation(name)
