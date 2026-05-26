from __future__ import annotations

import sys
from math import log

import torch
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("sep_bottleneck_gnn")
class SepBottleneckGNN(NodeModel):
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
        gate_hidden_channels: int = 32,
        separator_residual_init: float = 0.001,
        separator_token_init: float = 0.001,
        separator_max_scale: float = 0.1,
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
        self.separator_token = nn.Parameter(torch.zeros(hidden_channels))
        self.separator_residual_logit = nn.Parameter(
            torch.tensor(
                bounded_scale_logit(separator_residual_init, separator_max_scale),
                dtype=torch.float,
            )
        )
        self.separator_token_logit = nn.Parameter(
            torch.tensor(
                bounded_scale_logit(separator_token_init, separator_max_scale),
                dtype=torch.float,
            )
        )
        rng_state = torch.random.get_rng_state()
        self.separator_gate = nn.Sequential(
            nn.Linear((2 * hidden_channels) + 2, gate_hidden_channels),
            activation_layer(activation),
            nn.Linear(gate_hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.separator_update = nn.Linear(hidden_channels, hidden_channels)
        torch.random.set_rng_state(rng_state)
        self.propagation_steps = propagation_steps
        self.dropout = dropout
        self.activation = activation
        self.task = task
        self.separator_max_scale = float(separator_max_scale)
        self._structure_cache: dict[tuple[int, tuple[int, ...]], StructuralMarkers] = {}

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("SepBottleneckGNN requires edge_index")
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
        markers = self._structural_markers(edge_index, x.size(0), x.device)
        token_scale = self.separator_max_scale * torch.sigmoid(self.separator_token_logit)
        x = x + token_scale * markers.articulation_nodes.view(-1, 1) * self.separator_token.view(
            1, -1
        )

        states = [x]
        propagated = x
        correction = torch.zeros_like(x)
        for _ in range(self.propagation_steps):
            propagated = propagate_normalized(propagated, edge_index, edge_weight)
            correction = correction + separator_messages(
                propagated,
                edge_index,
                edge_weight,
                markers.separator_edges,
                markers.articulation_nodes,
                self.separator_gate,
            )
            states.append(propagated)

        weights = self.hop_logits.view(-1, 1, 1)
        base = (weights * torch.stack(states, dim=0)).sum(dim=0)
        residual = self.separator_update(correction / float(self.propagation_steps))
        residual_scale = self.separator_max_scale * torch.sigmoid(self.separator_residual_logit)
        return base + residual_scale * residual

    def _structural_markers(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> StructuralMarkers:
        key = (num_nodes, tuple(edge_index.detach().cpu().reshape(-1).tolist()))
        cached = self._structure_cache.get(key)
        if cached is None:
            cached = compute_structural_markers(edge_index.detach().cpu(), num_nodes)
            self._structure_cache[key] = cached
        return StructuralMarkers(
            articulation_nodes=cached.articulation_nodes.to(device),
            separator_edges=cached.separator_edges.to(device),
        )


class StructuralMarkers:
    def __init__(self, articulation_nodes: torch.Tensor, separator_edges: torch.Tensor) -> None:
        self.articulation_nodes = articulation_nodes
        self.separator_edges = separator_edges


def compute_structural_markers(edge_index: torch.Tensor, num_nodes: int) -> StructuralMarkers:
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    directed_pairs: list[tuple[int, int]] = []
    for src_raw, dst_raw in edge_index.t().tolist():
        src = int(src_raw)
        dst = int(dst_raw)
        directed_pairs.append((src, dst))
        if src == dst:
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)

    articulation_points, bridges = articulation_points_and_bridges(neighbors)
    articulation = torch.zeros(num_nodes, dtype=torch.float)
    for node in articulation_points:
        articulation[node] = 1.0

    separator_edges = torch.zeros(len(directed_pairs), dtype=torch.float)
    for idx, (src, dst) in enumerate(directed_pairs):
        if src == dst:
            continue
        pair = (src, dst) if src < dst else (dst, src)
        if pair in bridges or src in articulation_points or dst in articulation_points:
            separator_edges[idx] = 1.0
    return StructuralMarkers(articulation_nodes=articulation, separator_edges=separator_edges)


def articulation_points_and_bridges(
    neighbors: list[set[int]],
) -> tuple[set[int], set[tuple[int, int]]]:
    num_nodes = len(neighbors)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), num_nodes + 100))
    discovery = [-1] * num_nodes
    low = [0] * num_nodes
    parent = [-1] * num_nodes
    articulation_points: set[int] = set()
    bridges: set[tuple[int, int]] = set()
    time = 0

    def visit(node: int) -> None:
        nonlocal time
        discovery[node] = low[node] = time
        time += 1
        children = 0
        for dst in neighbors[node]:
            if discovery[dst] == -1:
                parent[dst] = node
                children += 1
                visit(dst)
                low[node] = min(low[node], low[dst])
                if parent[node] == -1 and children > 1:
                    articulation_points.add(node)
                if parent[node] != -1 and low[dst] >= discovery[node]:
                    articulation_points.add(node)
                if low[dst] > discovery[node]:
                    bridges.add((node, dst) if node < dst else (dst, node))
            elif dst != parent[node]:
                low[node] = min(low[node], discovery[dst])

    for node in range(num_nodes):
        if discovery[node] == -1:
            visit(node)
    return articulation_points, bridges


def propagate_normalized(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    messages = edge_weight.view(-1, 1) * x[src]
    return scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")


def separator_messages(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    separator_edges: torch.Tensor,
    articulation_nodes: torch.Tensor,
    gate: nn.Module,
) -> torch.Tensor:
    src, dst = edge_index
    active = separator_edges > 0
    if not bool(active.any()):
        return torch.zeros_like(x)
    src_active = src[active]
    dst_active = dst[active]
    structural = torch.stack(
        [articulation_nodes[src_active], articulation_nodes[dst_active]],
        dim=1,
    )
    gate_values = gate(torch.cat([x[src_active], x[dst_active], structural], dim=1))
    messages = edge_weight[active].view(-1, 1) * gate_values * x[src_active]
    return scatter(messages, dst_active, dim=0, dim_size=x.size(0), reduce="sum")


def activation_layer(name: str) -> nn.Module:
    return activation(name)


def bounded_scale_logit(value: float, max_scale: float) -> float:
    if max_scale < 0.0:
        raise ValueError("separator_max_scale must be non-negative")
    if max_scale == 0.0:
        return 0.0
    ratio = min(max(float(value) / float(max_scale), 1e-6), 1.0 - 1e-6)
    return log(ratio / (1.0 - ratio))
