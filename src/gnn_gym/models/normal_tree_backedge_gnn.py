from __future__ import annotations

import sys

import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("normal_tree_backedge_gnn")
class NormalTreeBackedgeGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        num_tree_orders: int = 1,
        edge_role_mode: str = "true",
        dfs_order_mode: str = "deterministic",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_tree_orders < 1:
            raise ValueError("num_tree_orders must be >= 1")
        edge_role_mode = normalize_mode(edge_role_mode)
        dfs_order_mode = normalize_mode(dfs_order_mode)
        if edge_role_mode not in {"true", "collapsed", "tree_only", "back_only", "shuffled"}:
            raise ValueError(f"Unsupported edge_role_mode: {edge_role_mode}")
        if dfs_order_mode not in {"deterministic", "random"}:
            raise ValueError(f"Unsupported dfs_order_mode: {dfs_order_mode}")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.up_messages = nn.ModuleList()
        self.down_messages = nn.ModuleList()
        self.back_messages = nn.ModuleList()
        self.updates = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.up_messages.append(edge_mlp(hidden_channels, activation))
            self.down_messages.append(edge_mlp(hidden_channels, activation))
            self.back_messages.append(edge_mlp(hidden_channels, activation))
            self.updates.append(nn.Linear(hidden_channels * 4, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
        self.activation = activation
        self.dropout = dropout
        self.task = task
        self.num_tree_orders = num_tree_orders
        self.edge_role_mode = edge_role_mode
        self.dfs_order_mode = dfs_order_mode
        self._structure_cache: dict[
            tuple[int, int, str, str, tuple[int, ...], tuple[int, ...] | None],
            list[TreeBackedgeMarkers],
        ] = {}

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("NormalTreeBackedgeGNN requires edge_index")
        marker_sets = self._markers(edge_index, x.size(0), x.device, batch=batch)
        x = self.input_proj(x.float())
        x = activation_layer(self.activation)(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        for idx in range(len(self.updates)):
            up, down, back = multi_order_channels(
                x,
                edge_index,
                marker_sets,
                self.up_messages[idx],
                self.down_messages[idx],
                self.back_messages[idx],
            )
            x = self.updates[idx](torch.cat([x, up, down, back], dim=1))
            x = self.norms[idx](x)
            x = activation_layer(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def _markers(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        batch: torch.Tensor | None = None,
    ) -> list[TreeBackedgeMarkers]:
        batch_cpu = batch.detach().cpu() if batch is not None else None
        key = (
            num_nodes,
            self.num_tree_orders,
            self.edge_role_mode,
            self.dfs_order_mode,
            tuple(edge_index.detach().cpu().reshape(-1).tolist()),
            tuple(batch_cpu.tolist()) if batch_cpu is not None else None,
        )
        cached = self._structure_cache.get(key)
        if cached is None:
            cached = compute_tree_backedge_marker_sets(
                edge_index.detach().cpu(),
                num_nodes,
                self.num_tree_orders,
                edge_role_mode=self.edge_role_mode,
                dfs_order_mode=self.dfs_order_mode,
                batch=batch_cpu,
            )
            self._structure_cache[key] = cached
        return [markers.to(device) for markers in cached]


class TreeBackedgeMarkers:
    def __init__(
        self,
        up_edges: torch.Tensor,
        down_edges: torch.Tensor,
        back_edges: torch.Tensor,
        source_depth: torch.Tensor,
        target_depth: torch.Tensor,
        span: torch.Tensor,
    ) -> None:
        self.up_edges = up_edges
        self.down_edges = down_edges
        self.back_edges = back_edges
        self.source_depth = source_depth
        self.target_depth = target_depth
        self.span = span

    def to(self, device: torch.device) -> TreeBackedgeMarkers:
        return TreeBackedgeMarkers(
            up_edges=self.up_edges.to(device),
            down_edges=self.down_edges.to(device),
            back_edges=self.back_edges.to(device),
            source_depth=self.source_depth.to(device),
            target_depth=self.target_depth.to(device),
            span=self.span.to(device),
        )


def compute_tree_backedge_markers(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_role_mode: str = "true",
    dfs_order_mode: str = "deterministic",
    batch: torch.Tensor | None = None,
) -> TreeBackedgeMarkers:
    edge_role_mode = normalize_mode(edge_role_mode)
    dfs_order_mode = normalize_mode(dfs_order_mode)
    order_idx = dfs_order_index(0, dfs_order_mode)
    if batch is None:
        markers = compute_tree_backedge_marker_set(edge_index, num_nodes, order_idx=order_idx)
    else:
        markers = compute_batched_tree_backedge_marker_set(
            edge_index,
            num_nodes,
            batch,
            order_idx=order_idx,
        )
    return apply_edge_role_mode(markers, edge_role_mode=edge_role_mode, order_idx=order_idx)


def compute_tree_backedge_marker_sets(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_orders: int,
    edge_role_mode: str = "true",
    dfs_order_mode: str = "deterministic",
    batch: torch.Tensor | None = None,
) -> list[TreeBackedgeMarkers]:
    if num_orders < 1:
        raise ValueError("num_orders must be >= 1")
    edge_role_mode = normalize_mode(edge_role_mode)
    dfs_order_mode = normalize_mode(dfs_order_mode)
    marker_sets = []
    for order_idx in range(num_orders):
        actual_order_idx = dfs_order_index(order_idx, dfs_order_mode)
        if batch is None:
            markers = compute_tree_backedge_marker_set(
                edge_index,
                num_nodes,
                actual_order_idx,
            )
        else:
            markers = compute_batched_tree_backedge_marker_set(
                edge_index,
                num_nodes,
                batch,
                actual_order_idx,
            )
        marker_sets.append(
            apply_edge_role_mode(
                markers,
                edge_role_mode=edge_role_mode,
                order_idx=actual_order_idx,
            )
        )
    return marker_sets


def normalize_mode(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def dfs_order_index(order_idx: int, dfs_order_mode: str) -> int:
    if dfs_order_mode == "deterministic":
        return order_idx
    if dfs_order_mode == "random":
        return order_idx + 10_000
    raise ValueError(f"Unsupported dfs_order_mode: {dfs_order_mode}")


def apply_edge_role_mode(
    markers: TreeBackedgeMarkers,
    edge_role_mode: str,
    order_idx: int,
) -> TreeBackedgeMarkers:
    if edge_role_mode == "true":
        return markers

    active_edges = (
        (markers.up_edges > 0) | (markers.down_edges > 0) | (markers.back_edges > 0)
    ).float()
    zeros = torch.zeros_like(active_edges)
    if edge_role_mode == "collapsed":
        return TreeBackedgeMarkers(
            up_edges=active_edges,
            down_edges=active_edges.clone(),
            back_edges=active_edges.clone(),
            source_depth=markers.source_depth,
            target_depth=markers.target_depth,
            span=markers.span,
        )
    if edge_role_mode == "tree_only":
        return TreeBackedgeMarkers(
            up_edges=markers.up_edges,
            down_edges=markers.down_edges,
            back_edges=zeros,
            source_depth=markers.source_depth,
            target_depth=markers.target_depth,
            span=markers.span,
        )
    if edge_role_mode == "back_only":
        return TreeBackedgeMarkers(
            up_edges=zeros,
            down_edges=zeros.clone(),
            back_edges=markers.back_edges,
            source_depth=markers.source_depth,
            target_depth=markers.target_depth,
            span=markers.span,
        )
    if edge_role_mode == "shuffled":
        edge_ids = torch.arange(active_edges.numel(), dtype=torch.long)
        roles = (edge_ids * 1_103_515_245 + 12_345 * (order_idx + 1)) % 3
        return TreeBackedgeMarkers(
            up_edges=((roles == 0) & (active_edges > 0)).float(),
            down_edges=((roles == 1) & (active_edges > 0)).float(),
            back_edges=((roles == 2) & (active_edges > 0)).float(),
            source_depth=markers.source_depth,
            target_depth=markers.target_depth,
            span=markers.span,
        )
    raise ValueError(f"Unsupported edge_role_mode: {edge_role_mode}")


def compute_batched_tree_backedge_marker_set(
    edge_index: torch.Tensor,
    num_nodes: int,
    batch: torch.Tensor,
    order_idx: int,
) -> TreeBackedgeMarkers:
    if batch.numel() != num_nodes:
        raise ValueError("batch must have one graph id per node")
    batch = batch.to(dtype=torch.long)
    edge_index = edge_index.to(dtype=torch.long)
    num_edges = edge_index.size(1)
    markers = empty_markers(num_edges)
    if num_edges == 0:
        return markers

    src_graph = batch[edge_index[0]]
    dst_graph = batch[edge_index[1]]
    if not bool(torch.equal(src_graph, dst_graph)):
        raise ValueError("Batched graph edge_index contains cross-graph edges")

    local_lookup = torch.full((num_nodes,), -1, dtype=torch.long)
    for graph_id in torch.unique(batch, sorted=True).tolist():
        node_ids = torch.nonzero(batch == int(graph_id), as_tuple=False).view(-1)
        if node_ids.numel() == 0:
            continue
        local_lookup[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long)
        edge_mask = src_graph == int(graph_id)
        edge_positions = torch.nonzero(edge_mask, as_tuple=False).view(-1)
        if edge_positions.numel() == 0:
            continue
        local_edge_index = local_lookup[edge_index[:, edge_positions]]
        local_markers = compute_tree_backedge_marker_set(
            local_edge_index,
            int(node_ids.numel()),
            order_idx,
        )
        copy_markers(markers, local_markers, edge_positions)
        local_lookup[node_ids] = -1
    return markers


def empty_markers(num_edges: int) -> TreeBackedgeMarkers:
    zeros = torch.zeros(num_edges, dtype=torch.float)
    return TreeBackedgeMarkers(
        up_edges=zeros.clone(),
        down_edges=zeros.clone(),
        back_edges=zeros.clone(),
        source_depth=zeros.clone(),
        target_depth=zeros.clone(),
        span=zeros.clone(),
    )


def copy_markers(
    target: TreeBackedgeMarkers,
    source: TreeBackedgeMarkers,
    edge_positions: torch.Tensor,
) -> None:
    target.up_edges[edge_positions] = source.up_edges
    target.down_edges[edge_positions] = source.down_edges
    target.back_edges[edge_positions] = source.back_edges
    target.source_depth[edge_positions] = source.source_depth
    target.target_depth[edge_positions] = source.target_depth
    target.span[edge_positions] = source.span


def compute_tree_backedge_marker_set(
    edge_index: torch.Tensor,
    num_nodes: int,
    order_idx: int,
) -> TreeBackedgeMarkers:
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

    parent, depth, tree_pairs = deterministic_dfs_forest(neighbors, order_idx)
    max_depth = max(max(depth), 1)
    up_edges = torch.zeros(len(directed_pairs), dtype=torch.float)
    down_edges = torch.zeros(len(directed_pairs), dtype=torch.float)
    back_edges = torch.zeros(len(directed_pairs), dtype=torch.float)
    source_depth = torch.zeros(len(directed_pairs), dtype=torch.float)
    target_depth = torch.zeros(len(directed_pairs), dtype=torch.float)
    span = torch.zeros(len(directed_pairs), dtype=torch.float)

    for idx, (src, dst) in enumerate(directed_pairs):
        src_depth = depth[src] / max_depth
        dst_depth = depth[dst] / max_depth
        source_depth[idx] = float(src_depth)
        target_depth[idx] = float(dst_depth)
        span[idx] = abs(float(src_depth - dst_depth))
        if src == dst:
            continue
        pair = (src, dst) if src < dst else (dst, src)
        if pair in tree_pairs:
            if parent[dst] == src:
                down_edges[idx] = 1.0
            elif parent[src] == dst:
                up_edges[idx] = 1.0
        else:
            back_edges[idx] = 1.0

    return TreeBackedgeMarkers(
        up_edges=up_edges,
        down_edges=down_edges,
        back_edges=back_edges,
        source_depth=source_depth,
        target_depth=target_depth,
        span=span,
    )


def deterministic_dfs_forest(
    neighbors: list[set[int]],
    order_idx: int = 0,
) -> tuple[list[int], list[int], set[tuple[int, int]]]:
    num_nodes = len(neighbors)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), num_nodes + 100))
    parent = [-1] * num_nodes
    depth = [0] * num_nodes
    visited = [False] * num_nodes
    tree_pairs: set[tuple[int, int]] = set()
    node_order = ordered_nodes(num_nodes, order_idx)

    def visit(node: int) -> None:
        visited[node] = True
        for dst in ordered_neighbors(neighbors[node], num_nodes, order_idx):
            if visited[dst]:
                continue
            parent[dst] = node
            depth[dst] = depth[node] + 1
            tree_pairs.add((node, dst) if node < dst else (dst, node))
            visit(dst)

    for node in node_order:
        if not visited[node]:
            visit(node)
    return parent, depth, tree_pairs


def ordered_nodes(num_nodes: int, order_idx: int) -> list[int]:
    if order_idx == 0:
        return list(range(num_nodes))
    if order_idx == 1:
        return list(reversed(range(num_nodes)))
    stride = (2 * order_idx) + 1
    while gcd(stride, num_nodes) != 1:
        stride += 2
    offset = (order_idx * 7) % max(num_nodes, 1)
    return sorted(range(num_nodes), key=lambda node: ((node * stride + offset) % num_nodes, node))


def ordered_neighbors(neighbors: set[int], num_nodes: int, order_idx: int) -> list[int]:
    if order_idx == 0:
        return sorted(neighbors)
    if order_idx == 1:
        return sorted(neighbors, reverse=True)
    order = {node: rank for rank, node in enumerate(ordered_nodes(num_nodes, order_idx))}
    return sorted(neighbors, key=lambda node: (order[node], node))


def gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return abs(left)


def masked_channel(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    mask: torch.Tensor,
    message_net: nn.Module,
) -> torch.Tensor:
    active = mask > 0
    if not bool(active.any()):
        return torch.zeros_like(x)
    src, dst = edge_index
    src_active = src[active]
    dst_active = dst[active]
    messages = message_net(torch.cat([x[src_active], edge_features[active]], dim=1))
    return scatter(messages, dst_active, dim=0, dim_size=x.size(0), reduce="mean")


def multi_order_channels(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    marker_sets: list[TreeBackedgeMarkers],
    up_message: nn.Module,
    down_message: nn.Module,
    back_message: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    up_values = []
    down_values = []
    back_values = []
    for markers in marker_sets:
        edge_features = torch.stack(
            [
                markers.source_depth,
                markers.target_depth,
                markers.span,
            ],
            dim=1,
        )
        up_values.append(masked_channel(x, edge_index, edge_features, markers.up_edges, up_message))
        down_values.append(
            masked_channel(x, edge_index, edge_features, markers.down_edges, down_message)
        )
        back_values.append(
            masked_channel(x, edge_index, edge_features, markers.back_edges, back_message)
        )
    return (
        torch.stack(up_values, dim=0).mean(dim=0),
        torch.stack(down_values, dim=0).mean(dim=0),
        torch.stack(back_values, dim=0).mean(dim=0),
    )


def edge_mlp(hidden_channels: int, activation_name: str) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_channels + 3, hidden_channels),
        activation_layer(activation_name),
        nn.Linear(hidden_channels, hidden_channels),
    )


def activation_layer(name: str) -> nn.Module:
    return activation(name)
