from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.utils import scatter

from gnn_gym.models.base import NodeModel, activation, norm_layer
from gnn_gym.registry import register_model


@register_model("tree_pack_gnn")
class TreePackGNN(NodeModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_trees: int = 4,
        tree_start_idx: int = 0,
        tree_pooling: str = "gated",
        use_graph_channel: bool = True,
        use_tree_channel: bool = True,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str | None = "batchnorm",
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_trees < 1:
            raise ValueError("num_trees must be >= 1")
        if tree_start_idx < 0:
            raise ValueError("tree_start_idx must be >= 0")
        tree_pooling = str(tree_pooling)
        if tree_pooling not in {"gated", "mean"}:
            raise ValueError(f"Unsupported tree_pooling: {tree_pooling}")
        self.output_channels = hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.graph_messages = nn.ModuleList()
        self.tree_messages = nn.ModuleList()
        self.view_gates = nn.ModuleList()
        self.updates = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.graph_messages.append(nn.Linear(hidden_channels, hidden_channels))
            self.tree_messages.append(tree_edge_mlp(hidden_channels, activation))
            self.view_gates.append(nn.Linear(hidden_channels, 1))
            self.updates.append(nn.Linear(hidden_channels * 3, hidden_channels))
            self.norms.append(norm_layer(norm, hidden_channels))
        self.num_trees = num_trees
        self.tree_start_idx = tree_start_idx
        self.tree_pooling = tree_pooling
        self.use_graph_channel = use_graph_channel
        self.use_tree_channel = use_tree_channel
        self.dropout = dropout
        self.activation = activation
        self.task = task
        self.last_gate_weights: list[torch.Tensor] = []
        self._tree_cache: dict[
            tuple[int, int, int, tuple[int, ...], tuple[int, ...] | None],
            TreePack,
        ] = {}

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("TreePackGNN requires edge_index")
        tree_pack = (
            self._tree_pack(edge_index, x.size(0), x.device, batch=batch)
            if self.use_tree_channel
            else None
        )
        x = self.input_proj(x.float())
        x = activation_layer(self.activation)(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        self.last_gate_weights = []
        for idx in range(len(self.updates)):
            graph_msg = (
                full_graph_channel(x, edge_index, self.graph_messages[idx])
                if self.use_graph_channel
                else torch.zeros_like(x)
            )
            if self.use_tree_channel and tree_pack is not None:
                tree_views = tree_view_channels(x, edge_index, tree_pack, self.tree_messages[idx])
                gate_weights = tree_view_weights(
                    tree_views,
                    self.view_gates[idx],
                    self.tree_pooling,
                )
                tree_msg = (gate_weights.unsqueeze(-1) * tree_views).sum(dim=1)
                self.last_gate_weights.append(gate_weights.detach().cpu())
            else:
                tree_msg = torch.zeros_like(x)
            x = self.updates[idx](torch.cat([x, graph_msg, tree_msg], dim=1))
            x = self.norms[idx](x)
            x = activation_layer(self.activation)(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def _tree_pack(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        batch: torch.Tensor | None = None,
    ) -> TreePack:
        batch_cpu = batch.detach().cpu() if batch is not None else None
        key = (
            num_nodes,
            self.num_trees,
            self.tree_start_idx,
            tuple(edge_index.detach().cpu().reshape(-1).tolist()),
            tuple(batch_cpu.tolist()) if batch_cpu is not None else None,
        )
        cached = self._tree_cache.get(key)
        if cached is None:
            cached = compute_tree_pack(
                edge_index.detach().cpu(),
                num_nodes,
                self.num_trees,
                tree_start_idx=self.tree_start_idx,
                batch=batch_cpu,
            )
            self._tree_cache[key] = cached
        return cached.to(device)


@dataclass(frozen=True)
class TreePack:
    tree_edge_masks: torch.Tensor
    source_depth: torch.Tensor
    target_depth: torch.Tensor

    def to(self, device: torch.device) -> TreePack:
        return TreePack(
            tree_edge_masks=self.tree_edge_masks.to(device),
            source_depth=self.source_depth.to(device),
            target_depth=self.target_depth.to(device),
        )


def compute_tree_pack(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_trees: int = 4,
    tree_start_idx: int = 0,
    batch: torch.Tensor | None = None,
) -> TreePack:
    if num_trees < 1:
        raise ValueError("num_trees must be >= 1")
    if tree_start_idx < 0:
        raise ValueError("tree_start_idx must be >= 0")
    edge_index = edge_index.to(dtype=torch.long)
    if batch is None:
        return compute_tree_pack_single(edge_index, num_nodes, num_trees, tree_start_idx)
    return compute_tree_pack_batched(
        edge_index,
        num_nodes,
        num_trees,
        tree_start_idx,
        batch.to(dtype=torch.long),
    )


def compute_tree_pack_batched(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_trees: int,
    tree_start_idx: int,
    batch: torch.Tensor,
) -> TreePack:
    if batch.numel() != num_nodes:
        raise ValueError("batch must have one graph id per node")
    num_edges = edge_index.size(1)
    masks = torch.zeros((num_trees, num_edges), dtype=torch.float)
    source_depth = torch.zeros((num_trees, num_edges), dtype=torch.float)
    target_depth = torch.zeros((num_trees, num_edges), dtype=torch.float)
    if num_edges == 0:
        return TreePack(masks, source_depth, target_depth)

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
        edge_positions = torch.nonzero(src_graph == int(graph_id), as_tuple=False).view(-1)
        if edge_positions.numel() == 0:
            local_lookup[node_ids] = -1
            continue
        local_edge_index = local_lookup[edge_index[:, edge_positions]]
        local_pack = compute_tree_pack_single(
            local_edge_index,
            int(node_ids.numel()),
            num_trees,
            tree_start_idx,
        )
        masks[:, edge_positions] = local_pack.tree_edge_masks
        source_depth[:, edge_positions] = local_pack.source_depth
        target_depth[:, edge_positions] = local_pack.target_depth
        local_lookup[node_ids] = -1
    return TreePack(masks, source_depth, target_depth)


def compute_tree_pack_single(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_trees: int,
    tree_start_idx: int = 0,
) -> TreePack:
    neighbors = build_neighbors(edge_index, num_nodes)
    directed_pairs = [(int(src), int(dst)) for src, dst in edge_index.t().tolist()]
    masks = torch.zeros((num_trees, len(directed_pairs)), dtype=torch.float)
    source_depth = torch.zeros_like(masks)
    target_depth = torch.zeros_like(masks)
    used_counts: dict[tuple[int, int], int] = {}
    for tree_idx in range(num_trees):
        parent, depth, tree_edges = build_tree_view(
            neighbors,
            tree_start_idx + tree_idx,
            used_counts,
        )
        for pair in tree_edges:
            used_counts[pair] = used_counts.get(pair, 0) + 1
        max_depth = max(max(depth), 1)
        for edge_pos, (src, dst) in enumerate(directed_pairs):
            source_depth[tree_idx, edge_pos] = float(depth[src] / max_depth)
            target_depth[tree_idx, edge_pos] = float(depth[dst] / max_depth)
            pair = ordered_pair(src, dst)
            is_tree_edge = pair in tree_edges and (parent[src] == dst or parent[dst] == src)
            if is_tree_edge:
                masks[tree_idx, edge_pos] = 1.0
    return TreePack(masks, source_depth, target_depth)


def build_neighbors(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for src_raw, dst_raw in edge_index.t().tolist():
        src = int(src_raw)
        dst = int(dst_raw)
        if src == dst:
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return neighbors


def build_tree_view(
    neighbors: list[set[int]],
    tree_idx: int,
    used_counts: dict[tuple[int, int], int],
) -> tuple[list[int], list[int], set[tuple[int, int]]]:
    if tree_idx % 4 == 0:
        return bfs_forest(neighbors, root_mode="high_degree", neighbor_mode="high_degree")
    if tree_idx % 4 == 1:
        return bfs_forest(neighbors, root_mode="farthest_low_degree", neighbor_mode="low_degree")
    if tree_idx % 4 == 2:
        return dfs_forest(neighbors, neighbor_mode="ascending")
    return bfs_forest(
        neighbors,
        root_mode="low_degree",
        neighbor_mode="low_overlap",
        used_counts=used_counts,
    )


def bfs_forest(
    neighbors: list[set[int]],
    root_mode: str,
    neighbor_mode: str,
    used_counts: dict[tuple[int, int], int] | None = None,
) -> tuple[list[int], list[int], set[tuple[int, int]]]:
    num_nodes = len(neighbors)
    parent = [-1] * num_nodes
    depth = [0] * num_nodes
    visited = [False] * num_nodes
    tree_edges: set[tuple[int, int]] = set()
    roots = root_order(neighbors, root_mode)
    for root in roots:
        if visited[root]:
            continue
        visited[root] = True
        queue = [root]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for dst in ordered_neighbors(neighbors, node, neighbor_mode, used_counts):
                if visited[dst]:
                    continue
                visited[dst] = True
                parent[dst] = node
                depth[dst] = depth[node] + 1
                tree_edges.add(ordered_pair(node, dst))
                queue.append(dst)
    return parent, depth, tree_edges


def dfs_forest(
    neighbors: list[set[int]],
    neighbor_mode: str,
) -> tuple[list[int], list[int], set[tuple[int, int]]]:
    num_nodes = len(neighbors)
    parent = [-1] * num_nodes
    depth = [0] * num_nodes
    visited = [False] * num_nodes
    tree_edges: set[tuple[int, int]] = set()
    stack: list[tuple[int, int | None]] = []
    for root in range(num_nodes):
        if visited[root]:
            continue
        stack.append((root, None))
        while stack:
            node, maybe_parent = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            if maybe_parent is not None:
                parent[node] = maybe_parent
                depth[node] = depth[maybe_parent] + 1
                tree_edges.add(ordered_pair(node, maybe_parent))
            for dst in reversed(ordered_neighbors(neighbors, node, neighbor_mode)):
                if not visited[dst]:
                    stack.append((dst, node))
    return parent, depth, tree_edges


def root_order(neighbors: list[set[int]], root_mode: str) -> list[int]:
    num_nodes = len(neighbors)
    degrees = [len(values) for values in neighbors]
    if root_mode == "high_degree":
        return sorted(range(num_nodes), key=lambda node: (-degrees[node], node))
    if root_mode == "low_degree":
        return sorted(range(num_nodes), key=lambda node: (degrees[node], node))
    if root_mode == "farthest_low_degree":
        start = sorted(range(num_nodes), key=lambda node: (-degrees[node], node))[0]
        distances = bfs_distances(neighbors, start)
        return sorted(range(num_nodes), key=lambda node: (-distances[node], degrees[node], node))
    raise ValueError(f"Unsupported root_mode: {root_mode}")


def bfs_distances(neighbors: list[set[int]], root: int) -> list[int]:
    distances = [-1] * len(neighbors)
    distances[root] = 0
    queue = [root]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for dst in sorted(neighbors[node]):
            if distances[dst] >= 0:
                continue
            distances[dst] = distances[node] + 1
            queue.append(dst)
    max_seen = max(distances)
    return [distance if distance >= 0 else max_seen + 1 for distance in distances]


def ordered_neighbors(
    neighbors: list[set[int]],
    node: int,
    neighbor_mode: str,
    used_counts: dict[tuple[int, int], int] | None = None,
) -> list[int]:
    degrees = [len(values) for values in neighbors]
    values = list(neighbors[node])
    if neighbor_mode == "ascending":
        return sorted(values)
    if neighbor_mode == "high_degree":
        return sorted(values, key=lambda dst: (-degrees[dst], dst))
    if neighbor_mode == "low_degree":
        return sorted(values, key=lambda dst: (degrees[dst], dst))
    if neighbor_mode == "low_overlap":
        used_counts = used_counts or {}
        return sorted(
            values,
            key=lambda dst: (used_counts.get(ordered_pair(node, dst), 0), degrees[dst], dst),
        )
    raise ValueError(f"Unsupported neighbor_mode: {neighbor_mode}")


def ordered_pair(src: int, dst: int) -> tuple[int, int]:
    return (src, dst) if src < dst else (dst, src)


def full_graph_channel(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    message_net: nn.Module,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    src, dst = edge_index
    messages = message_net(x[src])
    return scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="mean")


def tree_view_channels(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    tree_pack: TreePack,
    message_net: nn.Module,
) -> torch.Tensor:
    views = []
    for tree_idx in range(tree_pack.tree_edge_masks.size(0)):
        views.append(tree_channel(x, edge_index, tree_pack, tree_idx, message_net))
    return torch.stack(views, dim=1)


def tree_view_weights(
    tree_views: torch.Tensor,
    gate: nn.Module,
    tree_pooling: str,
) -> torch.Tensor:
    if tree_pooling == "mean":
        return torch.full(
            tree_views.shape[:2],
            1.0 / tree_views.size(1),
            dtype=tree_views.dtype,
            device=tree_views.device,
        )
    if tree_pooling == "gated":
        return gate(tree_views).squeeze(-1).softmax(dim=1)
    raise ValueError(f"Unsupported tree_pooling: {tree_pooling}")


def tree_channel(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    tree_pack: TreePack,
    tree_idx: int,
    message_net: nn.Module,
) -> torch.Tensor:
    mask = tree_pack.tree_edge_masks[tree_idx] > 0
    if not bool(mask.any()):
        return torch.zeros_like(x)
    src, dst = edge_index
    edge_features = torch.stack(
        [
            tree_pack.source_depth[tree_idx, mask],
            tree_pack.target_depth[tree_idx, mask],
            (tree_pack.source_depth[tree_idx, mask] - tree_pack.target_depth[tree_idx, mask]).abs(),
        ],
        dim=1,
    )
    messages = message_net(torch.cat([x[src[mask]], edge_features], dim=1))
    return scatter(messages, dst[mask], dim=0, dim_size=x.size(0), reduce="mean")


def tree_edge_mlp(hidden_channels: int, activation_name: str) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_channels + 3, hidden_channels),
        activation_layer(activation_name),
        nn.Linear(hidden_channels, hidden_channels),
    )


def activation_layer(name: str) -> nn.Module:
    return activation(name)
