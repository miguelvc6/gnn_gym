from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from torch_geometric.data import Data

from gnn_gym.data.adapters import DatasetBundle
from gnn_gym.registry import register_dataset


def _dataset_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return (config or {}).get("dataset", {})


def _root(config: dict[str, Any], default: str) -> str:
    return str(config.get("root", default))


def _single_mask_from_indices(num_nodes: int, idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx.view(-1).long()] = True
    return mask


def _normalize_node_masks(data: Data, split_idx: dict[str, torch.Tensor] | None = None) -> Data:
    if split_idx is not None:
        data.train_mask = _single_mask_from_indices(data.num_nodes, split_idx["train"])
        data.val_mask = _single_mask_from_indices(data.num_nodes, split_idx["valid"])
        data.test_mask = _single_mask_from_indices(data.num_nodes, split_idx["test"])
    for attr in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, attr)
        if mask.ndim > 1:
            setattr(data, attr, mask[:, 0])
    if data.y.ndim > 1 and data.y.shape[-1] == 1:
        data.y = data.y.view(-1)
    return data


@contextmanager
def _torch_load_compat() -> Any:
    original_load = torch.load

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


@register_dataset("toy-node")
def load_toy_node_dataset(config: dict[str, Any] | None = None) -> DatasetBundle:
    dataset_config = _dataset_config(config)
    num_nodes = int(dataset_config.get("num_nodes", 48))
    num_features = int(dataset_config.get("num_features", 8))
    num_classes = int(dataset_config.get("num_classes", 3))

    generator = torch.Generator().manual_seed(12345)
    x = torch.randn((num_nodes, num_features), generator=generator)
    weights = torch.randn((num_features, num_classes), generator=generator)
    y = (x @ weights).argmax(dim=1)

    source = torch.arange(num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
    skip = (source + 3) % num_nodes
    edge_index = torch.stack(
        [
            torch.cat([source, target, source, skip]),
            torch.cat([target, source, skip, source]),
        ],
        dim=0,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: int(0.6 * num_nodes)] = True
    val_mask[int(0.6 * num_nodes) : int(0.8 * num_nodes)] = True
    test_mask[int(0.8 * num_nodes) :] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return DatasetBundle(
        name="toy-node",
        task="node_classification",
        metric="accuracy",
        trainer="full_batch_node",
        evaluator="accuracy",
        data=data,
        num_features=num_features,
        num_outputs=num_classes,
    )


@register_dataset("toy-graph")
def load_toy_graph_dataset(config: dict[str, Any] | None = None) -> DatasetBundle:
    dataset_config = _dataset_config(config)
    num_graphs = int(dataset_config.get("num_graphs", 24))
    num_nodes = int(dataset_config.get("num_nodes_per_graph", 8))
    num_features = int(dataset_config.get("num_features", 6))
    generator = torch.Generator().manual_seed(23456)
    graphs: list[Data] = []
    for graph_idx in range(num_graphs):
        x = torch.randn((num_nodes, num_features), generator=generator)
        source = torch.arange(num_nodes, dtype=torch.long)
        target = (source + 1) % num_nodes
        edge_index = torch.stack([torch.cat([source, target]), torch.cat([target, source])])
        y = torch.tensor([[float(x.mean() > 0)]])
        graphs.append(Data(x=x, edge_index=edge_index, y=y, graph_idx=graph_idx))
    split_idx = {
        "train": torch.arange(0, int(0.6 * num_graphs)),
        "valid": torch.arange(int(0.6 * num_graphs), int(0.8 * num_graphs)),
        "test": torch.arange(int(0.8 * num_graphs), num_graphs),
    }
    return DatasetBundle(
        name="toy-graph",
        task="graph_binary_classification",
        metric="average_precision",
        trainer="graph_prediction",
        evaluator="average_precision",
        dataset=graphs,
        split_idx=split_idx,
        num_features=num_features,
        num_outputs=1,
    )


@register_dataset("normal-tree-backedge")
def load_normal_tree_backedge_dataset(config: dict[str, Any] | None = None) -> DatasetBundle:
    dataset_config = _dataset_config(config)
    num_graphs = int(dataset_config.get("num_graphs", 120))
    num_nodes = int(dataset_config.get("num_nodes_per_graph", 20))
    variant = str(dataset_config.get("variant", "cycle_matching_v4"))
    if num_nodes < 8 or num_nodes % 2 != 0:
        raise ValueError("normal-tree-backedge requires an even node count >= 8")

    graphs = []
    generator = torch.Generator().manual_seed(int(dataset_config.get("generation_seed", 45678)))
    for graph_idx in range(num_graphs):
        label = graph_idx % 2
        graphs.append(_normal_tree_backedge_graph(num_nodes, label, graph_idx, variant, generator))
    permutation = torch.randperm(num_graphs, generator=generator).tolist()
    graphs = [graphs[idx] for idx in permutation]
    train_end = int(0.6 * num_graphs)
    val_end = int(0.8 * num_graphs)
    split_idx = {
        "train": torch.arange(0, train_end),
        "valid": torch.arange(train_end, val_end),
        "test": torch.arange(val_end, num_graphs),
    }
    return DatasetBundle(
        name="normal-tree-backedge",
        task="graph_binary_classification",
        metric="average_precision",
        trainer="graph_prediction",
        evaluator="average_precision",
        dataset=graphs,
        split_idx=split_idx,
        num_features=int(graphs[0].num_features),
        num_outputs=1,
    )


def _normal_tree_backedge_graph(
    num_nodes: int,
    label: int,
    graph_idx: int,
    variant: str,
    generator: torch.Generator,
) -> Data:
    if variant == "endpoint_pairing_v2":
        return _endpoint_pairing_backedge_graph(num_nodes, label, graph_idx)
    if variant not in {"cycle_matching_v3", "cycle_matching_v4"}:
        raise ValueError(f"Unsupported normal-tree-backedge variant: {variant}")
    undirected_edges = {(idx, (idx + 1) % num_nodes) for idx in range(num_nodes)}
    matching = _sample_cycle_matching(num_nodes, label, generator)
    for src, dst in matching:
        pair = (src, dst) if src < dst else (dst, src)
        undirected_edges.add(pair)
    relabel = torch.randperm(num_nodes, generator=generator)
    relabeled_edges = {
        _ordered_pair(int(relabel[src]), int(relabel[dst])) for src, dst in undirected_edges
    }
    edge_index = _directed_edge_index(relabeled_edges)
    x = torch.ones((num_nodes, 1), dtype=torch.float)
    y = torch.tensor([[float(label)]])
    return Data(x=x, edge_index=edge_index, y=y, graph_idx=graph_idx)


def _endpoint_pairing_backedge_graph(num_nodes: int, label: int, graph_idx: int) -> Data:
    undirected_edges = {(idx, idx + 1) for idx in range(num_nodes - 1)}
    if label == 0:
        chords = [(0, 2), (4, 7), (num_nodes - 3, num_nodes - 1)]
    else:
        chords = [(0, 7), (2, num_nodes - 3), (4, num_nodes - 1)]
    for src, dst in chords:
        pair = (src, dst) if src < dst else (dst, src)
        undirected_edges.add(pair)
    chord_endpoint = torch.zeros(num_nodes)
    degree = torch.zeros(num_nodes)
    for src, dst in undirected_edges:
        degree[src] += 1.0
        degree[dst] += 1.0
    for src, dst in chords:
        chord_endpoint[src] = 1.0
        chord_endpoint[dst] = 1.0
    x = torch.stack(
        [
            torch.ones(num_nodes),
            chord_endpoint,
            degree / degree.max().clamp_min(1.0),
        ],
        dim=1,
    )
    sources = []
    targets = []
    for src, dst in sorted(undirected_edges):
        sources.extend([src, dst])
        targets.extend([dst, src])
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    y = torch.tensor([[float(label)]])
    return Data(x=x, edge_index=edge_index, y=y, graph_idx=graph_idx)


def _sample_cycle_matching(
    num_nodes: int,
    label: int,
    generator: torch.Generator,
) -> list[tuple[int, int]]:
    max_crossings = (num_nodes // 2) * ((num_nodes // 2) - 1) / 2
    low_max = int(0.40 * max_crossings)
    high_min = int(0.60 * max_crossings)
    for _ in range(10_000):
        matching = _random_nonlocal_matching(num_nodes, generator)
        crossings = _count_matching_crossings(matching)
        if label == 0 and crossings <= low_max:
            return matching
        if label == 1 and crossings >= high_min:
            return matching
    raise RuntimeError("Failed to sample a normal-tree-backedge matching")


def _random_nonlocal_matching(
    num_nodes: int,
    generator: torch.Generator,
) -> list[tuple[int, int]]:
    for _ in range(1_000):
        remaining = list(range(num_nodes))
        matching: list[tuple[int, int]] = []
        while remaining:
            src = remaining.pop(0)
            candidates = [
                dst
                for dst in remaining
                if min((dst - src) % num_nodes, (src - dst) % num_nodes) >= 3
            ]
            if not candidates:
                break
            candidate_idx = int(torch.randint(len(candidates), (1,), generator=generator).item())
            dst = candidates[candidate_idx]
            remaining.remove(dst)
            matching.append(_ordered_pair(src, dst))
        if len(matching) == num_nodes // 2:
            return matching
    raise RuntimeError("Failed to sample a nonlocal perfect matching")


def _count_matching_crossings(matching: list[tuple[int, int]]) -> int:
    crossings = 0
    arcs = [tuple(sorted(pair)) for pair in matching]
    for idx, (a, b) in enumerate(arcs):
        for c, d in arcs[idx + 1 :]:
            if (a < c < b < d) or (c < a < d < b):
                crossings += 1
    return crossings


def _ordered_pair(src: int, dst: int) -> tuple[int, int]:
    return (src, dst) if src < dst else (dst, src)


def _directed_edge_index(undirected_edges: set[tuple[int, int]]) -> torch.Tensor:
    sources = []
    targets = []
    for src, dst in sorted(undirected_edges):
        sources.extend([src, dst])
        targets.extend([dst, src])
    return torch.tensor([sources, targets], dtype=torch.long)


@register_dataset("toy-link")
def load_toy_link_dataset(config: dict[str, Any] | None = None) -> DatasetBundle:
    dataset_config = _dataset_config(config)
    num_nodes = int(dataset_config.get("num_nodes", 32))
    num_features = int(dataset_config.get("num_features", 8))
    generator = torch.Generator().manual_seed(34567)
    x = torch.randn((num_nodes, num_features), generator=generator)
    source = torch.arange(num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
    edge = torch.stack([source, target], dim=1)
    edge_index = torch.cat([edge, edge.flip(1)], dim=0).t().contiguous()
    split_idx = {
        "train": {"edge": edge[:20]},
        "valid": {
            "edge": edge[20:26],
            "edge_neg": torch.stack([source[20:26], (source[20:26] + 7) % num_nodes], dim=1),
        },
        "test": {
            "edge": edge[26:],
            "edge_neg": torch.stack([source[26:], (source[26:] + 11) % num_nodes], dim=1),
        },
    }
    return DatasetBundle(
        name="toy-link",
        task="link_prediction",
        metric="hits@50",
        trainer="link_prediction",
        evaluator="hits@50",
        data=Data(x=x, edge_index=edge_index),
        split_idx=split_idx,
        num_features=num_features,
        num_outputs=int((config or {}).get("model", {}).get("hidden_channels", 16)),
    )


def _load_planetoid(config: dict[str, Any] | None, name: str) -> DatasetBundle:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    dataset_config = _dataset_config(config)
    dataset = Planetoid(
        root=_root(dataset_config, "data/pyg"),
        name=name,
        transform=NormalizeFeatures(),
    )
    data = _normalize_node_masks(dataset[0])
    return DatasetBundle(
        name=name.lower(),
        task="node_classification",
        metric="accuracy",
        trainer="full_batch_node",
        evaluator="accuracy",
        data=data,
        dataset=dataset,
        num_features=dataset.num_features,
        num_outputs=dataset.num_classes,
    )


@register_dataset("cora")
def load_cora(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_planetoid(config, "Cora")


@register_dataset("pubmed")
def load_pubmed(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_planetoid(config, "PubMed")


def _load_ogbn(config: dict[str, Any] | None, name: str, trainer: str) -> DatasetBundle:
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset_config = _dataset_config(config)
    with _torch_load_compat():
        dataset = PygNodePropPredDataset(name=name, root=_root(dataset_config, "data/ogb"))
    split_idx = dataset.get_idx_split()
    data = _normalize_node_masks(dataset[0], split_idx)
    return DatasetBundle(
        name=name,
        task="node_classification",
        metric="accuracy",
        trainer=trainer,
        evaluator="ogb",
        data=data,
        dataset=dataset,
        split_idx=split_idx,
        num_features=dataset.num_features,
        num_outputs=int(dataset.num_classes),
    )


@register_dataset("ogbn-arxiv")
def load_ogbn_arxiv(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_ogbn(config, "ogbn-arxiv", "full_batch_node")


@register_dataset("ogbn-products")
def load_ogbn_products(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_ogbn(config, "ogbn-products", "neighbor_node")


def _load_heterophily(
    config: dict[str, Any] | None,
    registry_name: str,
    pyg_name: str,
) -> DatasetBundle:
    from torch_geometric.datasets import HeterophilousGraphDataset

    dataset_config = _dataset_config(config)
    dataset = HeterophilousGraphDataset(root=_root(dataset_config, "data/pyg"), name=pyg_name)
    data = _normalize_node_masks(dataset[0])
    return DatasetBundle(
        name=registry_name,
        task="node_classification",
        metric="accuracy",
        trainer="full_batch_node",
        evaluator="accuracy",
        data=data,
        dataset=dataset,
        num_features=dataset.num_features,
        num_outputs=dataset.num_classes,
    )


@register_dataset("roman-empire")
def load_roman_empire(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_heterophily(config, "roman-empire", "Roman-empire")


@register_dataset("amazon-ratings")
def load_amazon_ratings(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_heterophily(config, "amazon-ratings", "Amazon-ratings")


def _load_ogbg(config: dict[str, Any] | None, name: str, task: str, metric: str) -> DatasetBundle:
    from ogb.graphproppred import PygGraphPropPredDataset

    dataset_config = _dataset_config(config)
    with _torch_load_compat():
        dataset = PygGraphPropPredDataset(name=name, root=_root(dataset_config, "data/ogb"))
    split_idx = dataset.get_idx_split()
    num_outputs = int(getattr(dataset, "num_tasks", 1))
    return DatasetBundle(
        name=name,
        task=task,
        metric=metric,
        trainer="graph_prediction",
        evaluator="ogb",
        dataset=dataset,
        split_idx=split_idx,
        num_features=dataset.num_features,
        num_outputs=num_outputs,
        higher_is_better=True,
    )


@register_dataset("ogbg-molhiv")
def load_ogbg_molhiv(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_ogbg(config, "ogbg-molhiv", "graph_binary_classification", "rocauc")


@register_dataset("ogbg-molpcba")
def load_ogbg_molpcba(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_ogbg(config, "ogbg-molpcba", "graph_multilabel_classification", "ap")


def _load_lrgb(config: dict[str, Any] | None, name: str, task: str, metric: str) -> DatasetBundle:
    from torch_geometric.datasets import LRGBDataset

    dataset_config = _dataset_config(config)
    root = _root(dataset_config, "data/pyg")
    datasets = {
        split: LRGBDataset(root=root, name=name, split=split)
        for split in ("train", "val", "test")
    }
    sample = datasets["train"][0]
    y = sample.y
    num_outputs = int(y.numel() if y.ndim == 1 else y.shape[-1])
    return DatasetBundle(
        name=name.lower(),
        task=task,
        metric=metric,
        trainer="graph_prediction",
        evaluator=metric,
        dataset=datasets,
        split_idx=None,
        num_features=sample.num_node_features,
        num_outputs=num_outputs,
        higher_is_better=metric != "mae",
    )


@register_dataset("peptides-func")
def load_peptides_func(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_lrgb(
        config,
        "Peptides-func",
        "graph_multilabel_classification",
        "average_precision",
    )


@register_dataset("peptides-struct")
def load_peptides_struct(config: dict[str, Any] | None = None) -> DatasetBundle:
    return _load_lrgb(config, "Peptides-struct", "graph_regression", "mae")


@register_dataset("ogbl-collab")
def load_ogbl_collab(config: dict[str, Any] | None = None) -> DatasetBundle:
    from ogb.linkproppred import PygLinkPropPredDataset

    dataset_config = _dataset_config(config)
    with _torch_load_compat():
        dataset = PygLinkPropPredDataset(name="ogbl-collab", root=_root(dataset_config, "data/ogb"))
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    if data.x is None:
        embedding_dim = int(dataset_config.get("embedding_dim", 128))
        data.x = torch.nn.Embedding(data.num_nodes, embedding_dim).weight.detach()
    hidden = int((config or {}).get("model", {}).get("hidden_channels", 128))
    return DatasetBundle(
        name="ogbl-collab",
        task="link_prediction",
        metric="hits@50",
        trainer="link_prediction",
        evaluator="ogb",
        data=data,
        dataset=dataset,
        split_idx=split_edge,
        num_features=data.num_features,
        num_outputs=hidden,
        higher_is_better=True,
    )


def available_dataset_loaders() -> dict[str, Callable[[dict[str, Any] | None], DatasetBundle]]:
    return {
        "toy-node": load_toy_node_dataset,
        "toy-graph": load_toy_graph_dataset,
        "toy-link": load_toy_link_dataset,
        "cora": load_cora,
        "pubmed": load_pubmed,
        "ogbn-arxiv": load_ogbn_arxiv,
        "ogbn-products": load_ogbn_products,
        "roman-empire": load_roman_empire,
        "amazon-ratings": load_amazon_ratings,
        "ogbg-molhiv": load_ogbg_molhiv,
        "ogbg-molpcba": load_ogbg_molpcba,
        "peptides-func": load_peptides_func,
        "peptides-struct": load_peptides_struct,
        "ogbl-collab": load_ogbl_collab,
    }
