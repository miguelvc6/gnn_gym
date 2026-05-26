from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data


@dataclass(frozen=True)
class ShortcutResult:
    feature_set: str
    estimator: str
    num_features: int
    val_ap: float
    test_ap: float
    val_accuracy: float | None = None
    test_accuracy: float | None = None
    note: str = ""


@dataclass(frozen=True)
class PredictionAuditResult:
    audit: str
    max_abs_diff: float
    mean_abs_diff: float
    passed: bool
    num_trials: int
    tolerance: float
    note: str = ""


def graph_statistics_features(data: Data) -> tuple[np.ndarray, list[str]]:
    edge_pairs = _undirected_edges(data.edge_index)
    num_nodes = int(data.num_nodes)
    num_edges = len(edge_pairs)
    degree = np.zeros(num_nodes, dtype=np.float64)
    adjacency = [set() for _ in range(num_nodes)]
    for src, dst in edge_pairs:
        if src == dst:
            continue
        degree[src] += 1.0
        degree[dst] += 1.0
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    possible_edges = max(num_nodes * (num_nodes - 1) / 2.0, 1.0)
    triangle_count = _triangle_count(adjacency)
    four_cycle_edge_support = sum(
        _four_cycle_support(src, dst, adjacency) for src, dst in edge_pairs if src != dst
    )
    features = np.asarray(
        [
            float(num_nodes),
            float(num_edges),
            float(num_edges / possible_edges),
            float(degree.min()) if degree.size else 0.0,
            float(degree.mean()) if degree.size else 0.0,
            float(degree.max()) if degree.size else 0.0,
            float(degree.std()) if degree.size else 0.0,
            float(triangle_count),
            float(four_cycle_edge_support),
        ],
        dtype=np.float64,
    )
    names = [
        "num_nodes",
        "num_edges",
        "density",
        "degree_min",
        "degree_mean",
        "degree_max",
        "degree_std",
        "triangles",
        "four_cycle_edge_support_total",
    ]
    return features, names


def dataset_graph_statistics(dataset: Sequence[Data]) -> tuple[np.ndarray, list[str]]:
    rows = []
    names: list[str] | None = None
    for graph in dataset:
        features, feature_names = graph_statistics_features(graph)
        rows.append(features)
        names = feature_names
    if names is None:
        return np.empty((0, 0), dtype=np.float64), []
    return np.vstack(rows), names


def labels_from_graph_dataset(dataset: Sequence[Data]) -> np.ndarray:
    return np.asarray([float(graph.y.view(-1)[0].item()) for graph in dataset], dtype=np.float64)


def split_indices(split_idx: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "train": _to_numpy_index(split_idx["train"]),
        "valid": _to_numpy_index(split_idx["valid"]),
        "test": _to_numpy_index(split_idx["test"]),
    }


def evaluate_shortcut_controls(
    feature_sets: Mapping[str, np.ndarray],
    labels: np.ndarray,
    splits: Mapping[str, np.ndarray],
    *,
    random_state: int = 0,
    mlp_hidden: tuple[int, ...] = (16,),
    max_iter: int = 1_000,
) -> list[ShortcutResult]:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    train_idx = np.asarray(splits["train"], dtype=np.int64)
    val_idx = np.asarray(splits["valid"], dtype=np.int64)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    results = [
        _prevalence_result("class_prevalence_random_ap", labels, val_idx, test_idx),
    ]
    estimators: list[tuple[str, Callable[[], Any]]] = [
        (
            "logistic",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=max_iter, random_state=random_state),
            ),
        ),
        (
            "mlp",
            lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=mlp_hidden,
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=False,
                ),
            ),
        ),
    ]
    for feature_set, raw_features in feature_sets.items():
        features = _as_feature_matrix(raw_features)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{feature_set} has {features.shape[0]} rows for {labels.shape[0]} labels"
            )
        for estimator_name, estimator_factory in estimators:
            results.append(
                _fit_feature_control(
                    feature_set,
                    estimator_name,
                    estimator_factory,
                    features,
                    labels,
                    train_idx,
                    val_idx,
                    test_idx,
                )
            )
    return results


def required_shortcut_feature_sets(
    *,
    graph_statistics: np.ndarray,
    candidate_counts: Mapping[str, np.ndarray] | None = None,
    candidate_histograms: Mapping[str, np.ndarray] | None = None,
    same_features: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    feature_sets: dict[str, np.ndarray] = {"graph_statistics": _as_feature_matrix(graph_statistics)}
    for name, features in (candidate_counts or {}).items():
        feature_sets[f"candidate_count_{name}"] = _as_feature_matrix(features)
    for name, features in (candidate_histograms or {}).items():
        feature_sets[f"candidate_histogram_{name}"] = _as_feature_matrix(features)
    for name, features in (same_features or {}).items():
        feature_sets[f"same_feature_{name}"] = _as_feature_matrix(features)

    merged_parts = [feature_sets[name] for name in sorted(feature_sets)]
    if merged_parts:
        feature_sets["same_capacity_merged_control"] = np.concatenate(merged_parts, axis=1)
    return feature_sets


def shortcut_results_to_payload(
    *,
    results: Sequence[ShortcutResult],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "metadata": dict(metadata),
        "results": [asdict(result) for result in results],
        "claim_gate": {
            "rule": (
                "Synthetic candidate claims require validation AP above class prevalence, "
                "graph-statistics controls, exact candidate detector controls, same-feature "
                "controls, and the same-capacity merged neural control."
            ),
            "selection_metric": "val_ap",
            "test_metric_policy": "held-out reporting only",
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def random_node_relabel(data: Data, *, generator: torch.Generator | None = None) -> Data:
    num_nodes = int(data.num_nodes)
    old_to_new = torch.randperm(num_nodes, generator=generator)
    new_x = data.x.new_empty(data.x.shape)
    new_x[old_to_new] = data.x
    relabeled = data.clone()
    relabeled.x = new_x
    relabeled.edge_index = old_to_new[data.edge_index]
    if hasattr(data, "batch") and data.batch is not None:
        new_batch = data.batch.new_empty(data.batch.shape)
        new_batch[old_to_new] = data.batch
        relabeled.batch = new_batch
    return relabeled


def permute_edge_order(data: Data, *, generator: torch.Generator | None = None) -> Data:
    permuted = data.clone()
    edge_count = int(data.edge_index.size(1))
    order = torch.randperm(edge_count, generator=generator)
    permuted.edge_index = data.edge_index[:, order]
    if getattr(data, "edge_attr", None) is not None:
        permuted.edge_attr = data.edge_attr[order]
    return permuted


def audit_transform_invariance(
    model: torch.nn.Module,
    data: Data,
    transform: Callable[[Data, torch.Generator], Data],
    *,
    align: Callable[[torch.Tensor, Data, Data], torch.Tensor] | None = None,
    num_trials: int = 8,
    seed: int = 0,
    tolerance: float = 1e-5,
    audit_name: str = "transform_invariance",
) -> PredictionAuditResult:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        reference = _predict_data(model, data)
        diffs = []
        for _ in range(num_trials):
            transformed = transform(data, generator)
            prediction = _predict_data(model, transformed)
            if align is not None:
                prediction = align(prediction, data, transformed)
            diffs.append((prediction - reference).abs().detach().cpu().view(-1))
    if diffs:
        diff = torch.cat(diffs)
        max_abs_diff = float(diff.max().item())
        mean_abs_diff = float(diff.mean().item())
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
    return PredictionAuditResult(
        audit=audit_name,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        passed=max_abs_diff <= tolerance,
        num_trials=num_trials,
        tolerance=tolerance,
    )


def audit_batch_composition_invariance(
    model: torch.nn.Module,
    target: Data,
    context: Sequence[Data],
    *,
    target_position: int = 0,
    tolerance: float = 1e-5,
) -> PredictionAuditResult:
    model.eval()
    graphs = list(context)
    graphs.insert(target_position, target)
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        alone = _predict_data(model, target)
        together = model(
            batch.x,
            batch.edge_index,
            batch=getattr(batch, "batch", None),
            edge_attr=getattr(batch, "edge_attr", None),
        )
    if alone.shape[0] == int(target.num_nodes):
        graph_id = target_position
        prediction = together[batch.batch == graph_id]
    else:
        prediction = together[target_position : target_position + 1]
    diff = (prediction - alone).abs().detach().cpu().view(-1)
    max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
    return PredictionAuditResult(
        audit="batch_composition_invariance",
        max_abs_diff=max_abs_diff,
        mean_abs_diff=float(diff.mean().item()) if diff.numel() else 0.0,
        passed=max_abs_diff <= tolerance,
        num_trials=1,
        tolerance=tolerance,
    )


def canonical_edge_key(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    batch: torch.Tensor | None = None,
    undirected: bool = True,
) -> tuple[Any, ...]:
    edge_index = edge_index.detach().cpu().to(dtype=torch.long)
    if undirected:
        left = torch.minimum(edge_index[0], edge_index[1])
        right = torch.maximum(edge_index[0], edge_index[1])
        edge_index = torch.stack([left, right], dim=0)
    edges = sorted((int(src), int(dst)) for src, dst in edge_index.t().tolist())
    batch_key = (
        tuple(int(value) for value in batch.detach().cpu().tolist())
        if batch is not None
        else None
    )
    return (int(num_nodes), tuple(edges), batch_key)


def audit_cache_key_stability(
    key_fn: Callable[[Data], Any],
    data: Data,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    original = key_fn(data)
    edge_permuted = key_fn(permute_edge_order(data, generator=generator))
    relabeled = random_node_relabel(data, generator=generator)
    return {
        "original_equals_edge_permuted": original == edge_permuted,
        "original_equals_relabel": original == key_fn(relabeled),
        "note": (
            "Canonical graph-level caches should usually be edge-order stable. Relabel stability "
            "is required only when the cached object claims canonical graph isomorphism behavior."
        ),
    }


def audit_capped_enumeration_tie_stress(
    enumerate_fn: Callable[[Data], Any],
    data: Data,
    *,
    num_relabels: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    reference = enumerate_fn(data)
    changed = 0
    samples = []
    for _ in range(num_relabels):
        relabeled = random_node_relabel(data, generator=generator)
        value = enumerate_fn(relabeled)
        is_changed = value != reference
        changed += int(is_changed)
        samples.append({"changed": bool(is_changed), "value": repr(value)})
    return {
        "num_relabels": num_relabels,
        "changed_count": changed,
        "stable_under_all_relabels": changed == 0,
        "reference": repr(reference),
        "samples": samples,
    }


def metric_invalidation_record(
    *,
    invalidated_at: str,
    reason: str,
    invalidated_files: Sequence[str],
    replacement_files: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "invalidated_at": invalidated_at,
        "reason": reason,
        "invalidated_files": list(invalidated_files),
        "replacement_files": list(replacement_files or []),
        "policy": (
            "Metrics produced before correctness-changing fixes are not evidence for architecture "
            "claims. Use replacement files or rerun the experiment."
        ),
    }


def _predict_data(model: torch.nn.Module, data: Data) -> torch.Tensor:
    return model(
        data.x,
        data.edge_index,
        batch=getattr(data, "batch", None),
        edge_attr=getattr(data, "edge_attr", None),
    )


def _prevalence_result(
    feature_set: str,
    labels: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> ShortcutResult:
    return ShortcutResult(
        feature_set=feature_set,
        estimator="class_prevalence",
        num_features=0,
        val_ap=float(labels[val_idx].mean()) if val_idx.size else float("nan"),
        test_ap=float(labels[test_idx].mean()) if test_idx.size else float("nan"),
        note="Expected AP of random scores equals positive-class prevalence.",
    )


def _fit_feature_control(
    feature_set: str,
    estimator_name: str,
    estimator_factory: Callable[[], Any],
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> ShortcutResult:
    estimator = estimator_factory()
    estimator.fit(features[train_idx], labels[train_idx])
    val_scores = estimator.predict_proba(features[val_idx])[:, 1]
    test_scores = estimator.predict_proba(features[test_idx])[:, 1]
    val_pred = (val_scores >= 0.5).astype(np.float64)
    test_pred = (test_scores >= 0.5).astype(np.float64)
    return ShortcutResult(
        feature_set=feature_set,
        estimator=estimator_name,
        num_features=int(features.shape[1]),
        val_ap=float(average_precision_score(labels[val_idx], val_scores)),
        test_ap=float(average_precision_score(labels[test_idx], test_scores)),
        val_accuracy=float(accuracy_score(labels[val_idx], val_pred)),
        test_accuracy=float(accuracy_score(labels[test_idx], test_pred)),
    )


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        return features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(f"Expected 1D or 2D feature matrix, got shape {features.shape}")
    return features


def _to_numpy_index(index: Any) -> np.ndarray:
    if isinstance(index, torch.Tensor):
        return index.detach().cpu().numpy().astype(np.int64)
    return np.asarray(index, dtype=np.int64)


def _undirected_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    pairs = set()
    for src_raw, dst_raw in edge_index.detach().cpu().to(dtype=torch.long).t().tolist():
        src = int(src_raw)
        dst = int(dst_raw)
        pairs.add((src, dst) if src <= dst else (dst, src))
    return pairs


def _triangle_count(adjacency: Sequence[set[int]]) -> int:
    count = 0
    for src, neighbors in enumerate(adjacency):
        for mid in neighbors:
            if mid <= src:
                continue
            for dst in adjacency[mid]:
                if dst > mid and dst in neighbors:
                    count += 1
    return count


def _four_cycle_support(src: int, dst: int, adjacency: Sequence[set[int]]) -> int:
    left = adjacency[src] - {dst}
    right = adjacency[dst] - {src}
    return sum(len(adjacency[left_node].intersection(right)) for left_node in left)
