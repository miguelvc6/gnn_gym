from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from gnn_gym.data.loaders import load_dataset
from gnn_gym.models.normal_tree_backedge_gnn import compute_tree_backedge_markers


@dataclass(frozen=True)
class ShortcutResult:
    model: str
    val_ap: float
    test_ap: float
    val_accuracy: float
    test_accuracy: float


def graph_stats(data: Data) -> list[float]:
    num_nodes = int(data.num_nodes)
    undirected_edges = {
        tuple(sorted((int(src), int(dst))))
        for src, dst in data.edge_index.t().tolist()
        if int(src) != int(dst)
    }
    num_edges = len(undirected_edges)
    density = 0.0
    if num_nodes > 1:
        density = (2.0 * num_edges) / float(num_nodes * (num_nodes - 1))
    degrees = torch.zeros(num_nodes, dtype=torch.float)
    adjacency = [set() for _ in range(num_nodes)]
    for src, dst in undirected_edges:
        degrees[src] += 1.0
        degrees[dst] += 1.0
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    triangle_count = 0
    for src in range(num_nodes):
        for dst in adjacency[src]:
            if dst <= src:
                continue
            common = adjacency[src].intersection(adjacency[dst])
            triangle_count += sum(1 for node in common if node > dst)

    markers = compute_tree_backedge_markers(data.edge_index.cpu(), num_nodes)
    active_back = markers.back_edges > 0
    back_spans = markers.span[active_back]
    if back_spans.numel() == 0:
        back_span_mean = 0.0
        back_span_std = 0.0
        back_span_max = 0.0
    else:
        back_span_mean = float(back_spans.mean().item())
        back_span_std = float(back_spans.std(unbiased=False).item())
        back_span_max = float(back_spans.max().item())

    return [
        float(num_nodes),
        float(num_edges),
        density,
        float(degrees.min().item()),
        float(degrees.mean().item()),
        float(degrees.max().item()),
        float(degrees.std(unbiased=False).item()),
        float(triangle_count),
        float(active_back.sum().item() / 2.0),
        back_span_mean,
        back_span_std,
        back_span_max,
    ]


def evaluate_shortcuts() -> list[ShortcutResult]:
    bundle = load_dataset(
        "normal-tree-backedge",
        {
            "dataset": {
                "name": "normal-tree-backedge",
                "variant": "cycle_matching_v4",
                "num_graphs": 120,
                "num_nodes_per_graph": 20,
                "generation_seed": 45678,
            }
        },
    )
    features = np.asarray([graph_stats(graph) for graph in bundle.dataset], dtype=np.float64)
    labels = np.asarray([float(graph.y.item()) for graph in bundle.dataset], dtype=np.float64)
    train_idx = np.asarray(bundle.split_idx["train"], dtype=np.int64)
    val_idx = np.asarray(bundle.split_idx["valid"], dtype=np.int64)
    test_idx = np.asarray(bundle.split_idx["test"], dtype=np.int64)
    estimators = [
        (
            "logistic_graph_stats",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000, random_state=0)),
        ),
        (
            "mlp_graph_stats",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(16,),
                    max_iter=1_000,
                    random_state=0,
                    early_stopping=False,
                ),
            ),
        ),
    ]
    results = []
    for name, estimator in estimators:
        estimator.fit(features[train_idx], labels[train_idx])
        val_scores = estimator.predict_proba(features[val_idx])[:, 1]
        test_scores = estimator.predict_proba(features[test_idx])[:, 1]
        val_pred = (val_scores >= 0.5).astype(np.float64)
        test_pred = (test_scores >= 0.5).astype(np.float64)
        results.append(
            ShortcutResult(
                model=name,
                val_ap=float(average_precision_score(labels[val_idx], val_scores)),
                test_ap=float(average_precision_score(labels[test_idx], test_scores)),
                val_accuracy=float(accuracy_score(labels[val_idx], val_pred)),
                test_accuracy=float(accuracy_score(labels[test_idx], test_pred)),
            )
        )
    return results


if __name__ == "__main__":
    for result in evaluate_shortcuts():
        print(
            "\t".join(
                [
                    result.model,
                    f"val_ap={result.val_ap:.4f}",
                    f"test_ap={result.test_ap:.4f}",
                    f"val_acc={result.val_accuracy:.4f}",
                    f"test_acc={result.test_accuracy:.4f}",
                ]
            )
        )
