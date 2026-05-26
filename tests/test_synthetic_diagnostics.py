from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from gnn_gym.evaluation.synthetic_diagnostics import (
    audit_batch_composition_invariance,
    audit_cache_key_stability,
    audit_capped_enumeration_tie_stress,
    audit_transform_invariance,
    canonical_edge_key,
    dataset_graph_statistics,
    evaluate_shortcut_controls,
    graph_statistics_features,
    metric_invalidation_record,
    permute_edge_order,
    random_node_relabel,
    required_shortcut_feature_sets,
)


class SumGraphModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_attr
        if batch is None:
            return x.sum(dim=0, keepdim=True)
        out = x.new_zeros((int(batch.max().item()) + 1, x.size(1)))
        out.index_add_(0, batch, x)
        return out


def _triangle(label: float = 1.0) -> Data:
    return Data(
        x=torch.ones((3, 2)),
        edge_index=torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]]),
        y=torch.tensor([[label]]),
    )


def _path(label: float = 0.0) -> Data:
    return Data(
        x=torch.ones((4, 2)),
        edge_index=torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]]),
        y=torch.tensor([[label]]),
    )


def test_graph_statistics_features_count_basic_structure() -> None:
    features, names = graph_statistics_features(_triangle())

    values = dict(zip(names, features, strict=True))
    assert values["num_nodes"] == 3.0
    assert values["num_edges"] == 3.0
    assert values["triangles"] == 1.0


def test_shortcut_controls_include_required_feature_groups() -> None:
    dataset = [_triangle(1.0), _path(0.0), _triangle(1.0), _path(0.0), _triangle(1.0), _path(0.0)]
    graph_stats, _ = dataset_graph_statistics(dataset)
    labels = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float).numpy()
    feature_sets = required_shortcut_feature_sets(
        graph_statistics=graph_stats,
        candidate_counts={"detector": graph_stats[:, :1]},
        candidate_histograms={"detector": graph_stats[:, -2:]},
        same_features={"candidate_exact": graph_stats[:, :3]},
    )

    results = evaluate_shortcut_controls(
        feature_sets,
        labels,
        {
            "train": torch.tensor([0, 1, 2, 3]),
            "valid": torch.tensor([4]),
            "test": torch.tensor([5]),
        },
        max_iter=200,
    )

    result_keys = {(result.feature_set, result.estimator) for result in results}
    assert ("class_prevalence_random_ap", "class_prevalence") in result_keys
    assert ("graph_statistics", "logistic") in result_keys
    assert ("candidate_count_detector", "mlp") in result_keys
    assert ("candidate_histogram_detector", "logistic") in result_keys
    assert ("same_feature_candidate_exact", "mlp") in result_keys
    assert ("same_capacity_merged_control", "mlp") in result_keys


def test_relabel_and_edge_order_transforms_preserve_graph_shape() -> None:
    data = _triangle()
    generator = torch.Generator().manual_seed(0)

    relabeled = random_node_relabel(data, generator=generator)
    edge_permuted = permute_edge_order(data, generator=generator)

    assert relabeled.x.shape == data.x.shape
    assert relabeled.edge_index.shape == data.edge_index.shape
    assert edge_permuted.edge_index.shape == data.edge_index.shape


def test_transform_and_batch_audits_pass_for_sum_model() -> None:
    model = SumGraphModel()
    data = _triangle()

    edge_order_audit = audit_transform_invariance(
        model,
        data,
        lambda graph, generator: permute_edge_order(graph, generator=generator),
        num_trials=3,
        audit_name="edge_order_invariance",
    )
    batch_audit = audit_batch_composition_invariance(model, data, [_path()], target_position=0)

    assert edge_order_audit.passed
    assert batch_audit.passed


def test_canonical_cache_key_is_edge_order_stable() -> None:
    data = _triangle()
    audit = audit_cache_key_stability(
        lambda graph: canonical_edge_key(graph.edge_index, num_nodes=int(graph.num_nodes)),
        data,
    )

    assert audit["original_equals_edge_permuted"]


def test_capped_enumeration_tie_stress_reports_instability() -> None:
    data = _triangle()

    def label_sensitive_selector(graph: Data) -> tuple[int, int]:
        return tuple(graph.edge_index[:, 0].tolist())

    audit = audit_capped_enumeration_tie_stress(label_sensitive_selector, data, num_relabels=3)

    assert audit["num_relabels"] == 3
    assert "stable_under_all_relabels" in audit


def test_metric_invalidation_record_lists_replacements() -> None:
    record = metric_invalidation_record(
        invalidated_at="2026-05-26",
        reason="correctness-changing cache fix",
        invalidated_files=["old.json"],
        replacement_files=["new.json"],
    )

    assert record["invalidated_files"] == ["old.json"]
    assert record["replacement_files"] == ["new.json"]
