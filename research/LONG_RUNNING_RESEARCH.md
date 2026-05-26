# Long-Running Research Protocol

This protocol is for multi-session GNN architecture research. The goal is to accumulate reliable
evidence about mechanisms, not to chase incidental leaderboard movement.

## Evaluation Protocol For Architecture Claims

1. Use `val_metric` for architecture and hyperparameter selection.
2. Record `test_metric`, but never select architectures or hyperparameters using test performance.
3. A seed-0 result is only a screening signal.
4. Any promising seed-0 architecture/config must be confirmed across seeds `[0, 1, 2]`.
5. Architecture claims must be based on config-level aggregation, not model-level mixed-config
   averages.
6. The config-level grouping key is `architecture_config_hash`.
7. Do not claim a model beats a baseline unless the exact confirmed config beats the baseline mean
   validation metric under the same budget.
8. Use task-appropriate baselines:
   - Cora/PubMed node classification: current confirmed `gpr_gnn`.
   - Older node comparisons: GAT, APPNP, JK-GCN, and GATv2 when relevant.
   - MolHIV graph prediction: current GIN/GCN/GAT/MLP baselines; add edge-aware baselines after
     edge-attribute-capable models are implemented.
9. Test metrics are held-out reporting only.
10. Toy datasets are crash checks only, never architecture evidence.

The raw aggregate table records run-level rows. Config evidence should come from
`*_by_config_mean_std.csv` exports, grouped by `architecture_config_hash`. Model-level summaries are
mixed-config diagnostics only and must not be used for selecting a winning architecture/config.

## Novelty And Scientific Insight Standard

Simple hyperparameter tuning is useful engineering, not a novel architecture. Reimplementing known
methods such as APPNP, GPR-GNN, GATv2, GCNII, GraphSAGE, GINE, GPS, or Graphormer should be labeled
as baseline work unless the experiment adds a genuinely new mechanism. Hybridizing two known methods
is not automatically novel; it needs a specific mechanism and a falsifiable claim.

Each architecture idea should state:

```text
Scientific hypothesis:
Mechanism:
Why this is not just a known baseline:
Closest known related architectures:
Expected insight if it succeeds:
Expected insight if it fails:
Target task family:
Primary baseline to beat:
Minimal falsifying experiment:
Confirmation protocol:
Complexity/runtime risk:
Implementation boundary:
```

A failed architecture can still be scientifically useful if it tests a clear hypothesis and records
what was learned. The long-running objective is to build research memory over months: ideas,
evidence, negative results, refined hypotheses, and better baselines.

## Long-Running Workflow

1. Read this protocol, `research/PROGRAM.md`, `research/INSIGHTS.md`, and the active scratchpad.
2. Pick or add one architecture idea with a scientific hypothesis.
3. Check the closest known baselines before calling the idea novel.
4. Implement the smallest bounded version.
5. Run toy crash checks.
6. Run a seed-0 fast screen.
7. Confirm only promising configs over seeds `[0, 1, 2]`.
8. Aggregate by `architecture_config_hash`.
9. Promote only durable conclusions to `research/INSIGHTS.md`.
10. Keep negative results if they clarify the hypothesis space.
11. Periodically add new ideas, but avoid repeating discarded mechanisms without a new reason.

## Warnings

Future agents must not:

- select from `test_metric`
- use toy metrics as evidence
- claim novelty for known architectures
- claim improvement from a mixed-config mean/std table
- edit trainers/evaluators to favor a candidate architecture unless the idea explicitly requires a
  new training objective
- compare across incompatible metrics as if they were one global score
- hide negative results

## Synthetic Diagnostic Shortcut And Audit Protocol

Synthetic graph diagnostics are useful only when the intended mechanism is separated from cheaper
shortcuts. Before using a synthetic metric as architecture evidence, run the reusable suite in
`gnn_gym.evaluation.synthetic_diagnostics` or an idea-specific adapter built on it.

Required shortcut controls:

- class prevalence / random AP
- graph statistics logistic and MLP
- candidate detector counts
- candidate detector histograms
- same-feature logistic and MLP using exactly the candidate's exposed detector features
- same-capacity merged neural control over all shortcut and detector features
- an ablation that removes the proposed decomposition while keeping parameters comparable

Required graph audits:

- random node relabeling
- edge-order invariance
- batch-composition invariance
- capped-enumeration / tie-stress tests for any bounded selector
- cache-key stability, including edge-order stability for graph-level caches
- explicit invalidation records for metrics produced before correctness-changing fixes

Do not claim improvement from synthetic metrics unless the candidate beats the exact shortcut
controls and same-capacity controls by validation metric. Test metrics remain held-out reporting
only. If a shortcut or audit is not applicable, the experiment note must say why.

## Infrastructure Notes

- `config_hash` identifies a resolved run and can vary by seed.
- `architecture_config_hash` identifies the seed-independent architecture/training/dataset config.
- Same model/dataset/hyperparameters across different seeds should share `architecture_config_hash`.
- Different architecture or training hyperparameters should produce different
  `architecture_config_hash` values.
- `neighbor_node` should use PyG `NeighborLoader` for scalable runs when the optional sampling
  backend is available; the fallback path is for smoke and compatibility only.
- Graph and link trainers pass `edge_attr` when present. Existing models may ignore it, but future
  edge-aware models can accept `edge_attr` in `forward`.
