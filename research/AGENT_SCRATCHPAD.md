# Agent Scratchpad

This file is for working memory across research sessions: what was tried, what happened, what is worth trying next, and what to avoid repeating. It is allowed to be messy, but every entry should be dated and actionable.

Do not use this file for final claims. Promote durable conclusions to `research/INSIGHTS.md`.

## Next parallel round candidates

- Continue:
  - `CycleCutGNN-lite` only as a structural-feature diagnostic. Do not repeat the current split
    claim unless split cut/cycle channels beat cycle-only and merged edge-state MPNN controls by
    validation metric.
  - `DualShadowGNN` only as pseudo-face/motif diagnostic support. Do not claim planar-dual novelty;
    next task must match pseudo-face count, cycle-length histogram, and collapsed pseudo-face
    statistics.
  - `ObstructionTokenGNN` as detector/shortcut-audit infrastructure only, not as a token
    message-passing win.
- Revise:
  - `BagAutomatonGNN` needs post-fix retraining before any evidence use; old high seed-0 metrics are
    archived because they came from a node-id-sensitive implementation.
  - `ChordlessCycleMemoryGNN` needs a cleaner diagnostic that controls chordless-count, triangle,
    degree, and prevalence shortcuts; current capped enumeration is fixed but the architecture
    result is negative/inconclusive.
- Archive:
  - `BranchSetGNN`, `ListColorGNN`, `ClassFlowGNN`, and `RegularPatchGNN` should not be repeated
    unless their specific blockers are fixed with new diagnostics.
  - Do not repeat TreePack learned view gating unless there is a new canonical or sharply selective
    tree-witness mechanism.
- New ideas:
  - Matched-histogram pseudo-face arrangement diagnostic.
  - Separator-bottleneck task with graph-stat, bag-stat, and detector-count controls included from
    the start.
  - Small exact cycle/cut or Hodge-projection diagnostic compared directly to merged edge-state
    MPNN.
  - Canonical witness-selection task for trees/cycles compared against single-witness controls.
- Infrastructure needed:
  - Standard shortcut suite: prevalence, graph stats, candidate detector counts/histograms,
    same-feature logistic/MLP, and same-capacity neural ablations.
  - Standard graph-model audit suite: relabeling, edge-order, batch-composition, capped-enumeration,
    cache-bound, and tie-stress tests.
  - Policy that pre-fix metrics are invalid evidence after correctness-changing implementation
    changes.
  - Precomputed structural transforms before trying expensive detectors on molecule or larger graph
    benchmarks.

## Current Benchmark Discipline

- Keep the harness fixed while evaluating an architecture idea.
- Prefer `toy-node`, `toy-graph`, and `toy-link` for crash checks.
- Toy metrics are never architecture evidence.
- Use Cora/PubMed as the first real node-classification checks.
- Use `ogbg-molhiv` as the first real graph-prediction check.
- Avoid running `ogbn-products`, Peptides, or `ogbl-collab` during casual iteration unless the user
  explicitly wants a long run.
- Select configs by validation metric only; record test metrics for held-out reporting.
- Treat seed `0` as screening only and confirm promising configs across seeds `[0, 1, 2]`.
- Use config-level aggregation by `architecture_config_hash` for claims. Model-level mixed-config
  means are diagnostics only.
- Every architecture idea should state a scientific hypothesis, closest known baselines, and a
  minimal falsifying experiment before implementation.

## Experiment Log Template

```md
## YYYY-MM-DD - Experiment Name

Branch/commit:
Files changed:
Command:
Result:
Keep/discard:
Notes:
Next:
```

## 2026-05-26 - Research Infrastructure Hardening Note

Branch/commit: local uncommitted work
Files changed: benchmark plumbing and research protocol Markdown
Command:

```bash
uv run ruff check .
uv run pytest
```

Result:

- Added seed-independent `architecture_config_hash` protocol for config-level evidence.
- Research claims should use `*_by_config_mean_std.csv`; `*_by_model_mean_std.csv` is mixed-config
  diagnostic output only.
- Edge attributes now belong in the model call path for future edge-aware graph/link models.
- `neighbor_node` is expected to use PyG `NeighborLoader` when available; fallback is for smoke and
  compatibility, not serious `ogbn-products` evidence.

Keep/discard: keep as protocol update

Next:

- Re-export old aggregate tables with the new export command before using them for future claims.
- Add edge-aware graph baselines before making strong MolHIV architecture claims.

## Tried

## 2026-05-26 - TreePackGNN-lite Diagnostic Cycle

Branch/commit: local uncommitted work on HEAD `a25be44c5e94c63860b057461dcafd4b4e529eee`
Files changed: `src/gnn_gym/models/tree_pack_gnn.py`, `configs/models/tree_pack_gnn.yaml`,
`tests/test_tree_pack_gnn.py`, `tests/test_model_shapes.py`, `tests/test_registry.py`,
`research/experiments/tree_pack_gnn.md`, `research/AGENT_SCRATCHPAD.md`,
`research/INSIGHTS.md`, aggregate tables and TreePack audit JSON files under `results/tables/`
Command:

```bash
uv run gnngym train --model tree_pack_gnn --dataset toy-graph --seed 0 --override training.max_epochs=2 --override training.patience=2 --override model.hidden_channels=16 --override model.head_hidden_channels=16 --override model.num_trees=4 --override trainer.batch_size=8
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed {0,1,2} --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16
UV_CACHE_DIR=/tmp/uv-cache uv run python research/experiments/normal_tree_backedge_permutation_audit.py --run-dir <tree_pack_run> --num-relabels 16 --seed 123 --batch-size 16 --output results/tables/tree_pack_gnn_permutation_audit_seed*_wide.json
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- Toy-graph crash check passed.
- TreePack hidden-32 seed-0 screen was not promising: val/test AP `0.5856/0.5479`.
- TreePack hidden-64 seed-0 screen was promising: val/test AP `0.7305/0.6629`.
- Confirmed hidden-64 config `architecture_config_hash=8ec4a7ff` across seeds `[0,1,2]`:
  validation AP `0.7462 +/- 0.0914`, test AP `0.5838 +/- 0.0815`.
- This beats single-order NormalTreeBackedgeGNN (`0.6635/0.5516`) and naive four-order DFS
  averaging (`0.6490/0.5626`) by mean validation/test AP.
- Permutation audit max prediction ranges were `0.0002`, `0.0013`, and `0.0027`, much lower than
  single-order NormalTreeBackedgeGNN ranges up to about `0.71`.

Keep/discard: keep as positive synthetic diagnostic evidence, not a real benchmark claim.

Notes:

- TreePack appears to fix the relabeling instability that made the single DFS witness unsuitable.
- The seed-2 held-out test AP was only `0.5001`, so generalization is still fragile.
- Runtime is higher than single-order NormalTreeBackedgeGNN (`31.12s` mean versus `14.34s`) but
  lower than the old four-order DFS average (`75.73s`).

Next:

- Continue TreePack for one bounded ablation cycle before moving to `CycleCutGNN-lite`.
- Run residual-only/full-graph, individual tree-view, mean-pooling-vs-gating, and view-use
  diagnostics. Stop if the residual or one tree type explains the gain.

## 2026-05-26 - TreePackGNN-lite Ablation Cycle

Branch/commit: local uncommitted work on HEAD `a25be44c5e94c63860b057461dcafd4b4e529eee`
Files changed: `src/gnn_gym/models/tree_pack_gnn.py`, `configs/models/tree_pack_gnn.yaml`,
`tests/test_tree_pack_gnn.py`, `tests/test_model_shapes.py`,
`research/experiments/tree_pack_gate_diagnostics.py`, `research/experiments/tree_pack_gnn.md`,
`research/AGENT_SCRATCHPAD.md`, `research/INSIGHTS.md`, aggregate tables and diagnostic JSON files
under `results/tables/`
Command:

```bash
uv run pytest tests/test_tree_pack_gnn.py tests/test_model_shapes.py::test_model_output_shape
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16 --override model.use_tree_channel=false
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16 --override model.use_graph_channel=false
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16 --override model.tree_pooling=mean
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=1 --override model.tree_start_idx={0,1,2,3} --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed {1,2} --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=1 --override model.tree_start_idx={0,3} --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16
UV_CACHE_DIR=/tmp/uv-cache uv run python research/experiments/tree_pack_gate_diagnostics.py --run-dir <tree_pack_run> --split test --output results/tables/tree_pack_gnn_gate_diagnostics_seed*.json
UV_CACHE_DIR=/tmp/uv-cache uv run python research/experiments/normal_tree_backedge_permutation_audit.py --run-dir <single_tree0_run> --num-relabels 16 --seed 123 --batch-size 16 --output results/tables/tree_pack_gnn_single_tree0_permutation_audit_seed*.json
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- Residual-only/full-graph control collapsed to chance on seed 0: val/test AP `0.4167/0.4167`.
- Tree-only 4-tree gated pack retained signal but underperformed the full pack on seed 0:
  val/test AP `0.6339/0.5992`.
- Four-tree mean pooling was worse than the learned gate on seed-0 validation and poor on held-out
  test: val/test AP `0.7089/0.4259`.
- Single-tree seed-0 controls found that high-degree BFS (`tree_start_idx=0`) reached
  val/test AP `0.7540/0.6795`, deterministic DFS reached `0.7142/0.6171`, farthest/low-degree BFS
  reached `0.5218/0.5285`, and low-overlap/low-degree BFS reached `0.7947/0.6364`.
- Confirmed high-degree BFS single-tree control `architecture_config_hash=098eb1ec` across seeds
  `[0,1,2]`: validation AP `0.7493 +/- 0.0113`, test AP `0.6812 +/- 0.0185`, mean train time
  `16.21s`.
- Confirmed low-overlap/low-degree BFS single-tree control `architecture_config_hash=fe4388ea`
  across seeds `[0,1,2]`: validation AP `0.7176 +/- 0.0881`, test AP `0.6989 +/- 0.0632`, mean
  train time `14.50s`.
- Full 4-tree gated pack `architecture_config_hash=8ec4a7ff` remains at validation AP
  `0.7462 +/- 0.0914`, test AP `0.5838 +/- 0.0815`, mean train time `31.12s`.
- Gate diagnostics were mostly near-uniform; layer-2 gate entropy was `1.3179`, `1.3863`, and
  `1.2175` for seeds `0`, `1`, and `2` versus maximum four-view entropy `1.3863`.
- High-degree BFS single-tree relabeling max prediction ranges were `0.0113`, `0.0068`, and
  `0.0141`: much better than old single-order NormalTreeBackedgeGNN, but less stable than the
  full TreePack max range of `0.0027`.

Keep/discard: discard the current learned tree-pack/gating claim as not supported by ablations.
Keep the high-degree BFS single-tree result as a strong synthetic control.

Notes:

- The ordinary graph residual is not carrying the result.
- The current learned gate does not clearly select views; it often behaves close to mean pooling.
- A single deterministic BFS witness explains or exceeds the full pack on confirmed validation/test
  while being faster and lower variance.

Next:

- Move to `CycleCutGNN-lite`.
- Revisit TreePack only with a genuinely new mechanism for canonicalizing or sharply selecting tree
  views; do not keep adding more unweighted or weakly gated tree witnesses.

## 2026-05-26 - NormalTreeBackedgeGNN Edge-Role And Relabeling Audit

Branch/commit: local uncommitted work on HEAD `a25be44c5e94c63860b057461dcafd4b4e529eee`
Files changed: `src/gnn_gym/models/normal_tree_backedge_gnn.py`,
`configs/models/normal_tree_backedge_gnn.yaml`, `tests/test_normal_tree_backedge_gnn.py`,
`research/experiments/normal_tree_backedge_permutation_audit.py`,
`research/experiments/normal_tree_backedge_gnn.md`, `research/AGENT_SCRATCHPAD.md`,
`research/INSIGHTS.md`, and aggregate/diagnostic tables under `results/tables/`
Command:

```bash
uv run pytest tests/test_normal_tree_backedge_gnn.py tests/test_model_shapes.py -q
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed {0,1,2} --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=1 --override model.edge_role_mode={true,collapsed,tree_only,back_only,shuffled} --override model.dfs_order_mode=deterministic --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed {0,1,2} --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=1 --override model.edge_role_mode=true --override model.dfs_order_mode=random --override trainer.batch_size=16
uv run python research/experiments/normal_tree_backedge_permutation_audit.py --run-dir <true-role-run> --num-relabels 16 --seed 123 --batch-size 16
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
uv run ruff check .
```

Result:

- `cycle_matching_v4` prevalence is `0.4167` on validation and test, so random AP is about
  `0.4167`.
- True edge roles: val AP `0.7292 +/- 0.1321`, test AP `0.5198 +/- 0.0428`.
- Collapsed roles: val AP `0.6549 +/- 0.0757`, test AP `0.3491 +/- 0.0418`.
- Tree-only: val AP `0.6517 +/- 0.1046`, test AP `0.4856 +/- 0.0802`.
- Back-only: val AP `0.6155 +/- 0.0600`, test AP `0.5535 +/- 0.0128`.
- Shuffled roles: val AP `0.6395 +/- 0.0608`, test AP `0.4033 +/- 0.1064`.
- Random DFS order: val AP `0.5702 +/- 0.0081`, test AP `0.4782 +/- 0.0320`.
- Permutation audit over 16 random relabelings of the same test graphs gave relabeled AP means
  `0.5557`, `0.4504`, and `0.5791` for seeds `0`, `1`, and `2`; mean prediction variance across
  seeds was `0.0255`, with max per-graph prediction ranges up to about `0.71`.

Keep/discard:

- Keep the result as mechanism-specific synthetic support on validation: true edge roles beat both
  collapsed and shuffled controls.
- Do not keep the current model as benchmark-ready evidence. Prediction stability under relabeling
  is poor, and depth/span normalization is currently batch-composition sensitive.

Notes:

- The mechanism signal is real enough to distinguish true roles from controls, but it is not a
  graph invariant in the current implementation.
- The random-order control underperformed true deterministic roles, so arbitrary DFS choice is part
  of the learned signal rather than a harmless implementation detail.
- A same-checkpoint audit initially changed AP when batch size changed; this exposed that marker
  depth normalization uses the maximum depth of the batched disconnected union instead of per graph.

Next:

- Stop refining single-order NormalTreeBackedgeGNN for broader claims.
- If this idea is revisited, first fix per-graph normalization and use a canonical or learned
  distribution over DFS/normal trees.
- For the next idea-bank cycle, move to `TreePackGNN`; use `CycleCutGNN-lite` after that if
  TreePack is infeasible or also weak.

## 2026-05-26 - NormalTreeBackedgeGNN Multi-Order DFS Averaging

Branch/commit: local uncommitted work on HEAD `a25be44c5e94c63860b057461dcafd4b4e529eee`
Files changed: `src/gnn_gym/models/normal_tree_backedge_gnn.py`,
`configs/models/normal_tree_backedge_gnn.yaml`, `tests/test_model_shapes.py`,
`tests/test_normal_tree_backedge_gnn.py`, `research/experiments/normal_tree_backedge_gnn.md`,
`research/AGENT_SCRATCHPAD.md`, `research/INSIGHTS.md`, aggregate tables under `results/tables/`
Command:

```bash
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=4 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=4 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=4 --override trainer.batch_size=16
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
uv run ruff check .
uv run pytest
```

Result:

- Added `model.num_tree_orders` support to `NormalTreeBackedgeGNN-lite`.
- The model can now compute several deterministic DFS forests with different root/neighbor orders
  and average shared up/down/back-edge message channels.
- On `cycle_matching_v4`, the 4-order variant (`architecture_config_hash=2fb8a3d0`) reached
  validation AP `0.6490 +/- 0.0864` and test AP `0.5626 +/- 0.2606`.
- The previous single-order variant (`architecture_config_hash=f9449a35`) remains better on mean
  validation AP: `0.6635 +/- 0.0788`, with lower test variance `0.5516 +/- 0.0865`.
- Runtime increased from about `14.34s` mean to `75.73s` mean.
- `uv run ruff check .` passed.
- `uv run pytest` passed: `55 passed, 2 skipped`.

Keep/discard: keep the parameterized implementation, but keep the default YAML at
`num_tree_orders=1`; discard naive four-order averaging as a robustness fix.

Notes:

- Seed 2 had high test AP for the four-order model, but the seed-0/seed-1 test AP values were poor,
  so this is not a stable improvement.
- Naive averaging likely washes out useful traversal-specific span information.

Next:

- Do not keep expanding this line with unweighted order averaging.
- If revisiting NormalTreeBackedgeGNN, try learned order gating or explicit cycle/matching features.
- Otherwise move to `TreePackGNN` or the next idea-bank candidate.

## 2026-05-26 - NormalTreeBackedgeGNN Hardened Diagnostic

Branch/commit: local uncommitted work on HEAD `a25be44c5e94c63860b057461dcafd4b4e529eee`
Files changed: `src/gnn_gym/data/catalog.py`,
`configs/datasets/normal_tree_backedge.yaml`,
`tests/test_normal_tree_backedge_gnn.py`,
`research/experiments/normal_tree_backedge_shortcut_check.py`,
`research/experiments/normal_tree_backedge_gnn.md`,
`research/AGENT_SCRATCHPAD.md`, `research/INSIGHTS.md`, aggregate tables under `results/tables/`
Command:

```bash
uv run python research/experiments/normal_tree_backedge_shortcut_check.py
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model gcn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 ...
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 ...
uv run gnngym train --model gcn --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 ...
uv run gnngym train --model gcn --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 ...
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 ...
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 ...
uv run ruff check .
uv run pytest
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- Hardened `normal-tree-backedge` with `cycle_matching_v4`: 20-node 3-regular graphs built from a
  cycle plus a nonlocal perfect matching, constant node features, random relabeling per graph, and
  labels from low- versus high-crossing chord arrangement.
- Shortcut graph-stat baselines did not solve it:
  - logistic graph stats: val AP `0.5777`, test AP `0.5191`
  - MLP graph stats: val AP `0.6419`, test AP `0.5192`
- Confirmed the diagnostic screen across seeds `[0,1,2]`:
  - `gin`, `architecture_config_hash=42244e3d`: val/test AP `0.4167 +/- 0.0000`
  - `gcn`, `architecture_config_hash=f594dd4a`: val/test AP `0.4167 +/- 0.0000`
  - `normal_tree_backedge_gnn`, `architecture_config_hash=f9449a35`: val AP
    `0.6635 +/- 0.0788`, test AP `0.5516 +/- 0.0865`
- `uv run ruff check .` passed.
- `uv run pytest` passed: `54 passed, 2 skipped`.

Keep/discard: keep as a useful synthetic diagnostic and a mechanism-supporting result; do not claim
general benchmark improvement.

Notes:

- The validation advantage suggests DFS tree/back-edge channels expose signal that constant-feature
  GIN/GCN cannot access on matched 3-regular graphs.
- The held-out test mean is weak and variable, so the current deterministic DFS scaffold may be
  overfitting traversal artifacts from arbitrary random node labels.

Next:

- Try a multi-root or multi-order NormalTreeBackedgeGNN that averages several DFS/normal-tree
  decompositions on `cycle_matching_v4`.
- Only after improving held-out diagnostic AP should this idea move to real graph benchmarks.

## 2026-05-26 - NormalTreeBackedgeGNN-lite And First Back-Edge Diagnostic

Branch/commit: local uncommitted work on HEAD `a25be44c5e94c63860b057461dcafd4b4e529eee`
Files changed: `src/gnn_gym/models/normal_tree_backedge_gnn.py`,
`configs/models/normal_tree_backedge_gnn.yaml`, `src/gnn_gym/data/catalog.py`,
`configs/datasets/normal_tree_backedge.yaml`, `tests/test_model_shapes.py`,
`tests/test_registry.py`, `tests/test_normal_tree_backedge_gnn.py`,
`research/experiments/normal_tree_backedge_gnn.md`, aggregate tables under `results/tables/`
Command:

```bash
uv run gnngym train --model normal_tree_backedge_gnn --dataset toy-graph --seed 0 --override training.max_epochs=2 --override training.patience=2 --override model.hidden_channels=16 --override model.head_hidden_channels=16 --override trainer.batch_size=8
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 0 --override dataset.variant=endpoint_pairing_v2 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=endpoint_pairing_v2 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run ruff check .
uv run pytest
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- Implemented `NormalTreeBackedgeGNN-lite` from the graph-theory idea bank.
- Added `normal-tree-backedge`, a small synthetic graph diagnostic with path backbones and
  short-span versus long-span chord/back-edge pairings.
- Toy-graph crash check passed.
- On the final diagnostic variant, `gin` reached seed-0 validation/test AP `1.0000/1.0000`
  (`architecture_config_hash=bb49e9d0`).
- `normal_tree_backedge_gnn` also reached seed-0 validation/test AP `1.0000/1.0000`
  (`architecture_config_hash=83e3c616`), with more parameters and longer runtime.
- Because the primary baseline solved the diagnostic, no confirmation was run.
- `uv run ruff check .` passed.
- `uv run pytest` passed: `53 passed, 2 skipped`.

Keep/discard: keep implementation and diagnostic as infrastructure, but do not use this diagnostic
as evidence for the architecture.

Notes:

- The first diagnostic attempt also saturated. The final `dataset.variant=endpoint_pairing_v2` runs
  were added to avoid grouping stale code-generation rows under the same config hash.
- The current diagnostic is too easy for GIN despite matched chord endpoints. It still leaks local
  structure through features and small graph size.

Next:

- Either harden the synthetic task with stronger local-neighborhood matching and no endpoint marker,
  or move to `TreePackGNN` for a Cora/PubMed-compatible next idea.
- Do not spend MolHIV or larger graph benchmark time on `NormalTreeBackedgeGNN-lite` until the
  diagnostic distinguishes it from GIN.

## 2026-05-26 - SepBottleneckGNN-lite First Graph-Theory Idea-Bank Run

Branch/commit: local uncommitted work on HEAD `a25be44c5e94c63860b057461dcafd4b4e529eee`
Files changed: `src/gnn_gym/models/sep_bottleneck_gnn.py`,
`configs/models/sep_bottleneck_gnn.yaml`, `tests/test_model_shapes.py`,
`tests/test_registry.py`, `tests/test_sep_bottleneck_gnn.py`,
`research/experiments/sep_bottleneck_gnn.md`, aggregate tables under `results/tables/`
Command:

```bash
uv run gnngym train --model sep_bottleneck_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym train --model sep_bottleneck_gnn --dataset cora --seed 0 --override training.max_epochs=50 --override training.patience=15
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 0 --override training.max_epochs=50 --override training.patience=15
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 0 --override training.max_epochs=200 --override training.patience=50
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 1 --override training.max_epochs=200 --override training.patience=50
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 2 --override training.max_epochs=200 --override training.patience=50
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
uv run ruff check .
uv run pytest
```

Result:

- Implemented `SepBottleneckGNN-lite` from
  `research/original-ideas/graph_theory_original_gnn_ideas.md`.
- Toy-node crash check passed.
- Cora seed-0 fast screen reached validation `0.7580`, below confirmed GPR Cora mean `0.7827`;
  did not confirm.
- PubMed seed-0 fast screen reached validation `0.8080`, so it was confirmed.
- PubMed confirmation for `architecture_config_hash=0f352f9d` across seeds `[0,1,2]` reached
  validation `0.8000 +/- 0.0053` and test `0.7763 +/- 0.0117`, slightly below confirmed GPR
  PubMed validation mean `0.8007`.
- `uv run ruff check .` passed.
- `uv run pytest` passed: `50 passed, 2 skipped`.

Keep/discard: keep code and notes as a bounded negative/near-miss experiment; do not claim
improvement over GPR.

Notes:

- The initial two-layer/dropout-0.5 default was too brittle under patience 15 because validation
  could plateau just long enough to early-stop before late recovery. The final default uses the
  stronger one-layer decoupled propagation family.
- The separator channel restores RNG state after separator-only module initialization, preserving
  the base fallback initialization for seed-controlled comparisons.
- The cheap separator score may be too broad on citation graphs because every articulation-incident
  edge is treated as separator-adjacent.

Next:

- Add a small `separator-bottleneck-node` diagnostic before more citation-network Sep variants.
- If the diagnostic is positive, refine the separator score to distinguish true bridge/cross-block
  edges from ordinary articulation-incident edges.
- If the diagnostic is negative, discard this separator-residual mechanism and move to the next idea
  bank candidate.

## 2026-05-22 - Node Baseline Hyperparameter Search Plan

Branch/commit: local worktree, current HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: `research/AGENT_SCRATCHPAD.md`
Command:

```bash
git status --short
uv run gnngym train --model <model> --dataset toy-node --seed 0 --override training.max_epochs=2
uv run gnngym run-sweep --config <compact per-model sweep config>
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- Existing `results/tables/baseline_all_runs_mean_std.csv` shows GAT as the current strongest node
  baseline by validation accuracy: Cora `0.7313 +/- 0.0110`, PubMed `0.7607 +/- 0.0167`.
- Current GCN baseline is weaker and unstable, especially PubMed: Cora `0.6380 +/- 0.0330`,
  PubMed `0.6507 +/- 0.1307`.
- Current GIN baseline is stronger than GCN but below GAT: Cora `0.6613 +/- 0.0115`,
  PubMed `0.7300 +/- 0.0212`.
- Test metrics are recorded in the baseline tables but will not be used for selecting configs.

Keep/discard: keep plan

Notes:

- First search batch will use seed `0`, Cora/PubMed, `training.max_epochs=50`,
  `training.patience=15`, and the fixed trainers/evaluators.
- Compact GCN batch: include default plus modest wider/deeper settings:
  `hidden_channels in [64, 128]`, `num_layers in [2, 3]`, `dropout=0.2`,
  `training.lr in [0.01, 0.005]`.
- Compact GAT batch: include default heads and test attention dropout removal:
  `hidden_channels in [16, 32]`, `heads=4`, `dropout=0.2`,
  `attention_dropout in [0.0, 0.2]`, `training.lr in [0.01, 0.005]`.
- Compact GIN batch: include default plus wider/deeper settings:
  `hidden_channels in [64, 128]`, `num_layers in [3, 5]`, `dropout=0.2`,
  `eps_trainable=true`, `training.lr in [0.01, 0.005]`.
- Ideas mined from `research/original-ideas/`: the bounded architecture candidates after baseline
  tuning are `revision_gnn` or a smaller residual/recurrent-GIN-style model. Cavity/non-backtracking
  edge-state models are interesting but likely a larger first implementation because they carry
  directed edge state. RIGN should be staged later because its deep supervision and improvement
  losses would otherwise require trainer changes.

Next:

- Run toy-node crash checks for GCN/GAT/GIN under representative first-batch overrides.
- Run the compact sweeps and aggregate/export tables.
- Pick confirmation candidates using validation metric only; confirm across seeds `0, 1, 2` with
  `training.max_epochs=200` and `training.patience=50` before changing any default YAML.

## 2026-05-22 - Node Hyperparameter Search Results And RevisionGNN First Pass

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: `configs/experiments/research_2026_05_22_*.yaml`,
`src/gnn_gym/models/revision_gnn.py`, `configs/models/revision_gnn.yaml`,
`src/gnn_gym/registry.py`, `tests/test_model_shapes.py`, research notes
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gcn_batch1.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gat_batch1.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_batch1.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gcn_batch2.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gat_batch2.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_batch2.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_pubmed_confirm.yaml
uv run gnngym train --model revision_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_revision_gnn_fast.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
uv run ruff check .
uv run pytest
```

Result:

- Aggregated research table now has `185` rows in `results/tables/research_all_runs.csv`.
- GCN: no fast-search config beat the seed-0 baseline. Best Cora fast result was
  `0.6280` validation for `hidden=64,layers=3,dropout=0.7,lr=0.01,weight_decay=0.0001`; best
  PubMed fast result was `0.6160` validation for `hidden=64,layers=2,dropout=0.5,lr=0.01,
  weight_decay=0.0`.
- GAT: no refinement beat the existing seed-0 baseline. Default-like `hidden=32,heads=4,
  dropout=0.2,attention_dropout=0.2,lr=0.01` remained best on Cora at `0.7240` validation; best
  PubMed fast result was `0.7560`, below the existing seed-0 baseline `0.7660`.
- GIN: `hidden=128,layers=3,dropout=0.5,eps_trainable=true,lr=0.01` confirmed on Cora across seeds
  `0,1,2`: validation mean `0.6773` versus baseline `0.6613`. Test mean was `0.6627` versus
  baseline `0.6710`; recorded only as held-out reporting.
- GIN PubMed candidate `hidden=64,layers=5,dropout=0.2,eps_trainable=true,lr=0.01` did not confirm:
  validation mean `0.7147` versus baseline `0.7300`.
- Architecture idea: implemented bounded `revision_gnn` from the belief-revision design mine. It
  passed toy-node and shape tests. The best fast config was confirmed across seeds `0,1,2` with the
  200-epoch budget and trailed confirmed GAT validation means on both Cora (`0.7020` versus
  `0.7313`) and PubMed (`0.7253` versus `0.7607`). Discard first-pass implementation.
- Validation metric was used for decisions. Test metrics were recorded but not used to choose.
- `uv run ruff check .` passed.
- `uv run pytest` passed: `17 passed, 2 skipped`.

Keep/discard:

- Keep confirmed Cora GIN tuning as a dataset-specific research config/insight.
- Discard GCN batch 1/2 and GAT batch 1/2 refinements.
- Discard PubMed GIN deeper candidate.
- Discard first-pass `revision_gnn` as a performance candidate; keep code only as an explicit
  bounded experiment artifact unless the project wants to remove discarded model code.

Notes:

- Do not update `configs/models/gin.yaml`: the Cora tuning did not transfer to PubMed.
- `research_results.tsv` and `results/runs/` remain local research artifacts.
- The sweep TSV writer initially recorded seed `0` as an empty field because it used a truthiness
  check when serializing metadata. Fixed in `src/gnn_gym/experiments/sweep.py` and covered by
  `tests/test_runtime_features.py`; older rows in the local ledger still show the previous blank.

Next:

- If continuing hyperparameter search, focus GIN Cora around `hidden=128,layers=3,dropout=0.4-0.6`
  or GAT PubMed around the existing default; do not spend more fast budget on the explored GCN
  settings.
- If continuing architecture work, prefer a residual GIN or non-backtracking model with a sharper
  same-budget comparison target rather than expanding `revision_gnn` without diagnostics.

## 2026-05-19 - Repository Scaffold And Multi-Task Trainers

Branch/commit: local uncommitted work
Files changed: `src/gnn_gym/`, `configs/`, `tests/`, `pyproject.toml`, `uv.lock`
Command:

```bash
uv run ruff check .
uv run pytest
uv run gnngym train --model gcn --dataset toy-node --seed 0 --override training.max_epochs=2
uv run gnngym train --model gcn --dataset cora --seed 0 --override training.max_epochs=1
uv run gnngym train --model gat --dataset pubmed --seed 0 --override training.max_epochs=1
uv run gnngym train --model gin --dataset ogbg-molhiv --seed 0 --override training.max_epochs=1
```

Result:

- Lint passes.
- Unit tests pass.
- Toy node, graph, link, neighbor-node smoke paths pass.
- Cora, PubMed, and MolHIV smoke runs completed.

Keep/discard: keep

Notes:

- OGB loaders need a PyTorch 2.6 compatibility wrapper around `torch.load(weights_only=False)`.
- Current graph trainer pools model node outputs directly. Long term, this should become cleaner
  encoder/head separation.

Next:

- Add a formal experiment runner that writes a compact TSV/CSV summary per architecture trial.
- Add architecture-specific result tables with mean/std over seeds.

## Future Queue

- Implement `res_gcn` as first architecture research candidate.
- Add richer graph pooling head.
- Add GraphSAGE as a scalable node-classification baseline.
- Add a small fixed-budget benchmark config analogous to autoresearch's fixed 5-minute run.

## 2026-05-23 - Complete Remaining Markdown Ideas Pass

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: new model/config/sweep files for `nb_light_gnn`, `confidence_appnp_net`,
`bethe_gnn`, `dual_primal_gnn`, `equilibrium_belief_gnn`, `region_collapse_gnn`, `kikuchi_gnn`,
`loop_corrected_gnn`, `decimation_gnn`, and `walk_belief_transformer`; graph pooling head support;
research notes and exported tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_nb_light_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_confidence_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_bethe_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_dual_primal_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_equilibrium_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_region_collapse_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_kikuchi_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_loop_corrected_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_decimation_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_walk_belief_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_concat_pool_toy_graph.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_concat_pool_molhiv_fast.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `nb_light_gnn`: implemented generated directed-edge light propagation. Best validation was Cora
  `0.6900`, PubMed `0.7900`; below confirmed GPR and below confirmed `nb_belief_gnn` on PubMed.
- `confidence_appnp_net`: implemented generated confidence-gated APPNP. Best validation was Cora
  `0.6420`, PubMed `0.7180`; discarded.
- `bethe_gnn`: implemented no-auxiliary-loss Bethe-style node/edge consistency. Best validation was
  Cora `0.4580`, PubMed `0.5580`; discarded.
- `dual_primal_gnn`: implemented coupled node/edge factor states. Best validation was Cora
  `0.5020`, PubMed `0.6620`; discarded.
- `equilibrium_belief_gnn`: implemented unrolled recurrent fixed-point-style propagation. Best
  validation was Cora `0.3160`, PubMed `0.4780`; discarded.
- `region_collapse_gnn`: implemented learned soft region coarsening. Best validation was Cora
  `0.6520`, PubMed `0.7520`; discarded.
- `kikuchi_gnn`: implemented ego-region messages. Best validation was Cora `0.5460`, PubMed
  `0.6960`; discarded.
- `loop_corrected_gnn`: implemented capped triangle loop correction. Best validation was Cora
  `0.4060`, PubMed `0.6440`; discarded.
- `decimation_gnn`: implemented trainer-free hidden-state decimation. Best validation was Cora
  `0.6020`, PubMed `0.7520`; discarded.
- `walk_belief_transformer`: implemented deterministic short-walk transformer. Best validation was
  Cora `0.6640`, PubMed `0.7200`; discarded.
- Dataset-conditioned pooling: implemented opt-in `mean_max_add` pooling. Toy-graph smoke passed.
  Short MolHIV seed-0 check gave `mean_max_add` validation `0.7489` versus matched short mean
  pooling `0.7163`, but below existing GIN MolHIV baseline mean `0.7619`; do not promote.
- Aggregated research table now has `1731` rows.

Keep/discard:

- No new model from this pass beats confirmed GPR on Cora/PubMed.
- Keep the implementations as explicit bounded experiment artifacts and negative results.
- Keep `mean_max_add` as an opt-in graph pooling mode only; do not update any default model YAML.

Notes:

- The broad result is consistent with the earlier loop: complex belief/region/recurrent machinery
  under the fixed harness does not beat simple decoupled learnable propagation on Cora/PubMed.
- Ideas that likely require auxiliary losses or trainer changes, especially Bethe/RIGN/decimation,
  should not be reopened as plain encoders.

Next:

- Use `gpr_gnn` as the node benchmark to beat.
- Future work should move to matched `ogbn-arxiv` checks or graph-task-specific architectures
  rather than further Cora/PubMed encoder-only variants.

## 2026-05-22 - Generated Architecture Search: NB-APPNP, Residual APPNP, GPR

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: `research/GENERATED_ARCHITECTURE_IDEAS.md`, new model/config/sweep files for
`nb_appnp_net`, `gated_appnp_net`, `res_appnp_net`, and `gpr_gnn`; research notes and exported
tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_nb_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gated_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_pubmed_confirm_prop5.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_cora_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_pubmed_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_cora_refined_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_pubmed_refined_confirm.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `nb_appnp_net`: discarded. Best fast validation was Cora `0.6720`, PubMed `0.7240`.
- `gated_appnp_net`: discarded. Best fast validation was Cora `0.7180`, PubMed `0.7300`.
- `res_appnp_net`: not a new best. Cora confirmation was validation mean `0.7733`, below APPNP
  `0.7753`; PubMed confirmation was `0.7900`, above APPNP `0.7880` but below NB-belief `0.7947`.
  A second PubMed config confirmed at `0.7833`.
- `gpr_gnn`: kept. Initial confirmation reached Cora validation mean `0.7807`, above APPNP
  `0.7753`, and PubMed validation mean `0.8007`, above NB-belief `0.7947` and APPNP `0.7880`.
- GPR refinement: keep refined Cora config `dropout=0.1,propagation_steps=16,alpha=0.1`, which
  confirmed validation mean `0.7827`. Discard refined PubMed config `dropout=0.3,alpha=0.05`, which
  confirmed at `0.7967`, below the original GPR PubMed `0.8007`.
- Aggregated research table now has `1318` rows.

Keep/discard:

- Keep `gpr_gnn` as the strongest current Cora/PubMed node architecture from this loop.
  Validation-selected configs: Cora `hidden=128,layers=1,dropout=0.1,K=16,alpha=0.1`; PubMed
  `hidden=64,layers=1,dropout=0.2,K=10,alpha=0.1`.
- Discard direct NB/APPNP fusion and unconstrained APPNP gating.
- Keep `res_appnp_net` only as an informative negative/marginal experiment; do not promote it over
  `gpr_gnn`.

Notes:

- Conservative generated ideas worked better than elaborate hybrids. The successful change was
  learnable hop weights around an APPNP-like prior, not more edge-state machinery.
- Seed-0 lifts are still noisy: residual APPNP looked strong at seed 0 but did not beat the best
  confirmed baselines across seeds.

Next:

- Run a matched `gpr_gnn`/APPNP/GAT scale check on `ogbn-arxiv` only after the Cora/PubMed notes
  are clean and tests pass.

## 2026-05-22 - Belief-Mined Architecture Sweeps: Frustration, NB, Entropy, RIGN, Ladder, Survey

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: new model/config/sweep files for `frustration_gnn`, `nb_belief_gnn`,
`entropy_gated_gnn`, `rign_gnn`, `temp_ladder_gnn`, and `survey_gnn`; research notes and exported
tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_frustration_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_nb_belief_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_nb_belief_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_entropy_gated_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_rign_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_temp_ladder_fast.yaml
uv run gnngym train --model survey_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_survey_fast.yaml
uv run gnngym train --model cavity_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_cavity_fast.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `nb_belief_gnn`: kept for PubMed. Confirmed across seeds `0,1,2` with validation mean `0.7947`,
  above confirmed APPNP PubMed `0.7880`. Held-out test mean was lower than APPNP and is recorded
  only for reporting.
- `frustration_gnn`: discarded. Best fast validation was Cora `0.6840`, PubMed `0.7460`.
- `entropy_gated_gnn`: discarded. Best fast validation was Cora `0.6940`, PubMed `0.7820`.
- `rign_gnn`: discarded as an encoder-only shortcut. Best fast validation was Cora `0.5680`,
  PubMed `0.7060`.
- `temp_ladder_gnn`: discarded. Best fast validation was Cora `0.6100`, PubMed `0.7340`.
- `survey_gnn`: discarded. Best fast validation was Cora `0.6960`, PubMed `0.7760`.
- `cavity_gnn`: discarded. Best fast validation was Cora `0.3340`, PubMed `0.6720`.
- Aggregated research table now has `997` rows.

Keep/discard:

- Keep `nb_belief_gnn` as a PubMed validation improvement, with the caveat that the current Python
  reverse-edge lookup is a small-dataset prototype.
- Discard the first-pass frustration, entropy-gated, RIGN encoder-only, temperature-ladder,
  survey-particle, and GRU-cavity models.

Notes:

- The strongest repeated signal from the original ideas is not generic recurrence or extra latent
  particles; it is directed non-backtracking edge state.
- The full GRU cavity update performed much worse than the simpler `nb_belief_gnn`, so future
  edge-state hybrids should preserve simple stable propagation and add a conservative fusion path.
- Encoder-only versions of ideas that originally relied on auxiliary objectives underperformed.
  Reopening RIGN or SurveyGNN should include the intended supervision/diagnostic rather than simply
  widening the same encoders.

Next:

- Create `research/GENERATED_ARCHITECTURE_IDEAS.md` from the observed insights and start testing a
  newly generated idea that combines APPNP's stable propagation with NB's directed edge signal.

## 2026-05-22 - Continued Architecture Search: ResGIN, JK-GCN, APPNP, GCNII, GATv2

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: new model/config/sweep files for `res_gin`, `jk_gcn`, `appnp_net`, `gcn2_net`,
and `gatv2`; research notes and exported tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_gin_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_jk_gcn_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_jk_gcn_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_cora_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_pubmed_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_cora_refined_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gcn2_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gatv2_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gatv2_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gatv2_pubmed_confirm.yaml
uv run gnngym train --model appnp_net --dataset ogbn-arxiv --seed 0 --override training.max_epochs=50 ...
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `res_gin`: discarded. Best fast validation was Cora `0.5220`, PubMed `0.6980`.
- `jk_gcn`: kept for PubMed. Confirmed validation mean `0.7793` versus GAT PubMed `0.7607`.
- `appnp_net`: current strongest Cora/PubMed model. Confirmed Cora validation mean `0.7753`
  versus GAT `0.7313`; confirmed PubMed validation mean `0.7880` versus GAT `0.7607` and JK-GCN
  `0.7793`.
- APPNP Cora refinement found a seed-0 run at `0.8040`, but the 3-seed refined confirmation
  underperformed the original APPNP Cora confirmation (`0.7700` versus `0.7753`), so keep the
  original Cora APPNP config.
- `gcn2_net`: discarded. Best fast validation was Cora `0.4880`, PubMed `0.6260`.
- `gatv2`: kept as a Cora attention-family improvement. Confirmed Cora validation mean `0.7553`
  versus GAT `0.7313`; PubMed did not confirm versus GAT.
- APPNP `ogbn-arxiv` seed-0 scale smoke completed with validation `0.5810`, test `0.5123`, and
  train time `14.7s`. No arxiv baseline table exists yet, so this is not an improvement claim.
- `results/tables/research_all_runs.csv` now has `603` rows.

Keep/discard:

- Keep `appnp_net`, `jk_gcn`, and Cora `gatv2` as validated architecture results.
- Discard `res_gin`, `gcn2_net`, PubMed `gatv2`, and refined APPNP Cora config.

Notes:

- APPNP should now be the node architecture benchmark to beat on Cora/PubMed.
- For medium-node follow-up, establish matched `gcn`, `gat`, `jk_gcn`, and `appnp_net` baselines on
  `ogbn-arxiv` before making claims.

Next:

- Run lint/tests after the expanded model set.
- If continuing, try APPNP weight decay and alpha refinements, or a graph-specific GIN pooling
  experiment on `ogbg-molhiv` with a small run budget.

## 2026-05-26 - Synthetic Diagnostic Shortcut/Audit Suite

Scope:

- Added reusable synthetic diagnostic helpers in `gnn_gym.evaluation.synthetic_diagnostics`.
- The suite covers class-prevalence AP, graph-statistic logistic/MLP controls, candidate detector
  count/histogram controls, same-feature controls, same-capacity merged controls, model prediction
  invariance audits, cache-key audits, capped-enumeration tie stress, and metric-invalidation
  records.
- Applied the suite first in the existing CycleCutGNN-lite and DualShadowGNN-lite worktrees.

Branch-local outputs:

- CycleCutGNN-lite:
  `results/tables/cycle_cut_gnn_lite_synthetic_diagnostic_audit.json` and
  `results/tables/cycle_cut_gnn_lite_graph_audit_suite.json`.
- DualShadowGNN-lite:
  `results/tables/dual_shadow_gnn_synthetic_diagnostic_audit.json` and
  `results/tables/dual_shadow_gnn_graph_audit_suite.json`.

Findings:

- CycleCutGNN-lite's best prior validation AP (`cycle_only_mean_val_ap=0.7686`) only narrowly
  clears the strongest exact same-feature / merged control (`0.7576`). Treat this as weak synthetic
  support, not a broad architecture claim.
- DualShadowGNN-lite's confirmed validation AP (`0.8802`) clears the strongest new control
  (`0.8571`), but the margin is modest and the capped-enumeration stress audit remains a clear risk
  under tight face caps.
- Both branch-local graph audits passed edge-order, random relabeling, and batch-composition checks
  on small audit graphs. Both current raw model cache keys are edge-order unstable, so cache hits are
  not canonical even when predictions are stable.

Policy update:

- Future synthetic diagnostic claims must beat exact shortcut and same-capacity controls by
  validation metric before being described as support. Older shortcut-only files are explicitly
  invalidated by replacement audit JSON when the suite adds correctness-relevant controls.
