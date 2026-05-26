# NormalTreeBackedgeGNN-lite

Architecture name: NormalTreeBackedgeGNN-lite

Source idea file: `research/original-ideas/graph_theory_original_gnn_ideas.md`

Scientific hypothesis: A graph's DFS tree gives a cheap hierarchical scaffold, and non-tree
back edges encode cycle closures. Separating tree propagation from back-edge cycle-closure messages
should help tasks where graphs share a similar backbone but differ in long-range back-edge patterns.

Mechanism: Precompute a deterministic DFS forest per graph. Mark directed edges as upward tree,
downward tree, or non-tree/back-edge. Add normalized source/destination depth and back-edge span
features. Run separate learned message channels for upward tree, downward tree, and back-edge
messages, then combine them with a residual node update.

Why this is not just a known baseline: GIN and GCN aggregate over all edges with one structural
treatment. GAT can learn generic attention but does not impose a DFS tree/back-edge decomposition or
explicit cycle-closure span features.

Closest known related architectures: TreeLSTM/tree GNNs, cycle-aware GNNs, positional encodings
from traversal depth or random walks, and edge-type message-passing networks.

Expected insight if success: DFS tree/back-edge factorization can expose cycle-closure information
that ordinary graph-level GNNs do not reliably separate.

Expected insight if failure: The DFS scaffold may be too arbitrary, or the synthetic task may be
solvable by ordinary message passing without explicit tree/back-edge channels.

Primary baseline: `gin` on the same synthetic graph classification diagnostic.

Minimal falsifying experiment: Add a small `normal-tree-backedge` synthetic graph dataset where all
graphs share a path backbone but differ in whether added back edges have long-span or short-span
cycle closures. Run seed-0 fast screens for `gin` and `normal_tree_backedge_gnn`.

Confirmation protocol: Confirm across seeds `[0, 1, 2]` only if seed-0 validation improves over
`gin` on the synthetic diagnostic. Do not claim Cora/PubMed improvement from this graph-task
diagnostic.

Complexity/runtime risk: Low to medium. DFS edge tagging is linear in edge count and cached by graph
signature. Multiple channels add a small constant-factor cost.

Implementation boundary: Add one model, one model config, one synthetic diagnostic dataset config
and loader, focused tests, and research notes. Do not alter trainers or evaluators.

## Results

Implementation summary: Added `NormalTreeBackedgeGNN-lite`, a deterministic DFS-forest encoder that
marks directed edges as upward tree, downward tree, or non-tree/back-edge edges. The model uses
separate learned message channels for those three edge classes plus normalized source depth, target
depth, and span features. Added `normal-tree-backedge`, a synthetic graph classification diagnostic
where graphs share a path backbone and differ by short-span versus long-span chord/back-edge
pairings.

Files changed:

- `src/gnn_gym/models/normal_tree_backedge_gnn.py`
- `configs/models/normal_tree_backedge_gnn.yaml`
- `src/gnn_gym/data/catalog.py`
- `configs/datasets/normal_tree_backedge.yaml`
- `tests/test_model_shapes.py`
- `tests/test_registry.py`
- `tests/test_normal_tree_backedge_gnn.py`
- `results/tables/research_all_runs.csv`
- `results/tables/research_all_runs_by_config_mean_std.csv`
- `results/tables/research_all_runs_by_config_table.md`
- `results/tables/research_all_runs_by_config_table.tex`
- `results/tables/research_all_runs_by_model_mean_std.csv`
- `results/tables/research_all_runs_by_model_table.md`
- `results/tables/research_all_runs_by_model_table.tex`
- `research/experiments/normal_tree_backedge_gnn.md`
- `research/AGENT_SCRATCHPAD.md`
- `research/INSIGHTS.md`

Command log:

```bash
uv run pytest tests/test_normal_tree_backedge_gnn.py tests/test_model_shapes.py::test_model_output_shape tests/test_registry.py
uv run ruff check .
uv run gnngym train --model normal_tree_backedge_gnn --dataset toy-graph --seed 0 --override training.max_epochs=2 --override training.patience=2 --override model.hidden_channels=16 --override model.head_hidden_channels=16 --override trainer.batch_size=8
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 0 --override dataset.variant=endpoint_pairing_v2 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=endpoint_pairing_v2 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run pytest
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Results table:

| Dataset | Model | Budget | Seeds | architecture_config_hash | Val metric | Test metric | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toy-graph | normal_tree_backedge_gnn | 2 epochs / patience 2 | 0 | `7c263e4b` | crash passed | crash passed | Smoke only |
| normal-tree-backedge | gin | 50 epochs / patience 15 | 0 | `bb49e9d0` | 1.0000 | 1.0000 | Baseline solves diagnostic |
| normal-tree-backedge | normal_tree_backedge_gnn | 50 epochs / patience 15 | 0 | `83e3c616` | 1.0000 | 1.0000 | Ties baseline; no confirmation |

Comparison to baseline: `normal_tree_backedge_gnn` tied `gin` at seed-0 validation AP 1.0, but it
used more parameters and took longer on the diagnostic. This is not an improvement signal and does
not justify seed confirmation.

Interpretation: The model implementation works, but the synthetic diagnostic is too easy for a
standard GIN baseline. The result does not falsify the DFS tree/back-edge mechanism itself; it
falsifies this diagnostic as a useful discriminator.

Failure modes: The diagnostic still leaks enough structure through local degree and chord-endpoint
features for GIN to solve immediately. The edge-span pairing signal needs a harder construction,
probably with stronger degree/profile matching and without simple endpoint markers.

What this teaches us: A NormalTreeBackedgeGNN experiment needs a better synthetic task before real
benchmarking. Perfect baseline performance on the diagnostic means no architecture claim can be
made from this run.

Next recommended step: Either harden the diagnostic with matched local neighborhoods and no endpoint
markers, or move to `TreePackGNN` if the next cycle should test a Cora/PubMed-compatible idea
without more synthetic task design.

## Hardened Diagnostic: cycle_matching_v4

Implementation summary: Replaced the easy endpoint-pairing diagnostic with `cycle_matching_v4`.
Each graph is a 20-node cycle plus a nonlocal perfect matching, so every graph has 20 nodes,
30 undirected edges, density 0.1579, and all nodes have degree 3. Node features are constant, and
each graph is randomly relabeled before it is stored. Labels depend on whether the hidden cycle
matching has low or high chord-crossing arrangement, not on node count, edge count, degree
distribution, or cycle rank.

Shortcut checks: Added `research/experiments/normal_tree_backedge_shortcut_check.py`, which computes
node count, edge count, density, degree summary, triangle count, DFS back-edge count, and DFS
back-edge span summaries, then fits logistic and MLP baselines on those graph-level statistics.

Shortcut results:

| Baseline | Val AP | Test AP | Val accuracy | Test accuracy |
| --- | --- | --- | --- | --- |
| logistic graph stats | 0.5777 | 0.5191 | 0.6667 | 0.5417 |
| MLP graph stats | 0.6419 | 0.5192 | 0.5417 | 0.5000 |

Command log:

```bash
uv run python research/experiments/normal_tree_backedge_shortcut_check.py
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model gcn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model gin --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model gcn --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model gcn --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.num_layers=3 --override model.dropout=0.2 --override model.pooling=mean_max_add --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override trainer.batch_size=16
uv run ruff check .
uv run pytest
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Results table:

| Model | Seeds | architecture_config_hash | Val AP mean/std | Test AP mean/std | Interpretation |
| --- | --- | --- | --- | --- | --- |
| GIN | 0,1,2 | `42244e3d` | 0.4167 +/- 0.0000 | 0.4167 +/- 0.0000 | Does not solve diagnostic |
| GCN | 0,1,2 | `f594dd4a` | 0.4167 +/- 0.0000 | 0.4167 +/- 0.0000 | Does not solve diagnostic |
| NormalTreeBackedgeGNN | 0,1,2 | `f9449a35` | 0.6635 +/- 0.0788 | 0.5516 +/- 0.0865 | Validation advantage, weak held-out generalization |

Comparison to baseline: The hardened diagnostic no longer saturates for GIN/GCN, and graph-stat
shortcut baselines do not solve it. `NormalTreeBackedgeGNN-lite` has a clear validation advantage
over GIN/GCN on the diagnostic, but the test mean is much weaker than validation and remains only
moderately above chance.

Interpretation: This is useful diagnostic evidence that DFS tree/back-edge features expose signal
unavailable to ordinary constant-feature GIN/GCN on this matched regular graph task. It is not a
real-benchmark architecture claim, and the validation/test gap suggests sensitivity to random DFS
tree choice, split size, or overfitting.

Failure modes: Deterministic DFS depends on arbitrary relabeled node IDs. The model may be learning
unstable traversal artifacts rather than a robust normal-tree invariant. Average precision also has
high variance on the small validation/test splits.

What this teaches us: The mechanism is no longer untested: it can exploit a diagnostic designed
around long-range chord arrangement after local statistics and standard GNN shortcuts are blocked.
The next scientific question is robustness to DFS root/order.

Next recommended step: Redesign `NormalTreeBackedgeGNN-lite` to average over multiple deterministic
DFS/normal-tree orderings or sampled roots, then rerun `cycle_matching_v4`. Move to real graph
benchmarks only if the diagnostic validation advantage survives and held-out test AP improves.

## Multi-Order DFS Averaging

Implementation summary: Added `model.num_tree_orders` to `NormalTreeBackedgeGNN-lite`. For each
graph, the model now can build several deterministic DFS forests using different root and neighbor
orders. The up-tree, down-tree, and back-edge message channels are shared across orders, and their
outputs are averaged before the node update. The default YAML remains `num_tree_orders=1` because
the four-order variant did not improve the diagnostic.

Command log:

```bash
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=4 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=4 --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=4 --override trainer.batch_size=16
uv run ruff check .
uv run pytest
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Results table:

| Model variant | Seeds | architecture_config_hash | Val AP mean/std | Test AP mean/std | Mean train seconds |
| --- | --- | --- | --- | --- | --- |
| single-order DFS | 0,1,2 | `f9449a35` | 0.6635 +/- 0.0788 | 0.5516 +/- 0.0865 | 14.34 |
| 4-order DFS average | 0,1,2 | `2fb8a3d0` | 0.6490 +/- 0.0864 | 0.5626 +/- 0.2606 | 75.73 |

Per-seed multi-order results:

| Seed | Val AP | Test AP | Best epoch |
| --- | --- | --- | --- |
| 0 | 0.6000 | 0.4683 | 33 |
| 1 | 0.5981 | 0.3623 | 9 |
| 2 | 0.7488 | 0.8572 | 30 |

Interpretation: Averaging four deterministic DFS orderings did not stabilize the diagnostic. Mean
validation AP was slightly lower than the single-order model, test AP variance increased, and
runtime increased by roughly 5x. The likely issue is that naive averaging blurs useful
order-specific back-edge span signals rather than producing an invariant normal-tree feature.

What this teaches us: The current mechanism benefits from one deterministic decomposition on
validation, but simple multi-order averaging is not the right robustness fix. The next refinement
should use a learned order gate, root-specific pooling, or explicit cycle/matching features rather
than unweighted averaging.

Next recommended step: Stop expanding NormalTreeBackedgeGNN until there is a sharper robustness
mechanism. For the next architecture cycle, move to `TreePackGNN` or another idea-bank candidate.

## Edge-Role And Relabeling Audit

Implementation summary: Added ablation switches to the same `NormalTreeBackedgeGNN-lite`
implementation: `model.edge_role_mode` can use true DFS roles, collapse all active edges to the
same role signal, keep only tree edges, keep only back edges, or deterministically shuffle role
labels. Added `model.dfs_order_mode=random` as a cheap pseudo-random DFS root/order control. Added
`research/experiments/normal_tree_backedge_permutation_audit.py` to reload trained checkpoints,
randomly relabel the same test graphs, recompute DFS markers, and measure prediction stability.

Files changed:

- `src/gnn_gym/models/normal_tree_backedge_gnn.py`
- `configs/models/normal_tree_backedge_gnn.yaml`
- `tests/test_normal_tree_backedge_gnn.py`
- `research/experiments/normal_tree_backedge_permutation_audit.py`
- `results/tables/normal_tree_backedge_ablation_cycle_matching_v4.csv`
- `results/tables/normal_tree_backedge_ablation_cycle_matching_v4.md`
- `results/tables/normal_tree_backedge_permutation_audit_true_seed0.json`
- `results/tables/normal_tree_backedge_permutation_audit_2026-05-26_12-26-54__normal_tree_backedge_gnn__normal-tree-backedge__seed-1__cdd5baf7.json`
- `results/tables/normal_tree_backedge_permutation_audit_2026-05-26_12-27-10__normal_tree_backedge_gnn__normal-tree-backedge__seed-2__d2547fdc.json`
- aggregate tables under `results/tables/`

Command log:

```bash
uv run pytest tests/test_normal_tree_backedge_gnn.py tests/test_model_shapes.py -q
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed {0,1,2} --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=1 --override model.edge_role_mode={true,collapsed,tree_only,back_only,shuffled} --override model.dfs_order_mode=deterministic --override trainer.batch_size=16
uv run gnngym train --model normal_tree_backedge_gnn --dataset normal-tree-backedge --seed {0,1,2} --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_tree_orders=1 --override model.edge_role_mode=true --override model.dfs_order_mode=random --override trainer.batch_size=16
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
uv run python research/experiments/normal_tree_backedge_permutation_audit.py --run-dir <true-role-run> --num-relabels 16 --seed 123 --batch-size 16
uv run ruff check .
```

Class prevalence / random AP baseline: `cycle_matching_v4` has validation prevalence `10/24 =
0.4167` and test prevalence `10/24 = 0.4167`, so random AP is about `0.4167`.

Results table:

| Ablation | Seeds | architecture_config_hash | Val AP mean/std | Test AP mean/std | Mean runtime seconds |
| --- | --- | --- | --- | --- | --- |
| true roles | 0,1,2 | `287c0583` | 0.7292 +/- 0.1321 | 0.5198 +/- 0.0428 | 15.35 |
| collapsed roles | 0,1,2 | `cb56a709` | 0.6549 +/- 0.0757 | 0.3491 +/- 0.0418 | 11.73 |
| tree-edge only | 0,1,2 | `f2337adb` | 0.6517 +/- 0.1046 | 0.4856 +/- 0.0802 | 9.05 |
| back-edge only | 0,1,2 | `50346584` | 0.6155 +/- 0.0600 | 0.5535 +/- 0.0128 | 9.57 |
| shuffled roles | 0,1,2 | `b6f56cd7` | 0.6395 +/- 0.0608 | 0.4033 +/- 0.1064 | 8.20 |
| random DFS order | 0,1,2 | `21ead346` | 0.5702 +/- 0.0081 | 0.4782 +/- 0.0320 | 12.02 |

Permutation-invariance audit:

| Seed | Original test AP | Relabeled AP mean/std over 16 relabelings | Mean prediction variance | Max prediction range |
| --- | --- | --- | --- | --- |
| 0 | 0.5384 | 0.5557 +/- 0.0797 | 0.0566 | 0.7055 |
| 1 | 0.5500 | 0.4504 +/- 0.1074 | 0.0004 | 0.0937 |
| 2 | 0.4709 | 0.5791 +/- 0.1066 | 0.0197 | 0.7087 |
| mean | 0.5198 | 0.5284 +/- 0.0685 across seeds | 0.0255 | 0.5026 |

Comparison to controls: True edge roles beat collapsed and shuffled roles on validation AP by about
`+0.074` and `+0.090`, respectively, which is mechanism-specific synthetic support under the
validation-selection rule. Test AP does not cleanly support the same ranking: back-edge-only has the
highest test AP mean, and all test splits are small.

Interpretation: The validation signal is tied to real DFS role labels rather than only model
capacity, because collapsed and shuffled controls are worse. However, the permutation audit shows
substantial prediction movement under random node relabeling, and a same-checkpoint audit also
exposed batch-composition sensitivity from normalizing DFS depths over the whole PyG batch.

Failure modes: The current implementation is non-canonical. DFS markers depend on arbitrary node
IDs after relabeling, and depth/span normalization depends on the batched disconnected union rather
than each graph component separately. These issues make the mechanism unsuitable for broader
benchmark claims even though it has diagnostic signal.

What this teaches us: NormalTreeBackedgeGNN has mechanism-specific synthetic support, but not
graph-invariant support. The result should be treated as a useful diagnostic finding and an
implementation warning, not a validated architecture improvement.

Next recommended step: Stop refining the current single-order NormalTreeBackedgeGNN. Either
redesign it around a canonical or learned distribution over multiple DFS/normal trees with
per-graph normalization, or move to the next idea-bank architecture, preferably `TreePackGNN` before
`CycleCutGNN-lite`.

## Correctness Cleanup: Per-Graph Depth Normalization

Implementation summary: Fixed `NormalTreeBackedgeGNN-lite` preprocessing so DFS depth/span features
are normalized independently for each graph in a PyG graph-prediction batch. The prior code computed
markers on the batched disconnected union and used one maximum DFS depth across the whole batch,
which made structural features depend on which other graphs were batched together. The `batch=None`
path used by single-graph node-classification data preserves the previous behavior.

Regression test: Added a test showing that one graph's normalized structural features are unchanged
when it is batched with a deeper second graph.

Interpretation: This is a correctness cleanup only. No ablation or benchmark experiments were
rerun, so this note does not change the architecture evidence or performance conclusions above.
