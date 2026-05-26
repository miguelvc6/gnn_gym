# TreePackGNN-lite

Architecture name: TreePackGNN-lite

Source idea file: `research/original-ideas/graph_theory_original_gnn_ideas.md`

Scientific hypothesis: A graph may be better represented by a small set of complementary
spanning-tree views than by either ordinary local message passing or one arbitrary DFS tree.
Keeping tree views separate until a learned pooling/gating step may preserve useful global
path/cycle information while reducing dependence on a single node-order-sensitive witness.

Mechanism: Build a small deterministic pack of spanning-tree/forest views per graph, encode each
view separately with shared tree-message parameters, combine the views with learned per-node gates,
and retain a full-graph residual channel so bad tree views can be ignored.

Why this is not just a known baseline: The test is not "multiple trees are novel." The mechanism is
learned selection over structurally different spanning-tree witnesses while keeping the witnesses
separate until the gate. This is compared directly against single-order DFS and naive unweighted
multi-order DFS averaging.

Closest known related architectures: Tree-based GNNs, random-walk/graph augmentation ensembles,
DropEdge-style structural perturbations, and spanning-tree graph methods.

Expected insight if success: A small view pack can retain the useful synthetic signal from
tree/back-edge-style witnesses while greatly reducing arbitrary relabeling instability.

Expected insight if failure: Multiple tree witnesses may either wash out the relevant signal, like
unweighted DFS averaging, or collapse to the ordinary residual graph channel.

Target task family: synthetic graph diagnostics first; only later real graph tasks if the diagnostic
signal and invariance audits hold.

Primary baseline to beat: `normal_tree_backedge_gnn` single-order DFS on `cycle_matching_v4`, plus
the four-order unweighted average and ordinary `gin`/`gcn` baselines.

Minimal falsifying experiment: Run toy-graph crash check, then seed-0 `cycle_matching_v4`. Confirm
over seeds `[0,1,2]` only if seed 0 is competitive with NormalTreeBackedgeGNN validation AP.

Confirmation protocol: Use validation AP for selection. Audit relabeling instability on confirmed
checkpoints with 16 random relabelings of the same test graphs.

Complexity/runtime risk: Medium. Tree construction is cheap, but four tree-view passes increase
runtime versus one DFS witness.

Implementation boundary: Add one model, one config, focused shape/tree-view tests, and research
notes. Do not change trainers, datasets, evaluators, or NormalTreeBackedgeGNN.

## Implementation

Files added or changed:

- `src/gnn_gym/models/tree_pack_gnn.py`
- `configs/models/tree_pack_gnn.yaml`
- `tests/test_tree_pack_gnn.py`
- `tests/test_model_shapes.py`
- `tests/test_registry.py`
- `research/experiments/tree_pack_gnn.md`
- `research/AGENT_SCRATCHPAD.md`
- `research/INSIGHTS.md`
- aggregate and audit artifacts under `results/tables/`

Tree views:

1. BFS tree from a high-degree root.
2. BFS tree from a farthest/low-degree root.
3. Deterministic DFS tree.
4. Low-overlap degree-ordered BFS tree that avoids edges already used by earlier views.

The model does not average tree edge labels. Each tree view is encoded separately with shared tree
message parameters. A learned node-level softmax gate combines tree-view representations. A
full-graph residual message channel is included in every layer.

## Command Log

```bash
uv run gnngym train --model tree_pack_gnn --dataset toy-graph --seed 0 --override training.max_epochs=2 --override training.patience=2 --override model.hidden_channels=16 --override model.head_hidden_channels=16 --override model.num_trees=4 --override trainer.batch_size=8

uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=32 --override model.head_hidden_channels=32 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16

uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 0 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 1 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16
uv run gnngym train --model tree_pack_gnn --dataset normal-tree-backedge --seed 2 --override dataset.variant=cycle_matching_v4 --override training.max_epochs=50 --override training.patience=15 --override model.hidden_channels=64 --override model.head_hidden_channels=64 --override model.num_layers=3 --override model.num_trees=4 --override model.dropout=0.2 --override model.pooling=mean_max_add --override trainer.batch_size=16

UV_CACHE_DIR=/tmp/uv-cache uv run python research/experiments/normal_tree_backedge_permutation_audit.py --run-dir <tree_pack_run> --num-relabels 16 --seed 123 --batch-size 16 --output results/tables/tree_pack_gnn_permutation_audit_seed*_wide.json

uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

## Results

Config-level summary:

| Model/config | Seeds | architecture_config_hash | Val AP mean/std | Test AP mean/std | Mean train seconds | Params |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| GIN baseline | 3 | `42244e3d` | 0.4167 +/- 0.0000 | 0.4167 +/- 0.0000 | see aggregate | see aggregate |
| GCN baseline | 3 | `f594dd4a` | 0.4167 +/- 0.0000 | 0.4167 +/- 0.0000 | see aggregate | see aggregate |
| NormalTreeBackedge single-order | 3 | `f9449a35` | 0.6635 +/- 0.0788 | 0.5516 +/- 0.0865 | 14.34 | 35,649 |
| NormalTreeBackedge 4-order average | 3 | `2fb8a3d0` | 0.6490 +/- 0.0864 | 0.5626 +/- 0.2606 | 75.73 | 35,649 |
| TreePackGNN, hidden 32 | 1 | `8183ffd8` | 0.5856 | 0.5479 | 21.05 | 22,596 |
| TreePackGNN, hidden 64 | 3 | `8ec4a7ff` | 0.7462 +/- 0.0914 | 0.5838 +/- 0.0815 | 31.12 | 88,196 |

Per-seed confirmed TreePack result:

| Seed | Val AP | Test AP | Best epoch |
| ---: | ---: | ---: | ---: |
| 0 | 0.7305 | 0.6629 | 30 |
| 1 | 0.6638 | 0.5884 | 1 |
| 2 | 0.8444 | 0.5001 | 12 |

Decision: Keep TreePackGNN-lite as an interesting diagnostic architecture. It beats the previous
single-order and four-order NormalTreeBackedge validation means and modestly improves held-out test
mean, but seed 2 still shows weak held-out generalization.

## Permutation-Invariance Audit

Audit setup: Reload the best checkpoint for each confirmed seed, randomly relabel the same 24 test
graphs 16 times, recompute tree views, and compare predictions.

| Model | Seed | Original AP | Relabeled AP mean/std | Mean prediction variance | Max prediction range |
| --- | ---: | ---: | ---: | ---: | ---: |
| NormalTreeBackedge single-order | 0 | 0.5384 | 0.5557 +/- 0.0797 | 0.056555 | 0.7055 |
| NormalTreeBackedge single-order | 1 | 0.5500 | 0.4504 +/- 0.1074 | 0.000412 | 0.0937 |
| NormalTreeBackedge single-order | 2 | 0.4709 | 0.5791 +/- 0.1066 | 0.019650 | 0.7087 |
| TreePackGNN | 0 | 0.6629 | 0.6620 +/- 0.1176 | 0.0000000019 | 0.0002 |
| TreePackGNN | 1 | 0.5884 | 0.6692 +/- 0.1331 | 0.0000000566 | 0.0013 |
| TreePackGNN | 2 | 0.5001 | 0.4517 +/- 0.0854 | 0.0000001799 | 0.0027 |

Interpretation: TreePackGNN substantially reduces relabeling prediction instability relative to
single-order NormalTreeBackedgeGNN. The max per-graph prediction range drops from up to about
`0.71` to at most `0.0027`.

## Comparison To NormalTreeBackedgeGNN

TreePackGNN improves the two failure modes that motivated this cycle:

- It avoids a single arbitrary DFS witness.
- It avoids unweighted averaging of incompatible DFS witnesses by keeping each tree view separate
  until a learned gate.

On `cycle_matching_v4`, the confirmed hidden-64 TreePack config beats the single-order
NormalTreeBackedge validation mean (`0.7462` vs `0.6635`), improves held-out test mean (`0.5838`
vs `0.5516`), and sharply reduces relabeling instability. It also beats the naive four-order DFS
average on validation (`0.7462` vs `0.6490`) and has much lower test variance.

The cost is higher runtime than single-order NormalTreeBackedge (`31.12s` vs `14.34s`) and more
parameters (`88,196` vs `35,649`). It is still much cheaper than the previous four-order DFS average
runtime (`75.73s`).

## Scientific Interpretation

This is a positive diagnostic result for learned view selection over structurally diverse tree
witnesses. The stability audit suggests TreePackGNN is not merely repeating the single DFS
non-invariance problem. The validation/test gap and seed-2 test weakness mean it is not ready for a
real-benchmark claim.

## Ablation Cycle

Follow-up ablations tested whether the confirmed four-view result was actually caused by learned
view selection, the residual full-graph channel, or one dominant tree view.

Additional implementation controls:

- `model.use_graph_channel`: disables/enables the ordinary graph residual channel.
- `model.use_tree_channel`: disables/enables tree-view message channels.
- `model.tree_pooling`: `gated` or unweighted `mean` view pooling.
- `model.tree_start_idx`: selects which deterministic tree-view family starts a single-tree
  control.

Seed-0 ablations:

| Config | architecture_config_hash | Val AP | Test AP | Train seconds | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Full 4-tree gated pack | `8ec4a7ff` | 0.7305 | 0.6629 | 31.31 | reference seed-0 full pack |
| Graph residual only | `e8832f31` | 0.4167 | 0.4167 | 2.84 | residual channel alone is chance |
| Tree-only 4-tree gated pack | `250900ee` | 0.6339 | 0.5992 | 23.42 | tree channel carries signal but needs residual/fusion |
| 4-tree mean pooling | `cc125dc6` | 0.7089 | 0.4259 | 33.68 | learned gate beats mean pooling on seed-0 validation |
| Single tree 0, high-degree BFS | `098eb1ec` | 0.7540 | 0.6795 | 22.31 | strongest seed-0 validation among controls |
| Single tree 1, farthest/low-degree BFS | `7682e0ad` | 0.5218 | 0.5285 | 16.51 | weak |
| Single tree 2, deterministic DFS | `119b05b0` | 0.7142 | 0.6171 | 17.36 | useful but below tree 0 |
| Single tree 3, low-overlap/low-degree BFS | `fe4388ea` | 0.7947 | 0.6364 | 23.08 | strong seed-0 validation, unstable across seeds |

Confirmed single-tree controls:

| Config | Seeds | architecture_config_hash | Val AP mean/std | Test AP mean/std | Mean train seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| Full 4-tree gated pack | 3 | `8ec4a7ff` | 0.7462 +/- 0.0914 | 0.5838 +/- 0.0815 | 31.12 |
| Single tree 0, high-degree BFS | 3 | `098eb1ec` | 0.7493 +/- 0.0113 | 0.6812 +/- 0.0185 | 16.21 |
| Single tree 3, low-overlap/low-degree BFS | 3 | `fe4388ea` | 0.7176 +/- 0.0881 | 0.6989 +/- 0.0632 | 14.50 |

Gate diagnostics on the full four-tree confirmed checkpoints showed mostly high-entropy, near-uniform
view weights. The final layer mean gate weights were:

| Seed | Layer-2 mean gate weights | Layer-2 entropy |
| ---: | --- | ---: |
| 0 | `[0.2662, 0.2896, 0.1773, 0.2669]` | 1.3179 |
| 1 | `[0.2499, 0.2498, 0.2505, 0.2498]` | 1.3863 |
| 2 | `[0.2891, 0.2410, 0.1926, 0.2774]` | 1.2175 |

Maximum entropy for four views is about `1.3863`, so the learned gate usually does not perform
sharp view selection.

Single-tree-0 permutation audit:

| Config | Seed | Original AP | Relabeled AP mean/std | Mean prediction variance | Max prediction range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single tree 0, high-degree BFS | 0 | 0.6795 | 0.6251 +/- 0.0956 | 0.00000549 | 0.0113 |
| Single tree 0, high-degree BFS | 1 | 0.6637 | 0.5855 +/- 0.0928 | 0.00000204 | 0.0068 |
| Single tree 0, high-degree BFS | 2 | 0.7005 | 0.6183 +/- 0.0894 | 0.00000635 | 0.0141 |

The single high-degree BFS control is still far more relabeling-stable than the old single-order
NormalTreeBackedgeGNN, whose max prediction range reached about `0.71`, but it is less stable than
the full 4-tree pack, whose max range was at most `0.0027`.

## Ablation Interpretation

The residual-only control rules out the ordinary full-graph channel as the source of the result.
However, the strongest confirmed control is a single high-degree BFS tree, which slightly beats the
four-tree pack on validation and substantially beats it on held-out test AP with lower runtime and
lower seed variance. The four-tree learned gate also stayed close to uniform instead of learning a
clear view-selection policy.

This weakens the specific TreePack hypothesis. The evidence supports "some deterministic BFS tree
witness is useful on this diagnostic" more than "a learned pack of structurally diverse tree views
is better than one well-chosen tree witness." Treat TreePackGNN-lite as a negative or at best
inconclusive result for learned tree-pack selection. Preserve the single-tree high-degree BFS result
as a strong synthetic control, but move the next architecture cycle to `CycleCutGNN-lite` unless a
new mechanism can select or canonicalize tree views more sharply.
