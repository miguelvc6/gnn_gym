# SepBottleneckGNN-lite

Architecture name: SepBottleneckGNN-lite

Source idea file: `research/original-ideas/graph_theory_original_gnn_ideas.md`

Scientific hypothesis: Citation-graph node classification may lose useful signal when ordinary
diffusion treats robust intra-block edges and separator/bridge-mediated edges identically.

Mechanism: Precompute cheap undirected articulation-point and bridge tags from `edge_index`. Encode
nodes with an MLP, add a learned separator token to articulation nodes, run GPR-style propagation,
and add a conservative learned residual correction from messages sent along bridge or
articulation-incident edges.

Why this is not just a known baseline: GAT/GATv2 can learn generic edge attention, and GPR-GNN learns
hop weights, but this version hard-codes a graph-theoretic separator channel and tests whether that
specific structural distinction is useful.

Closest known related architectures: GPR-GNN, APPNP, GAT/GATv2 edge attention, structural-encoding
message passing, and oversquashing-aware or curvature-aware routing/rewiring methods.

Expected insight if it succeeds: Small separators may be useful architectural routing primitives,
not only diagnostics for oversquashing.

Expected insight if it fails: Cora/PubMed may not depend on cheap separator structure, or GPR-style
global diffusion may already dominate this bottleneck signal under shallow full-batch training.

Target task family: Node classification on Cora and PubMed.

Primary baseline to beat: Confirmed `gpr_gnn` validation metrics from `research/INSIGHTS.md`.

Minimal falsifying experiment: Toy-node crash check, then seed-0 fast screen on Cora and PubMed with
`training.max_epochs=50` and `training.patience=15`.

Confirmation protocol: Only if seed-0 validation beats or nearly matches the relevant confirmed GPR
baseline, run seeds `[0, 1, 2]` with `training.max_epochs=200` and `training.patience=50`, then
compare by `architecture_config_hash`.

Complexity/runtime risk: Low to medium. Bridge/articulation preprocessing is linear in the graph
size but currently implemented inside the model cache, so it should run once per graph signature.

Implementation boundary: Add one model file, one model config, focused shape/registry tests, and
research notes. Do not modify trainers, evaluators, or dataset adapters for this first pass.

## Results

Implementation summary: Added `SepBottleneckGNN-lite`, a GPR-style node encoder with cached
undirected articulation/bridge markers. Articulation nodes receive a small learned token, and
separator edges receive a bounded learned residual message channel. Separator-only module
initialization restores the RNG state so the base GPR-style path preserves the seed-0 fallback
initialization.

Files changed:

- `src/gnn_gym/models/sep_bottleneck_gnn.py`
- `configs/models/sep_bottleneck_gnn.yaml`
- `tests/test_model_shapes.py`
- `tests/test_registry.py`
- `tests/test_sep_bottleneck_gnn.py`
- `results/tables/research_all_runs.csv`
- `results/tables/research_all_runs_by_config_mean_std.csv`
- `results/tables/research_all_runs_by_config_table.md`
- `results/tables/research_all_runs_by_config_table.tex`
- `results/tables/research_all_runs_by_model_mean_std.csv`
- `results/tables/research_all_runs_by_model_table.md`
- `results/tables/research_all_runs_by_model_table.tex`
- `research/experiments/sep_bottleneck_gnn.md`
- `research/AGENT_SCRATCHPAD.md`
- `research/INSIGHTS.md`

Command log:

```bash
uv run pytest tests/test_sep_bottleneck_gnn.py tests/test_model_shapes.py::test_model_output_shape tests/test_registry.py
uv run ruff check .
uv run pytest
uv run gnngym train --model sep_bottleneck_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym train --model sep_bottleneck_gnn --dataset cora --seed 0 --override training.max_epochs=50 --override training.patience=15
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 0 --override training.max_epochs=50 --override training.patience=15
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 0 --override training.max_epochs=200 --override training.patience=50
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 1 --override training.max_epochs=200 --override training.patience=50
uv run gnngym train --model sep_bottleneck_gnn --dataset pubmed --seed 2 --override training.max_epochs=200 --override training.patience=50
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Results table:

| Dataset | Budget | Seeds | architecture_config_hash | Val metric | Test metric | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| toy-node | 2 epochs / patience 2 | 0 | `69c50854` | 0.5000 | 0.3000 | Crash check passed; not evidence |
| Cora | 50 epochs / patience 15 | 0 | `b3a0442c` | 0.7580 | 0.7820 | Below confirmed GPR Cora mean 0.7827; not confirmed |
| PubMed | 50 epochs / patience 15 | 0 | `13e4e1a7` | 0.8080 | 0.7840 | Promising seed-0 signal; confirmed |
| PubMed | 200 epochs / patience 50 | 0,1,2 | `0f352f9d` | 0.8000 +/- 0.0053 | 0.7763 +/- 0.0117 | Near miss; does not beat confirmed GPR mean 0.8007 |

Comparison to baseline: The confirmed PubMed config-level mean validation metric was 0.8000, which
is slightly below the confirmed `gpr_gnn` PubMed validation mean 0.8007 from `research/INSIGHTS.md`.
Cora seed-0 did not approach the confirmed `gpr_gnn` Cora validation mean 0.7827, so Cora was not
confirmed.

Interpretation: Separator-aware routing did not produce a confirmed improvement over the current
GPR baseline. The mechanism may be neutral or weakly harmful on citation networks under this
shallow decoupled propagation setup, though PubMed was close enough to suggest the implementation is
not simply broken.

Failure modes: The initial two-layer/dropout-0.5 default was a brittle screen because early stopping
could halt during a long low-validation plateau. The final model config uses a stronger one-layer
decoupled fallback. The separator channel is still only a cheap articulation/bridge proxy and may
mark too many citation edges as separator-adjacent.

What this teaches us: Cheap separator-aware residual routing is not enough to outperform GPR on
PubMed after confirmation, and Cora does not show a seed-0 signal. The next useful test should
separate architecture value from dataset mismatch with a small synthetic bottleneck diagnostic.

Next recommended step: Add a `separator-bottleneck-node` synthetic diagnostic before spending more
Cora/PubMed budget on separator variants. If the mechanism cannot beat GPR/GCN on a task designed
around articulation-mediated label transfer, discard this line or redesign the separator channel.
