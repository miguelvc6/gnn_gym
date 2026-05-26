# GNN Gym Research Program

This is the agent-facing research protocol inspired by `autoresearch/program.md`, adapted for GNN
architecture benchmarking.

For long-running architecture research, also read `research/LONG_RUNNING_RESEARCH.md`.

## Principle

The benchmark harness is the control. Architecture code is the variable.

Do not casually modify dataset adapters, trainers, evaluators, or aggregation when testing a new
architecture idea. Those changes make results harder to attribute.

## Evaluation Protocol For Architecture Claims

- Use `val_metric` as the selection metric.
- Record `test_metric`, but treat it as held-out reporting only.
- Treat seed `0` results as screening signals only.
- Confirm promising architecture/config results across seeds `[0, 1, 2]`.
- Make architecture claims from config-level aggregation grouped by `architecture_config_hash`.
- Do not claim a model beats a baseline from mixed-config model-level averages.
- Do not claim a model beats a baseline unless the exact confirmed config beats the baseline mean
  validation metric under the same budget.
- Toy datasets are crash checks only, never evidence for architecture quality.
- Do not compare across incompatible metrics as if they were one global score.

Task-appropriate comparison points:

- Cora/PubMed node classification: current confirmed `gpr_gnn`.
- Older node comparison points: GAT, APPNP, JK-GCN, and GATv2 when relevant.
- MolHIV graph prediction: current GIN/GCN/GAT/MLP baselines; add edge-aware baselines after
  edge-aware models are implemented on top of the edge-attribute plumbing.

## Novelty And Scientific Insight Standard

Hyperparameter tuning is useful engineering, not a novel architecture. Reimplementing known methods
such as APPNP, GPR-GNN, GATv2, GCNII, GraphSAGE, GINE, GPS, or Graphormer should be labeled as
baseline work unless there is a genuinely new mechanism. Hybridizing two known methods is not
automatically novel; it needs a specific mechanism and a falsifiable claim.

Each architecture idea should define:

- scientific hypothesis
- mechanism
- why this is not just a known baseline
- closest known related architectures
- expected insight if it succeeds
- expected insight if it fails
- target task family
- primary baseline to beat
- minimal falsifying experiment
- confirmation protocol
- complexity/runtime risk
- implementation boundary

Failed architectures should remain in the research memory when they clarify the hypothesis space.
Do not hide negative results.

## Long-Running Loop

1. Read this protocol, `research/LONG_RUNNING_RESEARCH.md`, `research/INSIGHTS.md`, and the active
   scratchpad.
2. Pick or add one architecture idea with a scientific hypothesis.
3. Check closest known baselines before calling it novel.
4. Implement the smallest bounded version.
5. Run toy crash checks.
6. Run a seed-0 fast screen.
7. Confirm only promising configs over seeds `[0, 1, 2]`.
8. Aggregate by `architecture_config_hash`.
9. Promote only durable conclusions to `research/INSIGHTS.md`.
10. Keep negative results if they clarify the hypothesis space.
11. Periodically add new ideas, but avoid repeating discarded mechanisms without a new reason.

## Setup For A Research Session

1. Read:
   - `README.md`
   - `AGENTS.md`
   - `GNN_GYM_PROJECT_SPEC.md`
   - `research/LONG_RUNNING_RESEARCH.md`
   - `research/ARCHITECTURE_IDEAS.md`
   - `research/AGENT_SCRATCHPAD.md`
   - `research/INSIGHTS.md`
2. Check git state.
3. Pick one idea from `research/ARCHITECTURE_IDEAS.md`.
4. Define the smallest commands that can falsify it.
5. Record the plan in `research/AGENT_SCRATCHPAD.md`.

For hyperparameter-only work on an existing architecture, the "idea" can be a bounded tuning goal
such as "improve GCN/GAT/GIN validation metrics on Cora and PubMed." Do not mix architecture
changes and hyperparameter sweeps in the same experimental batch unless the scratchpad explicitly
states why.

## Allowed Edits During Architecture Experiments

Usually allowed:

- `src/gnn_gym/models/<new_model>.py`
- `configs/models/<new_model>.yaml`
- `tests/test_model_shapes.py`
- focused tests for the new model
- `research/AGENT_SCRATCHPAD.md`
- `research/INSIGHTS.md` when a conclusion is durable

Avoid unless the idea requires it:

- `src/gnn_gym/training/`
- `src/gnn_gym/data/`
- `src/gnn_gym/evaluation/`
- global config defaults

## Smoke Commands

Use these before any real benchmark:

```bash
uv run ruff check .
uv run pytest
uv run gnngym train --model gcn --dataset toy-node --seed 0 --override training.max_epochs=2
uv run gnngym train --model gin --dataset toy-graph --seed 0 --override training.max_epochs=2
uv run gnngym train --model gcn --dataset toy-link --seed 0 --override training.max_epochs=2
```

Toy datasets are for crash checks only. Do not keep a hyperparameter setting because it improves a
toy metric.

## Fast Hyperparameter Search Protocol

Use this protocol when tuning already implemented models such as `gcn`, `gat`, and `gin`.

### Dataset Tiers

Tier 0: crash checks

- `toy-node`: node trainer and shape checks.
- `toy-graph`: graph trainer checks.
- `toy-link`: link trainer checks.

Tier 1: fast real iteration

- `cora`: primary fast node-classification target.
- `pubmed`: secondary node-classification target.

Tier 2: slower validation

- `ogbg-molhiv`: first real graph-prediction check, especially for `gin`.
- `ogbn-arxiv`: medium node-classification check once Cora/PubMed improve.

Avoid during fast autonomous iteration unless explicitly requested:

- `ogbn-products`
- Peptides datasets
- `ogbl-collab`

Those are larger benchmark runs, not first-pass search datasets.

### Metrics And Decisions

- Use `val_metric` as the optimization target.
- Record `test_metric`, but do not select hyperparameters using test performance.
- After a promising config is found with seed `0`, confirm it with seeds `[0, 1, 2]`.
- Compare confirmed configs using summaries grouped by `architecture_config_hash`, not mixed-config
  model-level means.
- Promote a config only when it improves mean validation metric or gives similar validation metric
  with simpler/cheaper settings.

### Fast Budgets

Use short budgets for search:

```text
Cora/PubMed:
  max_epochs: 50
  patience: 15
  seed: 0

MolHIV smoke:
  max_epochs: 3-5
  patience: 3-5
  seed: 0
```

Use confirmation budgets for candidates:

```text
Cora/PubMed:
  max_epochs: 200-300
  patience: 50
  seeds: [0, 1, 2]
```

### Initial Search Spaces

Start with compact, hand-curated sweeps rather than huge grids.

Shared knobs:

- `model.hidden_channels`
- `model.num_layers`
- `model.dropout`
- `model.norm`
- `training.lr`
- `training.weight_decay`

GCN first pass:

```text
hidden_channels: [32, 64, 128, 256]
num_layers: [2, 3, 4]
dropout: [0.2, 0.5, 0.7]
lr: [0.01, 0.005, 0.001]
weight_decay: [0.0005, 0.0001, 0.0]
```

GAT first pass:

```text
hidden_channels: [8, 16, 32, 64]
heads: [2, 4, 8]
num_layers: [2, 3]
dropout: [0.2, 0.5]
attention_dropout: [0.0, 0.2, 0.5]
lr: [0.01, 0.005, 0.001]
```

GIN first pass:

```text
hidden_channels: [32, 64, 128]
num_layers: [2, 3, 5]
dropout: [0.2, 0.5]
eps_trainable: [true, false]
lr: [0.01, 0.005, 0.001]
```

Do not run the full Cartesian product by default. Pick a compact subset, inspect results, then
refine.

## First Real Checks

For node models:

```bash
uv run gnngym train --model <model> --dataset cora --seed 0 --override training.max_epochs=50
uv run gnngym train --model <model> --dataset pubmed --seed 0 --override training.max_epochs=50
```

For graph models:

```bash
uv run gnngym train --model <model> --dataset ogbg-molhiv --seed 0 --override training.max_epochs=5
```

For link models:

```bash
uv run gnngym train --model <model> --dataset toy-link --seed 0 --override training.max_epochs=5
```

## Result Logging

For exploratory research, use a local uncommitted TSV named `research_results.tsv`:

```text
timestamp	run_id	commit	model	dataset	seed	metric	val_metric	test_metric	best_epoch	train_time_seconds	status	overrides	notes
```

Keep `results/runs/` out of git. Promote only intentional summaries under `results/tables/`.
Use `*_by_config_mean_std.csv` for architecture/config evidence. Treat `*_by_model_mean_std.csv`
as a mixed-config diagnostic table only.

For each research batch, also add a short section to `research/AGENT_SCRATCHPAD.md` with:

- goal
- dataset tier
- search space
- commands or sweep config used
- best runs
- discarded ideas
- next actions

Promote only stable conclusions to `research/INSIGHTS.md`.

## Sweep Workflow

Manual CLI overrides are fine for one or two runs:

```bash
uv run gnngym train \
  --model gcn \
  --dataset cora \
  --seed 0 \
  --override model.hidden_channels=128 \
  --override model.num_layers=3 \
  --override model.dropout=0.5 \
  --override training.lr=0.005
```

For autonomous experimentation, prefer a sweep config and runner so runs are reproducible and
aggregatable. If sweep support is not implemented yet, implement the minimum useful version before
launching a long search:

```yaml
experiment:
  name: node_hparam_search
  seeds: [0]

models:
  - gcn
  - gat
  - gin

datasets:
  - cora
  - pubmed

training:
  max_epochs: 50
  patience: 15

sweep:
  model.hidden_channels: [32, 64, 128]
  model.num_layers: [2, 3]
  model.dropout: [0.2, 0.5]
  training.lr: [0.01, 0.005]
```

Expected command shape:

```bash
uv run gnngym run-sweep --config configs/experiments/node_hparam_search.yaml
```

The sweep runner should expand override combinations, run each model/dataset/seed, append
`research_results.tsv`, and leave full artifacts in `results/runs/`.

## Keep Or Discard Rule

Keep a change when it improves the relevant validation metric without obvious regressions or when it
simplifies the code with no metric loss.

Discard a change when it:

- only improves toy data
- requires trainer/evaluator hacks for a standard architecture
- adds complexity without a clear metric or maintainability benefit
- breaks another task family

For hyperparameter sweeps, do not update default model YAMLs after a single seed. First confirm the
candidate over multiple seeds and at least Cora plus PubMed, unless the setting is clearly a
resource/speed improvement with no metric loss.
