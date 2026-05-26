# GNN Gym

GNN Gym is a reproducible benchmarking framework for graph neural network research.
It is designed to make graph experiments repeatable, comparable, and easy to extend:
add a model, register it, give it a config, run it across a fixed dataset suite, and
aggregate the resulting metrics across seeds.

The central idea is to keep the benchmark harness stable while changing only the
architecture under test. Dataset loading, task-specific training, evaluation, logging,
checkpointing, and aggregation are reusable components.

```text
dataset adapter + model encoder + task trainer + evaluator
```

This separation lets the same CLI run node classification, graph prediction, and link
prediction experiments without creating one-off scripts for each model or dataset.

## Project Status

The repository currently includes the core benchmark scaffold:

- registries for models, datasets, trainers, and evaluators
- YAML-driven run configuration
- a `gnngym` CLI for training, experiment batches, evaluation, aggregation, and table export
- toy node, graph, and link datasets for smoke testing
- baseline `mlp`, `gcn`, `gat`, and `gin` models
- full-batch node, neighbor-sampling node, graph-prediction, and link-prediction trainers
- configs for the intended core benchmark suite
- local run artifacts under `results/runs/` and aggregated tables under `results/tables/`

The long-term benchmark target is described in
[`GNN_GYM_PROJECT_SPEC.md`](GNN_GYM_PROJECT_SPEC.md). That file is the product contract
for the repository: it defines the architecture, dataset suite, implementation phases,
accepted commands, and rules for extending the system.

## Quickstart

Use `uv` for environment and command execution.

```bash
uv sync
uv run pytest
uv run ruff check .
```

Run a single training job:

```bash
uv run gnngym train --model gcn --dataset toy-node --seed 0
```

Run a small experiment batch:

```bash
uv run gnngym run-experiment --config configs/experiments/smoke_test.yaml
```

Aggregate local runs:

```bash
uv run gnngym aggregate --runs results/runs --out results/tables/all_runs.csv
uv run gnngym export-tables --input results/tables/all_runs.csv
```

`export-tables` writes config-level summaries grouped by `architecture_config_hash` and separate
model-level summaries. Use the config-level files for architecture claims; model-level summaries can
mix many hyperparameter settings and are diagnostics only.

Example with overrides:

```bash
uv run gnngym train \
  --model gcn \
  --dataset cora \
  --seed 0 \
  --override training.max_epochs=50 \
  --override model.hidden_channels=128 \
  --override model.dropout=0.5
```

## Core Benchmark Suite

GNN Gym is organized around a fixed suite of graph tasks:

- Node classification: Cora, PubMed, ogbn-arxiv, ogbn-products, Roman-empire,
  Amazon-ratings
- Graph prediction: ogbg-molhiv, ogbg-molpcba, Peptides-func, Peptides-struct
- Link prediction: ogbl-collab

Toy datasets exist only for crash checks and trainer smoke tests. Architecture or
hyperparameter decisions should be based on validation metrics from real datasets,
not toy metrics.

## Adding A Model

Standard message-passing models should usually require only:

```text
src/gnn_gym/models/<model_name>.py
configs/models/<model_name>.yaml
tests/test_model_shapes.py
```

Register new models with `@register_model("name")`. Avoid changing trainers,
dataset adapters, or evaluators unless the architecture genuinely requires a new
task capability. This keeps benchmark results attributable to the model change.

## Automated Research Workflow

GNN Gym is built to support agent-assisted architecture research. The intended process
is documented in [`research/PROGRAM.md`](research/PROGRAM.md) and
[`research/LONG_RUNNING_RESEARCH.md`](research/LONG_RUNNING_RESEARCH.md). The agent treats a
small set of Markdown files as persistent research memory.

At the start of a research session, the agent reads:

- [`README.md`](README.md) for the current project overview and command surface
- [`AGENTS.md`](AGENTS.md) for repository-specific operating rules
- [`GNN_GYM_PROJECT_SPEC.md`](GNN_GYM_PROJECT_SPEC.md) for the long-term benchmark design
- [`research/PROGRAM.md`](research/PROGRAM.md) for the experiment protocol
- [`research/LONG_RUNNING_RESEARCH.md`](research/LONG_RUNNING_RESEARCH.md) for config-level
  evidence rules and novelty standards
- [`research/ARCHITECTURE_IDEAS.md`](research/ARCHITECTURE_IDEAS.md) for candidate ideas
- [`research/AGENT_SCRATCHPAD.md`](research/AGENT_SCRATCHPAD.md) for recent trial state
- [`research/INSIGHTS.md`](research/INSIGHTS.md) for durable conclusions

The logic is:

1. Use `GNN_GYM_PROJECT_SPEC.md` to understand the invariant benchmark architecture:
   model registry, dataset adapters, trainers, evaluators, result layout, and accepted
   benchmark phases.
2. Use `research/ARCHITECTURE_IDEAS.md` as the backlog of candidate architecture changes.
   Ideas should be concrete enough to become bounded experiments.
3. Use `research/PROGRAM.md` to choose the smallest falsifying experiment. The harness is
   the control; architecture code and model config are the variables.
4. Use `research/AGENT_SCRATCHPAD.md` for working notes: plan, commands, run IDs, failures,
   discarded attempts, and next actions.
5. Use validation metrics for decisions. Test metrics are recorded, but not used to select
   configurations.
6. Use `architecture_config_hash` summaries for architecture claims. Do not select from mixed-config
   model-level averages.
7. Promote only evidence-backed conclusions to `research/INSIGHTS.md`.
8. Promote only curated result summaries to `results/tables/`; keep raw runs, checkpoints,
   and exploratory ledgers local.

For new architecture experiments, the preferred edit boundary is:

```text
src/gnn_gym/models/<candidate>.py
configs/models/<candidate>.yaml
tests/test_model_shapes.py
research/AGENT_SCRATCHPAD.md
research/INSIGHTS.md
```

For autonomous hyperparameter search, the agent should start with short budgets on Cora
and PubMed, confirm promising settings across multiple seeds, and avoid large OGB,
Peptides, or link-prediction runs until there is a clear reason.

## Research Commands

Crash checks before real benchmarks:

```bash
uv run ruff check .
uv run pytest
uv run gnngym train --model gcn --dataset toy-node --seed 0 --override training.max_epochs=2
uv run gnngym train --model gin --dataset toy-graph --seed 0 --override training.max_epochs=2
uv run gnngym train --model gcn --dataset toy-link --seed 0 --override training.max_epochs=2
```

First real checks for node models:

```bash
uv run gnngym train --model gcn --dataset cora --seed 0 --override training.max_epochs=50
uv run gnngym train --model gcn --dataset pubmed --seed 0 --override training.max_epochs=50
```

Full benchmark configs live in `configs/experiments/`, including:

- `smoke_test.yaml`
- `core_gcn.yaml`
- `core_gat.yaml`
- `core_gin.yaml`
- `core_all_baselines.yaml`

## Artifacts

Generated data, checkpoints, raw run directories, and exploratory ledgers should remain
local. Final aggregated tables belong under:

```text
results/tables/
```

Do not commit:

```text
data/
artifacts/
results/runs/
model checkpoints
research_results.tsv
```
