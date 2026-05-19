# AGENTS.md

## Project goal

This repository benchmarks graph neural network architectures across a fixed core suite of graph datasets.

The main design principle is that models, datasets, trainers, and evaluators are modular.

## Environment

Use uv.

Common commands:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run gnngym train --model gcn --dataset cora --seed 0
```

## Code rules

- New models go in `src/gnn_gym/models/`.
- Register new models with `@register_model("name")`.
- Do not modify trainers when adding a standard message-passing model unless necessary.
- Keep dataset-specific logic in `src/gnn_gym/data/`.
- Keep task-specific training logic in `src/gnn_gym/training/`.
- Every new model must include a shape test.
- Every new dataset adapter must include a smoke-loading test if feasible.

## Artifact rules

- Do not commit `data/`, `artifacts/`, `results/runs/`, or model checkpoints.
- Save final aggregated tables under `results/tables/`.
- Use deterministic seeds where possible.

## Research workflow

- Use `research/PROGRAM.md` for architecture research sessions.
- Put candidate architecture ideas in `research/ARCHITECTURE_IDEAS.md`.
- Use `research/AGENT_SCRATCHPAD.md` for trial notes and future attempts.
- Promote durable conclusions to `research/INSIGHTS.md`.
- During architecture experiments, prefer changing only model files, model configs, and focused tests.
- For autonomous hyperparameter search, iterate first on Cora and PubMed with short budgets before
  running larger OGB, Peptides, or link-prediction benchmarks.
- Optimize decisions on validation metrics only. Record test metrics, but do not choose configs by
  test performance.
- Keep a local uncommitted `research_results.tsv` ledger for experimental runs. Promote only
  curated summaries to `results/tables/`.
- Before long autonomous runs, prefer adding or using a sweep config/runner so each trial has
  reproducible overrides, run IDs, and aggregation metadata.
