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
