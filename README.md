# GNN Gym

GNN Gym is a reproducible benchmarking framework for graph neural network
architectures. The initial implementation focuses on a Phase 1 smoke-test
pipeline: registries, config loading, a synthetic node-classification dataset,
MLP/GCN baselines, local artifacts, and result aggregation.

## Quickstart

```bash
uv sync
uv run pytest
uv run gnngym train --model gcn --dataset toy-node --seed 0
uv run gnngym aggregate --runs results/runs --out results/tables/toy_runs.csv
```
