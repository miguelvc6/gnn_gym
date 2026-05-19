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

## Research Notes

Architecture research is organized under `research/`:

- `research/PROGRAM.md` defines the agent-facing experiment protocol.
- `research/ARCHITECTURE_IDEAS.md` is the backlog of candidate GNN architecture ideas.
- `research/AGENT_SCRATCHPAD.md` tracks what has been tried and what to try next.
- `research/INSIGHTS.md` stores durable conclusions from experiments.
