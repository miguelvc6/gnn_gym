# Agent Scratchpad

This file is for working memory across research sessions: what was tried, what happened, what is worth trying next, and what to avoid repeating. It is allowed to be messy, but every entry should be dated and actionable.

Do not use this file for final claims. Promote durable conclusions to `research/INSIGHTS.md`.

## Current Benchmark Discipline

- Keep the harness fixed while evaluating an architecture idea.
- Prefer `toy-node`, `toy-graph`, and `toy-link` for crash checks.
- Use Cora/PubMed as the first real node-classification checks.
- Use `ogbg-molhiv` as the first real graph-prediction check.
- Avoid running `ogbn-products`, Peptides, or `ogbl-collab` during casual iteration unless the user
  explicitly wants a long run.

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

## Tried

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
