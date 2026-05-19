# Shared Insights

This file is for durable lessons that should influence future GNN Gym work. Both human notes and
agent notes belong here, but entries should be evidence-backed.

## Entry Template

```md
## YYYY-MM-DD - Insight Title

Evidence:
Implication:
Follow-up:
```

## Insights

## 2026-05-19 - Separate The Research Harness From The Architecture Under Test

Evidence:

`autoresearch` is effective because it freezes data prep, dataloading, evaluation, metric reporting,
and run budget while allowing focused edits in one target file. GNN Gym has more task families than
autoresearch, so the equivalent control surface should be the model/config under test, not the whole
training pipeline.

Implication:

For architecture research, agents should usually modify only:

- `src/gnn_gym/models/<candidate>.py`
- `configs/models/<candidate>.yaml`
- focused shape/training tests

Trainer, evaluator, and dataset code should change only when the experiment explicitly requires a
new task capability.

Follow-up:

Create an experiment protocol file and command wrapper that makes this rule explicit.

## 2026-05-19 - Fixed Budgets Make Agent Experiments Easier To Compare

Evidence:

`autoresearch` compares LLM changes under a fixed wall-clock budget and one primary metric. GNN
benchmarks cannot collapse all tasks to one metric, but individual dataset/model/seed runs can still
use fixed epoch, patience, or wall-clock budgets.

Implication:

Use small fixed smoke budgets for agent iteration:

- `toy-node`, `toy-graph`, `toy-link`: 2-5 epochs
- Cora/PubMed: 20-50 epochs for quick signal
- MolHIV: 1-5 epochs for integration checks, longer only for serious evaluation

Follow-up:

Add `configs/experiments/research_smoke.yaml` and `research_core.yaml`.

## 2026-05-19 - Keep Human Ideas, Agent Scratchpad, And Conclusions Separate

Evidence:

`autoresearch/program.md` mixes instructions and experiment protocol in one file, which is fine for
a tiny repo but will get noisy in GNN Gym.

Implication:

GNN Gym should use three Markdown files:

- `ARCHITECTURE_IDEAS.md` for candidate ideas
- `AGENT_SCRATCHPAD.md` for trial state and next actions
- `INSIGHTS.md` for durable conclusions

Follow-up:

Keep these files small enough that an agent can read them every research session.
