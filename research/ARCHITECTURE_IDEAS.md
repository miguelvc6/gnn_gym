# Architecture Ideas

Use this file for human-authored or jointly-authored GNN architecture ideas before they become
code. Keep ideas concrete enough that an agent can turn one entry into a bounded experiment.

## Entry Template

```md
## YYYY-MM-DD - Short Name

Status: idea | queued | testing | kept | discarded
Target tasks: node | graph | link | all
Expected benefit:
Risk:
Minimal experiment:
Success criterion:

Description:

Implementation notes:
```

## Ideas

## 2026-05-19 - Residual Message Passing Baseline

Status: queued
Target tasks: node, graph
Expected benefit: Stabilize deeper GCN/GIN variants and make depth sweeps meaningful.
Risk: Extra skip/norm choices can obscure whether the convolution operator helped.
Minimal experiment: Add a residual wrapper around existing GCN/GIN layers with pre-norm and compare
against current GCN/GIN on `toy-node`, Cora, PubMed, and `toy-graph`.
Success criterion: Equal or better validation metric on at least two small datasets without worse
test metric on the others.

Description:

Add a conservative residual stack for homogeneous hidden dimensions:

- input projection to `hidden_channels`
- repeated message-passing block: norm, conv, activation, dropout, residual add
- output projection/head

Implementation notes:

Keep this as a new model, for example `res_gcn`, rather than changing the existing GCN baseline.

## 2026-05-19 - Dataset-Conditioned Pooling For Graph Tasks

Status: idea
Target tasks: graph
Expected benefit: Mean pooling can underperform on molecular and peptide tasks where graph size and
rare substructures matter.
Risk: Pooling changes are often small and noisy; evaluate over multiple seeds before keeping.
Minimal experiment: Add configurable concat pooling: `mean || max || add`, followed by a small MLP.
Success criterion: Better validation metric on `toy-graph` and at least one real graph dataset smoke
without adding trainer-specific model code.

Description:

Graph-level tasks should test whether richer global pooling improves signal capture.

Implementation notes:

This probably belongs in `src/gnn_gym/models/heads.py` or a reusable graph wrapper, not inside one
specific architecture.
