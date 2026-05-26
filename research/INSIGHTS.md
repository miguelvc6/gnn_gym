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

## 2026-05-23 - Plain Encoder Versions Of Larger Belief Ideas Did Not Beat GPR

Evidence:

Bounded encoder-only implementations of Bethe-style node/edge consistency, dual-primal factor
states, equilibrium recurrence, region collapse, Kikuchi ego regions, loop correction, decimation,
walk-belief transformers, lightweight non-backtracking propagation, and confidence-gated APPNP were
run on Cora/PubMed with the fixed harness. None beat confirmed `gpr_gnn`; the best PubMed result in
this pass was `nb_light_gnn` at seed-0 validation `0.7900`, still below confirmed GPR `0.8007` and
confirmed `nb_belief_gnn` `0.7947`.

Implication:

The current Cora/PubMed benchmark is saturated by decoupled feature encoding plus learnable
propagation weights. More complex inference-inspired ideas likely need either a different task
family, explicit auxiliary objectives, or trainer support to be fairly tested.

Follow-up:

Treat `gpr_gnn` as the node-classification target. Reopen Bethe/RIGN/decimation-style ideas only
with a staged trainer-aware plan, and prefer graph-task or medium-scale checks for future novelty.

## 2026-05-22 - Learnable PageRank Propagation Beats APPNP And NB On Validation

Evidence:

`gpr_gnn` confirmed across seeds `0,1,2` with the 200-epoch budget. On Cora, the refined config
`hidden_channels=128`, `num_layers=1`, `dropout=0.1`, `propagation_steps=16`, `alpha=0.1`,
`lr=0.01`, and `weight_decay=0.0005` reached validation mean `0.7827`, above APPNP's `0.7753`.
On PubMed, `hidden_channels=64`, `num_layers=1`, `dropout=0.2`, `propagation_steps=10`,
`alpha=0.1`, `lr=0.01`, and `weight_decay=0.0005` reached validation mean `0.8007`, above
NB-belief's `0.7947` and APPNP's `0.7880`. Held-out test means were Cora `0.7757` for the refined
Cora config and PubMed `0.7807`; these are recorded for reporting, not selection.

Implication:

The best current node-classification direction is decoupled MLP encoding plus learnable propagation
weights. This keeps APPNP's stable propagation prior while giving the model enough freedom to tune
the hop mixture.

Follow-up:

Use the refined Cora config and original PubMed config as the current validation-selected GPR
settings. Establish matched `gpr_gnn` and APPNP baselines on `ogbn-arxiv` before making medium-scale
claims.

## 2026-05-22 - Non-Backtracking Belief GNN Improves PubMed Validation

Evidence:

`nb_belief_gnn` confirmed across seeds `0,1,2` on PubMed with `hidden_channels=128`,
`num_steps=2`, `dropout=0.2`, `training.lr=0.005`, and `weight_decay=0.0005`. Mean validation was
`0.7947`, above APPNP's `0.7880` and GAT's `0.7607`. Mean test was lower than APPNP (`0.7597`
versus `0.7727`) and is recorded only as held-out reporting.

Implication:

Directed non-backtracking edge-state propagation is a promising PubMed validation direction. The
current implementation is a small-dataset prototype, not a scalable OGB-products-ready encoder.

Follow-up:

Optimize reverse-edge indexing and compare against APPNP on PubMed with matched runtime before
promoting this beyond a PubMed-specific result.

## 2026-05-22 - GATv2 Improves Cora But Not PubMed

Evidence:

`gatv2` confirmed across seeds `0,1,2` with the 200-epoch budget. On Cora,
`hidden_channels=32`, `heads=8`, `num_layers=2`, `dropout=0.2`, `attention_dropout=0.2`, and
`lr=0.01` reached validation mean `0.7553` versus the original GAT baseline `0.7313`. On PubMed,
the best fast GATv2 candidate did not confirm, reaching validation mean `0.7547` versus GAT
`0.7607`.

Implication:

GATv2 is a useful Cora baseline-strengthening model, but APPNP remains the stronger current
Cora/PubMed architecture.

Follow-up:

Use GATv2 as an attention-family comparison point on Cora. Do not prioritize more PubMed GATv2
tuning before APPNP refinements.

## 2026-05-22 - APPNP Is Strong On Cora And PubMed

Evidence:

`appnp_net` confirmed across seeds `0,1,2` with the 200-epoch budget. On Cora,
`hidden_channels=128`, `num_layers=1`, `dropout=0.5`, `propagation_steps=10`, `alpha=0.1`,
`lr=0.01`, and `weight_decay=0.0005` reached validation mean `0.7753` versus GAT's `0.7313`.
On PubMed, `hidden_channels=64`, `num_layers=1`, `dropout=0.2`, `propagation_steps=5`,
`alpha=0.1`, `lr=0.01`, and `weight_decay=0.0005` reached validation mean `0.7880` versus GAT's
`0.7607` and JK-GCN's `0.7793`.

Implication:

Decoupling feature transformation from propagation is currently the strongest node-classification
direction in this repo. Further node architecture work should compare against APPNP, not only the
original GCN/GAT/GIN baselines.

Follow-up:

Refine APPNP's `alpha`, propagation steps, dropout, and learning rate on Cora/PubMed, then consider
a medium node dataset only if the improvement remains stable.

## 2026-05-22 - JK-GCN Improves PubMed Over The GAT Baseline

Evidence:

A 3-seed confirmation on PubMed with `jk_gcn`, `hidden_channels=64`, `num_layers=3`,
`dropout=0.5`, `jk_mode=cat`, `training.lr=0.01`, `training.weight_decay=0.0005`,
`max_epochs=200`, and `patience=50` improved mean validation accuracy from GAT's `0.7607` to
`0.7793`. Per-seed validation was `0.7820`, `0.7840`, and `0.7720`.

Implication:

Retaining intermediate GCN representations is a useful PubMed architecture direction. The result is
not a global node benchmark replacement yet because the same fast sweep did not improve Cora.

Follow-up:

Use JK-GCN as the confirmed PubMed comparison point for future node architecture ideas. Refine
`jk_mode`, weight decay, and dropout on PubMed before promoting a broader default.

## 2026-05-22 - GIN Cora Benefits From Wider Dropout-Regularized Hidden Layers

Evidence:

A 3-seed confirmation on Cora with `gin`, `hidden_channels=128`, `num_layers=3`, `dropout=0.5`,
`eps_trainable=true`, `training.lr=0.01`, `max_epochs=200`, and `patience=50` improved mean
validation accuracy from `0.6613` to `0.6773` versus the existing GIN baseline. Per-seed validation
was `0.6560`, `0.7040`, and `0.6720`.

Implication:

This is a Cora-specific GIN tuning candidate, not a global default change. The matched PubMed
candidate did not confirm, and the Cora candidate's mean test metric was lower, so future decisions
should keep using validation metric for selection and treat test as held-out reporting.

Follow-up:

Use this Cora GIN config as a confirmed comparison point for future Cora-only architecture checks,
but do not update `configs/models/gin.yaml` unless additional datasets confirm the change.

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
