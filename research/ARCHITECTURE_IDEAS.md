# Architecture Ideas

Use this file for human-authored or jointly-authored GNN architecture ideas before they become
code. Keep ideas concrete enough that an agent can turn one entry into a bounded experiment.

## Entry Template

```md
## YYYY-MM-DD - Short Name

Status: idea | queued | testing | kept | discarded
Target tasks: node | graph | link | all
Scientific hypothesis:
Mechanism:
Why this is not just a known baseline:
Closest known related architectures:
Expected insight if it succeeds:
Expected insight if it fails:
Target task family:
Primary baseline to beat:
Minimal falsifying experiment:
Confirmation protocol:
Complexity/runtime risk:
Implementation boundary:

Description:

Implementation notes:
```

## Novelty And Evidence Standard

- Hyperparameter tuning is engineering work, not a novel architecture.
- Known methods such as APPNP, GPR-GNN, GATv2, GCNII, GraphSAGE, GINE, GPS, and Graphormer should
  be labeled as baselines unless the idea adds a genuinely new mechanism.
- Hybridizing two known methods is not automatically novel; the entry must name the mechanism and
  the falsifiable claim.
- Toy datasets are crash checks only.
- Architecture claims require config-level aggregation by `architecture_config_hash` and
  confirmation over seeds `[0, 1, 2]`.
- Negative results should remain here when they clarify what mechanism was tested and why it failed.

## Ideas

## 2026-05-23 - Bethe-Style Node/Edge Consistency GNN

Status: discarded
Target tasks: node
Expected benefit: Represent both node and edge beliefs and let edge states push locally consistent
node updates without trainer-level auxiliary losses.
Risk: Without an explicit Bethe consistency loss, the edge state can become noisy extra compute.
Minimal experiment: Add `bethe_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed or show a clear PubMed signal.

Description:

Maintain edge beliefs from endpoint node states, update edge beliefs recurrently, and update node
states from incoming/outgoing edge belief summaries.

First result:

Fast sweeps were far below GPR: best validation was Cora `0.4580` and PubMed `0.5580`. Discard this
no-auxiliary-loss Bethe formulation.

## 2026-05-23 - Kikuchi Ego-Region GNN

Status: discarded
Target tasks: node
Expected benefit: Test a bounded region-message variant where each node's ego neighborhood forms a
region state that sends messages back to member nodes.
Risk: The ego-region approximation may collapse into an expensive local smoother.
Minimal experiment: Add `kikuchi_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed.

Description:

Pool neighbor states into an ego-region representation for every node, then aggregate neighboring
region states back to nodes through a residual update.

First result:

Fast sweeps were not competitive: best validation was Cora `0.5460` and PubMed `0.6960`. Discard
this bounded ego-region version for the citation-node benchmark.

## 2026-05-23 - Loop-Corrected GNN

Status: discarded
Target tasks: node
Expected benefit: Treat short cycles as first-class correction objects instead of relying on repeated
edge-local message passing to infer loop structure.
Risk: Triangle enumeration adds overhead and may not help homophilous citation graphs.
Minimal experiment: Add `loop_corrected_gnn` with capped triangle correction, run toy-node, then
compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed or show a strong Cora signal.

Description:

Run GCN layers and, at each layer, compute capped triangle embeddings from current node states and
scatter loop corrections back to participating nodes.

First result:

Fast sweeps were weak and slower on PubMed: best validation was Cora `0.4060` and PubMed `0.6440`.
Discard this capped triangle-correction version.

## 2026-05-23 - Graph Decimation GNN

Status: discarded
Target tasks: node
Expected benefit: Approximate learned decimation by repeatedly clamping high-confidence hidden
states and propagating from that clamped state.
Risk: Without pseudo-label supervision or trainer support, confidence clamping may amplify bad
hidden states.
Minimal experiment: Add `decimation_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed.

Description:

Predict hidden-state confidence each round, maintain a detached clamped hidden state, and update the
current node state from graph messages over the current plus clamped representation.

First result:

Fast sweeps did not approach GPR: best validation was Cora `0.6020` and PubMed `0.7520`. Discard
this trainer-free decimation approximation.

## 2026-05-23 - Walk-Belief Transformer

Status: discarded
Target tasks: node
Expected benefit: Let each node read a deterministic short walk with a small transformer, providing
path-sequence context outside synchronous edge aggregation.
Risk: Deterministic walks can be brittle and may miss useful graph neighborhoods.
Minimal experiment: Add `walk_belief_transformer`, run toy-node, then compact Cora/PubMed fast
sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed.

Description:

Choose a deterministic successor for every node, build a short walk from each node, encode walk
tokens with a small transformer, and fuse the resulting path state back into the node embedding.

First result:

Fast sweeps were stable but below GPR: best validation was Cora `0.6640` and PubMed `0.7200`.
Discard this deterministic-walk first pass.

## 2026-05-23 - Dual-Primal Factor GNN

Status: discarded
Target tasks: node
Expected benefit: Promote edges to explicit factor states and run coupled node/edge updates without
changing the benchmark harness.
Risk: Edge/factor states can overfit or add unstable compute on small citation graphs.
Minimal experiment: Add `dual_primal_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed.

Description:

Maintain primal node states and dual edge states. Update edge states from endpoint nodes and
edge-neighborhood summaries, then update node states from incoming/outgoing edge factors.

First result:

Fast sweeps were poor: best validation was Cora `0.5020` and PubMed `0.6620`. Discard this bounded
dual-primal node variant.

## 2026-05-23 - Neural Region Collapse GNN

Status: discarded
Target tasks: node
Expected benefit: Learn soft global regions, reason over region states, and scatter region context
back to nodes to approximate graph coarsening.
Risk: Global region mixing may wash out local label structure in citation graphs.
Minimal experiment: Add `region_collapse_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed.

Description:

Learn node-to-region assignments, pool node states into a fixed number of soft regions, transform
region states, and scatter region context back to nodes alongside local GCN messages.

First result:

Best fast validation was Cora `0.6520` and PubMed `0.7520`, below GPR. Discard this node-focused
region-collapse first pass.

## 2026-05-23 - Equilibrium Belief GNN

Status: discarded
Target tasks: node
Expected benefit: Reuse one recurrent graph update toward a fixed point, approximating iterative
belief convergence without implicit differentiation.
Risk: Recurrent updates can converge to uninformative fixed points or oversmooth quickly.
Minimal experiment: Add `equilibrium_belief_gnn`, run toy-node, then compact Cora/PubMed fast
sweeps.
Success criterion: Beat confirmed GPR validation on Cora/PubMed.

Description:

Project node features once, then repeatedly apply a normalized graph update and GRU hidden-state
update for a fixed number of steps, with early stopping only in eval mode.

First result:

Fast sweeps collapsed badly: best validation was Cora `0.3160` and PubMed `0.4780`. Discard this
unrolled equilibrium version.

## 2026-05-22 - Cavity GNN

Status: discarded
Target tasks: node
Expected benefit: Keep the successful non-backtracking directed-edge principle while replacing the
plain edge-state MLP with a recurrent cavity update that can condition each directed message on the
sender, receiver, and incoming non-reverse context.
Risk: Extra recurrence can overfit or simply add compute over the simpler `nb_belief_gnn`.
Minimal experiment: Add `cavity_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed APPNP on Cora or confirmed NB-belief on PubMed by validation
metric, or show a cheaper/simpler tie.

Description:

Maintain directed edge messages. For each edge `i -> j`, aggregate incoming messages to `i` while
subtracting the reverse `j -> i` message, then update the directed message with a GRU cell.

Implementation notes:

This is the bounded version of `cavity_gnn` from
`research/original-ideas/belief-propagation_design_mine.md`. It intentionally reuses the fixed
node-classification trainer and returns only node embeddings.

First result:

Fast Cora/PubMed sweeps completed cleanly but failed badly: best validation was only `0.3340` on
Cora and `0.6720` on PubMed. The GRU cavity update is discarded as an overcomplicated variant of
the simpler non-backtracking model.

## 2026-05-22 - SurveyGNN Multi-Belief Particles

Status: discarded
Target tasks: node
Expected benefit: Let each node maintain several latent belief particles instead of one embedding,
then learn how to read out the most useful mixture.
Risk: Particles may collapse without a diversity objective, and compute scales with particle count.
Minimal experiment: Add `survey_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed APPNP or NB-belief validation on Cora/PubMed, or show enough
signal to justify a diversity loss.

Description:

Initialize multiple particles per node, update each particle with graph messages plus a consensus
exchange term, then read out an attention-weighted particle mixture.

Implementation notes:

This is a bounded encoder-only version of `survey_gnn` from
`research/original-ideas/belief-propagation_design_mine.md`.

First result:

Fast Cora/PubMed sweeps completed cleanly but were not competitive. Best validation was `0.6960`
on Cora and `0.7760` on PubMed, below confirmed APPNP on both datasets and below confirmed
NB-belief on PubMed. Discard this particle-only first pass unless reopening with an explicit
diversity objective or particle-collapse diagnostic.

## 2026-05-22 - Temperature-Ladder GNN

Status: discarded
Target tasks: node
Expected benefit: Combine sharp and smooth attention streams so the model can mix local selective
messages with broader smoothing behavior.
Risk: May act like an expensive attention ensemble and underperform APPNP propagation.
Minimal experiment: Add `temp_ladder_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GATv2, APPNP, or NB-belief validation on Cora/PubMed.

Description:

Run parallel attention streams with different softmax temperatures in each layer, concatenate their
outputs, and feed the existing task head.

Implementation notes:

This is a bounded version of the original temperature-ladder idea using custom per-destination
attention and fixed temperature lists.

First result:

Fast Cora/PubMed sweeps were not competitive (`0.6100` best Cora validation, `0.7340` best PubMed
validation), so this first pass is discarded.

## 2026-05-22 - Minimal RIGN Encoder

Status: discarded
Target tasks: node
Expected benefit: Test the recursive-improvement idea with separate solution (`y`) and latent
reasoning (`z`) states without changing trainers or adding deep-supervision losses.
Risk: Without the full RIGN objective, recurrence may become dead compute or just an expensive GCN.
Minimal experiment: Add `rign_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed APPNP or NB-belief validation on Cora/PubMed, or show enough
signal to justify a fuller RIGN implementation with trainer support.

Description:

Maintain a solution state and latent reasoning state. Each macro-step updates the latent state
through recurrent graph micro-steps, then applies a gated residual correction to the solution state.

Implementation notes:

This is a bounded encoder-only pass mined from
`research/original-ideas/recursive_improvement_graph_network_project.md`; no deep supervision or
improvement loss is added in this first test.

First result:

Fast Cora/PubMed sweeps were far below the confirmed APPNP and NB-belief baselines (`0.5680` best
Cora validation, `0.7060` best PubMed validation). Discard this encoder-only shortcut; a future
RIGN attempt should include the intended deep supervision or improvement objective.

## 2026-05-22 - Entropy-Gated GNN

Status: discarded
Target tasks: node
Expected benefit: Let uncertain receivers listen more strongly to confident senders, using a
learned uncertainty gate inspired by belief strength.
Risk: The uncertainty gate can collapse without explicit calibration supervision.
Minimal experiment: Add `entropy_gated_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GATv2, APPNP, or NB-belief validation on Cora/PubMed.

Description:

Each node predicts a scalar uncertainty. Edge messages are weighted by confident sender times
uncertain receiver before being aggregated into a residual node update.

Implementation notes:

This is a bounded encoder-only pass of the original `entropy_gated_gnn` idea; no logging,
calibration loss, or trainer hooks.

First result:

Fast Cora/PubMed sweeps did not beat confirmed APPNP or NB-belief baselines. Best validation was
`0.6940` on Cora and `0.7820` on PubMed, so this first pass is discarded.

## 2026-05-22 - Non-Backtracking Belief GNN

Status: kept
Target tasks: node
Expected benefit: Reduce immediate message echo by maintaining directed edge beliefs and updating
each edge from incoming messages that exclude the reverse edge.
Risk: Edge-state memory and Python reverse-edge lookup are acceptable for Cora/PubMed but not a
large-graph implementation.
Minimal experiment: Add `nb_belief_gnn`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed GATv2 or APPNP validation on Cora/PubMed, or show enough promise
to justify an optimized edge-state implementation.

Description:

Initialize directed edge messages from source node features, repeatedly aggregate incoming messages
to the source node while subtracting the reverse edge, then read out node states from incoming
directed messages.

Implementation notes:

This is the bounded first pass of `nb_belief_gnn` from
`research/original-ideas/belief-propagation_design_mine.md`; it is intentionally scoped to small
node datasets before any optimized sparse edge-state work.

First result:

PubMed confirmed across seeds `0,1,2` with validation mean `0.7947`, beating APPNP `0.7880`. Cora
did not improve. Keep as a PubMed-specific validation result and optimize only if runtime matters.

## 2026-05-22 - Frustration-Aware GNN

Status: discarded
Target tasks: node
Expected benefit: Split neighbor evidence into learned agreement and conflict channels, avoiding
the assumption that all edges should smooth node states in the same direction.
Risk: The frustration gate is unsupervised and may become noisy on homophilous citation graphs.
Minimal experiment: Add `frustration_gnn` as a node encoder, run toy-node, then compact Cora/PubMed
fast sweeps.
Success criterion: Beat confirmed GATv2 or APPNP validation on Cora/PubMed, or show a sufficiently
strong signal to justify trying heterophily datasets later.

Description:

For each edge, learn a frustration score from source/target hidden states, split incoming messages
into agreement and conflict channels, then update each node with both channels through a residual
MLP.

Implementation notes:

This is the bounded first pass of the `frustration_gnn` idea from
`research/original-ideas/belief-propagation_design_mine.md`; no auxiliary losses or trainer hooks.

First result:

Fast Cora/PubMed sweeps did not approach the confirmed APPNP or GATv2 baselines. Best validation was
`0.6840` on Cora and `0.7460` on PubMed, so this first pass is discarded for homophilous citation
graphs. It may still be relevant for heterophily-focused work.

## 2026-05-22 - GATv2 Attention Baseline

Status: kept
Target tasks: node
Expected benefit: Test whether dynamic attention ranking improves over the current GAT baseline on
Cora/PubMed without changing training code.
Risk: May be slower or more parameter-heavy without improving validation metrics.
Minimal experiment: Add `gatv2`, run toy-node, then compact Cora/PubMed sweeps near the existing
GAT search space.
Success criterion: Beat confirmed GAT validation on Cora or PubMed.

Description:

Replace GATConv with PyG GATv2Conv while keeping the same encoder/head interface and comparable
hidden/head/dropout knobs.

Implementation notes:

Treat this as a baseline-strengthening experiment rather than a new research direction unless it
beats APPNP.

First result:

Confirmed on Cora with validation mean `0.7553`, beating original GAT `0.7313`. PubMed did not
confirm versus GAT.

## 2026-05-22 - GCNII-Style Initial Residual Network

Status: discarded
Target tasks: node
Expected benefit: Allow deeper GCN-style propagation while preserving the original transformed
features through initial residual connections and identity mapping.
Risk: More depth may be unnecessary on Cora/PubMed, and the model may be sensitive to `alpha` and
`theta`.
Minimal experiment: Add `gcn2_net` using PyG `GCN2Conv`, run toy-node, then compact Cora/PubMed
fast sweeps.
Success criterion: Beat confirmed GAT, JK-GCN, or APPNP validation on at least one primary dataset.

Description:

Project input features once, keep that projection as the initial representation, and apply repeated
GCN2Conv layers that mix current hidden states with the initial representation.

Implementation notes:

Keep the task head unchanged; tune depth, dropout, `alpha`, and `theta` before considering any
larger dataset.

First result:

Fast Cora/PubMed sweeps were far below GAT and APPNP (`0.4880` best Cora validation, `0.6260` best
PubMed validation), so this implementation/search direction is discarded for now.

## 2026-05-22 - APPNP Propagation Network

Status: kept
Target tasks: node
Expected benefit: Separate feature transformation from graph propagation, which can help citation
graphs when last-layer message passing oversmooths or under-propagates labels.
Risk: APPNP may mostly act like a tuned propagation baseline and may be sensitive to `alpha`.
Minimal experiment: Add `appnp_net` as an MLP encoder followed by PyG APPNP propagation, run
toy-node, then compact Cora/PubMed sweeps under the fast budget.
Success criterion: Beat confirmed GAT or JK-GCN validation on Cora/PubMed without trainer changes.

Description:

Apply a small MLP to node features, then run personalized propagation with configurable propagation
steps and teleport probability before the standard task head.

Implementation notes:

Keep the propagated hidden representation as the encoder output so graph and node heads remain
unchanged.

First result:

Confirmed across seeds `0,1,2`: Cora validation mean `0.7753` and PubMed validation mean `0.7880`,
beating confirmed GAT on both datasets and confirmed JK-GCN on PubMed.

## 2026-05-22 - Jumping-Knowledge GCN

Status: kept
Target tasks: node
Expected benefit: Improve weak GCN baselines by exposing shallow and deeper representations to the
task head instead of relying only on the last message-passing layer.
Risk: Concatenation can inflate parameter count and may only memorize Cora-scale datasets.
Minimal experiment: Add `jk_gcn` as a separate model with configurable JK mode, run toy-node, then
run compact Cora/PubMed seed-0 sweeps under the same fast budget.
Success criterion: Improve validation over GCN and approach or beat GAT on at least one primary
dataset without trainer/evaluator changes.

Description:

Stack normal GCN layers but collect every hidden state and combine them with PyG JumpingKnowledge
before the existing task head.

Implementation notes:

Start with `jk_mode=cat` and fixed benchmark heads. If cat is promising but costly, compare `max`
mode later as a cheaper variant.

First result:

The PubMed candidate `hidden=64,layers=3,dropout=0.5,jk_mode=cat,lr=0.01` confirmed across seeds
`0,1,2` with validation mean `0.7793`, beating the confirmed GAT PubMed baseline `0.7607`. Cora did
not improve in the fast sweep.

## 2026-05-22 - Revision GNN

Status: discarded
Target tasks: node
Expected benefit: Preserve strong feature-only beliefs while letting graph evidence make gated
residual corrections, reducing damage when neighbor evidence is noisy.
Risk: May collapse to a residual GCN unless the learned revision gate adds useful selectivity.
Minimal experiment: Add `revision_gnn` as a new model, run a toy-node crash check, then run seed `0`
on Cora/PubMed with the same 50-epoch fast budget used for baseline tuning.
Success criterion: Beat the confirmed same-budget node baseline by validation metric on Cora or
PubMed without requiring trainer, evaluator, or dataset changes.

Description:

Initialize a node belief from features, compute graph evidence from the current belief with a
standard message-passing operator, and update the belief with a learned residual trust gate.

Implementation notes:

This is the smallest bounded version of the belief-revision idea mined from
`research/original-ideas/belief-propagation_design_mine.md`; it intentionally avoids auxiliary
losses, gate logging, and task-specific trainer hooks for the first pass.

First-pass result:

The bounded implementation was added as `revision_gnn` and passed toy-node, Cora, and PubMed runs
without harness changes. Confirming the best fast config across seeds `0,1,2` with the 200-epoch
budget gave validation means below the confirmed GAT baseline on both Cora (`0.7020` versus
`0.7313`) and PubMed (`0.7253` versus `0.7607`), so this minimal version is discarded. A future
revision should only be reopened with a more specific gate diagnostic or heterophily target.

## 2026-05-19 - Residual Message Passing Baseline

Status: discarded
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

First bounded pass:

Implement `res_gin` first because the previous tuning loop found a confirmed Cora-specific GIN
improvement. The bounded version uses an input projection, pre-norm residual GIN blocks, and an
unchanged task head/trainer interface.

First result:

Fast Cora/PubMed sweeps were not competitive (`0.5220` best Cora validation, `0.6980` best PubMed
validation). Discard this residual GIN block design unless revisiting with a gated or scaled
residual update.

## 2026-05-19 - Dataset-Conditioned Pooling For Graph Tasks

Status: discarded
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

First result:

Implemented an opt-in `mean_max_add` graph pooling head path and added a graph shape test. The
toy-graph smoke run completed. A short seed-0 MolHIV GIN check showed `mean_max_add` validation
`0.7489`, above the matched short mean-pooling run `0.7163` but below the existing GIN MolHIV
baseline mean `0.7619` and seed-0 baseline `0.7698`. Do not promote this pooling path as a default.
