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

## 2026-05-26 - Parallel Round 001 Found Structural Detectors, Not Architecture Wins

Evidence:

Nine remaining graph-theory idea-bank branches were implemented and reviewed in isolated worktrees:
CycleCutGNN-lite, BranchSetGNN, BagAutomatonGNN, ListColorGNN, ClassFlowGNN, DualShadowGNN,
ChordlessCycleMemoryGNN, RegularPatchGNN, and ObstructionTokenGNN. No branch produced an
unconfounded architecture improvement under the review protocol. The strongest synthetic signals
were structural: CycleCutGNN-lite default reached validation AP `0.7568 +/- 0.0466` on
`cycle_matching_v4`, but cycle-only was higher on validation (`0.7686`) and a merged edge-state MPNN
was close (`0.7403`) with higher held-out test AP. DualShadowGNN reached validation AP
`0.8802 +/- 0.0194`, but a pseudo-face histogram MLP reached val/test AP `0.8324/0.8860`, and
shadow-only was comparable to the full model. ObstructionTokenGNN reached perfect AP on its
triangular-prism-vs-`K3,3` diagnostic, but obstruction-count logistic/MLP controls also reached
`1.0/1.0`.

Implication:

The durable result is that structural detectors and handcrafted structural features expose signal
on the current synthetic diagnostics. The tested neural mechanisms were not isolated from simpler
controls, so none should be promoted as a validated novel architecture or benchmark-ready model.
Synthetic-only success must be described narrowly as detector or diagnostic support.

Follow-up:

Before continuing CycleCut, DualShadow, ChordlessCycleMemory, or ObstructionToken, build diagnostics
that match the candidate detector counts and histograms, and require the neural mechanism to beat
same-feature/logistic/MLP and same-capacity ablations by validation metric.

## 2026-05-26 - Shortcut Baselines Must Match The Candidate Structural Features

Evidence:

Round 001 reviewer fixes repeatedly changed the interpretation once stronger shortcuts were added.
ChordlessCycleMemoryGNN initially had validation AP `0.8828 +/- 0.0052`, but explicit-feature GIN
with basic statistics and chordless counts reached validation/test AP `0.9464/0.8270`, and logistic
chordless-count controls reached test AP `0.8356`. RegularPatchGNN's hardened diagnostic removed
the original edge-count shortcut, but degree-stat logistic still reached val/test AP `0.6798/0.7794`
while patch-channel ablations did not support a positive claim. ObstructionTokenGNN's detector
counts solved the diagnostic without token message passing. DualShadowGNN's pseudo-face histogram
baseline matched or exceeded the model on held-out reporting.

Implication:

Generic graph-stat baselines are not enough. For graph-theoretic architectures, the shortcut suite
must include the exact structural quantities the model computes: cycle counts, pseudo-face
histograms, obstruction counts, degree summaries, and merged/same-capacity versions of the model
without the proposed decomposition.

Follow-up:

Add a reusable shortcut-audit pattern for future synthetic graph tasks: prevalence, graph stats,
candidate detector counts, candidate detector histograms, same-feature logistic/MLP, and
same-capacity neural ablations. Do not claim mechanism support until these controls are beaten on
validation.

## 2026-05-26 - Structural Graph Claims Need Invariance And Post-Fix Metric Audits

Evidence:

Round 001 found multiple correctness risks before review fixes. BagAutomatonGNN's first high
seed-0 result depended on node-id-sensitive min-degree tie breaking and failed relabeling audit, so
the result was archived after the implementation was rewritten as an invariant edge-bag scaffold.
ChordlessCycleMemoryGNN's capped cycle enumeration was still label-order dependent after the first
fix pass; a focused fix replaced capped DFS-order truncation with invariant cycle ranking and was
re-reviewed successfully. BranchSetGNN's old audit AP did not line up with the run table, and the
post-fix implementation invalidated the old metric rows. CycleCutGNN-lite completed a relabeling
audit after fixes with max prediction range `8.94e-08`.

Implication:

For structural graph models, relabeling, edge-order, batch-composition, capped-enumeration, and
cache-key audits are part of the evidence, not optional polish. Metrics collected before a
correctness-changing fix cannot support the fixed implementation.

Follow-up:

Standardize graph-model audits for every new synthetic architecture: relabeling invariance,
edge-order invariance, batch-composition checks, cap/tie stress tests, and explicit marking of
pre-fix metrics as archived or invalid evidence.

## 2026-05-26 - Round 001 Node-Idea Screens Reinforce GPR As The Node Target

Evidence:

ListColorGNN was rerun on a harder non-saturated synthetic heterophily diagnostic. The best
ListColor validation accuracy was `0.9083`, while `gpr_gnn` reached validation `0.9167`; fixed
same-channel suppression did not beat the no-suppression multi-channel ablation. ClassFlowGNN was
revised from a hidden-flow proxy into a class-logit potential flow with evidence injection, but the
flow-enabled path reached Cora/PubMed seed-0 validation `0.4260/0.7600` versus flow-disabled
`0.7020/0.7620`.

Implication:

Neither list-color channel allocation nor class-flow residuals are currently competitive with the
existing GPR-style node baseline under the tested protocols. The ClassFlow result is specifically a
negative result for the current trainer-free class-flow formulation, not for all conservation-law
training objectives.

Follow-up:

Keep `gpr_gnn` as the node-classification validation target. Reopen ListColor only with a real
heterophily benchmark or a diagnostic where GPR fails. Reopen ClassFlow only with a dedicated
source/sink task and, likely, an explicit training objective rather than a trainer-free residual.

## 2026-05-26 - TreePack Ablations Favor One BFS Witness Over Learned View Packing

Evidence:

The follow-up TreePackGNN-lite ablations tested the confirmed four-tree gated pack
`architecture_config_hash=8ec4a7ff` against residual-only, tree-only, mean-pooling, and single-tree
controls on `normal-tree-backedge` `cycle_matching_v4`. The residual-only control was chance on
seed 0 (`0.4167/0.4167` val/test AP), so the ordinary graph channel does not explain the result.
The tree-only pack kept some signal (`0.6339/0.5992`), and mean tree pooling was weaker than gated
pooling on seed-0 validation (`0.7089` vs `0.7305`). However, the confirmed single high-degree BFS
tree control `architecture_config_hash=098eb1ec` reached validation AP `0.7493 +/- 0.0113` and test
AP `0.6812 +/- 0.0185` across seeds `[0,1,2]`, beating the full four-tree pack validation/test
means (`0.7462/0.5838`) with lower runtime. Gate diagnostics for the four-tree pack were mostly
near-uniform, with layer-2 entropy `1.3179`, `1.3863`, and `1.2175` versus maximum entropy
`1.3863`.

Implication:

The specific learned tree-pack selection hypothesis is not supported by the ablations. The durable
signal is that a deterministic BFS tree witness can be useful and fairly relabeling-stable on this
synthetic diagnostic, not that keeping several tree views separate until a learned gate is better
than a well-chosen single view. The full pack remains the most relabeling-stable variant audited,
but it is slower and underperforms the single high-degree BFS control on confirmed validation and
held-out test AP.

Follow-up:

Stop extending the current TreePackGNN-lite design. Preserve the single high-degree BFS control as a
synthetic baseline, and move the next bounded architecture cycle to `CycleCutGNN-lite`. Reopen
TreePack only with a new mechanism that canonicalizes or sharply selects tree witnesses rather than
adding more weakly gated views.

## 2026-05-26 - TreePackGNN Stabilizes Tree-Witness Signal On The Backedge Diagnostic

Evidence:

`tree_pack_gnn` was implemented as a bounded four-view spanning-tree encoder with shared tree
message parameters, learned node-level view gating, and a full-graph residual channel. On
`normal-tree-backedge` `cycle_matching_v4`, the confirmed hidden-64 config
`architecture_config_hash=8ec4a7ff` reached validation AP `0.7462 +/- 0.0914` and test AP
`0.5838 +/- 0.0815` across seeds `[0,1,2]`. This beats the previous single-order
NormalTreeBackedgeGNN validation/test means (`0.6635/0.5516`) and the naive four-order DFS average
(`0.6490/0.5626`). Relabeling audits over 16 random relabelings reduced max per-graph prediction
range from up to about `0.71` for single-order NormalTreeBackedgeGNN to at most `0.0027` for
TreePackGNN.

Implication:

Keeping structurally diverse tree witnesses separate until a learned gate is a better robustness
fix than either one arbitrary DFS witness or unweighted multi-order DFS averaging. The result is
still synthetic diagnostic evidence, not a real benchmark claim, because seed 2 had weak held-out
test AP and the model is larger/slower than the single-order baseline.

Follow-up:

This was the pre-ablation result. The bounded ablation cycle is complete; use the later TreePack
ablation insight above for current decisions.

## 2026-05-26 - NormalTreeBackedgeGNN Has Mechanism-Specific But Non-Invariant Synthetic Signal

Evidence:

On `normal-tree-backedge` `cycle_matching_v4`, true DFS edge roles reached validation AP
`0.7292 +/- 0.1321` and test AP `0.5198 +/- 0.0428` across seeds `[0,1,2]`. Role controls were
worse on validation: collapsed roles `0.6549 +/- 0.0757`, tree-only `0.6517 +/- 0.1046`,
back-only `0.6155 +/- 0.0600`, shuffled roles `0.6395 +/- 0.0608`, and a pseudo-random DFS order
`0.5702 +/- 0.0081`. A permutation audit over 16 random relabelings of the same test graphs found
relabeled AP means `0.5557`, `0.4504`, and `0.5791` for seeds `0`, `1`, and `2`, with max
per-graph prediction ranges up to about `0.71`.

Implication:

The diagnostic signal is partly mechanism-specific because true roles beat collapsed and shuffled
role labels on validation. The current architecture is still not graph-invariant: arbitrary node
relabeling can substantially move predictions, and the current depth/span normalization is also
batch-composition sensitive.

Follow-up:

Do not make broader benchmark claims for the current single-order NormalTreeBackedgeGNN. Revisit it
only after per-graph normalization and a canonical or learned distribution over DFS/normal trees;
otherwise move to `TreePackGNN` or `CycleCutGNN-lite`.

## 2026-05-26 - Naive Multi-Order DFS Averaging Did Not Improve NormalTreeBackedgeGNN

Evidence:

`NormalTreeBackedgeGNN-lite` was extended with `model.num_tree_orders`, which averages shared
up-tree, down-tree, and back-edge message channels over several deterministic DFS root/neighbor
orders. On `normal-tree-backedge` `cycle_matching_v4`, the four-order config
`architecture_config_hash=2fb8a3d0` reached validation AP `0.6490 +/- 0.0864` and test AP
`0.5626 +/- 0.2606` across seeds `[0,1,2]`, versus the single-order config
`architecture_config_hash=f9449a35` at validation AP `0.6635 +/- 0.0788` and test AP
`0.5516 +/- 0.0865`. Mean runtime increased from about `14.34s` to `75.73s`.

Implication:

Simple unweighted averaging over DFS decompositions is not a good robustness fix. It likely blurs
the traversal-specific back-edge span signal while adding substantial cost.

Follow-up:

Keep `num_tree_orders` available for experiments but keep the default at one order. Revisit this
line only with a learned order gate, root-specific pooling, or a sharper cycle feature; otherwise
move to `TreePackGNN` or another idea-bank candidate.

## 2026-05-26 - Hardened Back-Edge Diagnostic Supports The Mechanism But Shows Weak Generalization

Evidence:

The `normal-tree-backedge` diagnostic was hardened as `cycle_matching_v4`: every graph has 20 nodes,
30 undirected edges, degree 3 at every node, constant node features, and random node relabeling.
Labels depend on low- versus high-crossing nonlocal perfect matchings over a hidden cycle order.
Shortcut graph-stat baselines using node count, edge count, density, degree summaries, triangle
count, DFS back-edge count, and DFS back-edge span summaries did not solve it: logistic stats
reached val/test AP `0.5777/0.5191`, and an MLP over the same stats reached `0.6419/0.5192`.
Across seeds `[0,1,2]`, GIN and GCN both stayed at val/test AP `0.4167`, while
`normal_tree_backedge_gnn` with `architecture_config_hash=f9449a35` reached validation AP
`0.6635 +/- 0.0788` and test AP `0.5516 +/- 0.0865`.

Implication:

The DFS tree/back-edge channel has real diagnostic signal after obvious count, degree, and
standard-message-passing shortcuts are blocked. However, the validation/test gap means this is not
yet robust architecture evidence for real benchmarks.

Follow-up:

Test a multi-root or multi-order NormalTreeBackedgeGNN on `cycle_matching_v4` to reduce dependence
on arbitrary DFS order from random relabeling. Do not move to real graph benchmarks until held-out
diagnostic AP improves.

## 2026-05-26 - First Normal-Tree Back-Edge Diagnostic Was Too Easy For GIN

Evidence:

`NormalTreeBackedgeGNN-lite` separated deterministic DFS-tree upward, downward, and back-edge
message channels and added a `normal-tree-backedge` synthetic graph diagnostic. On the final
`endpoint_pairing_v2` diagnostic, both `gin` (`architecture_config_hash=bb49e9d0`) and
`normal_tree_backedge_gnn` (`architecture_config_hash=83e3c616`) reached seed-0 validation and test
average precision `1.0000`.

Implication:

This diagnostic is not useful evidence for the normal-tree/back-edge mechanism because the ordinary
GIN baseline solves it immediately. The model implementation works, but no architecture improvement
or failure claim should be made from this task.

Follow-up:

Before benchmarking `NormalTreeBackedgeGNN-lite` on real graph tasks, build a harder diagnostic with
matched local neighborhoods and no simple chord-endpoint feature leakage, or move to a different
graph-theory idea.

## 2026-05-26 - Separator-Aware Routing Was A PubMed Near Miss, Not A Confirmed Improvement

Evidence:

`SepBottleneckGNN-lite` added cached articulation/bridge markers, a small articulation-node token,
and a bounded residual channel for separator-adjacent edges on top of a GPR-style encoder. Cora
seed-0 fast screening reached validation `0.7580`, below the confirmed GPR Cora mean `0.7827`.
PubMed seed-0 reached `0.8080`, but confirmation across seeds `[0,1,2]` for
`architecture_config_hash=0f352f9d` reached validation `0.8000 +/- 0.0053`, slightly below the
confirmed GPR PubMed mean `0.8007`.

Implication:

Cheap articulation/bridge-aware residual routing is not a confirmed improvement over GPR on
Cora/PubMed. Citation networks may not expose the separator bottleneck mechanism strongly enough,
or the current separator score may be too broad to help.

Follow-up:

Test this mechanism on a small synthetic separator-bottleneck node task before spending more
Cora/PubMed budget on separator variants. If it fails there too, move to the next graph-theory idea.

## 2026-05-26 - Config-Level Evidence Is Required For Architecture Claims

Evidence:

The previous aggregate summaries grouped by task, dataset, model, and metric only. That mixed many
hyperparameter settings into one mean/std row, making the table unsuitable for selecting exact
architecture/config candidates across seeds.

Implication:

Future architecture claims must use config-level summaries grouped by `architecture_config_hash`.
Model-level summaries are useful diagnostics for broad trends, but they are mixed-config evidence
and must not be used to choose a winning configuration.

Follow-up:

Re-export existing aggregate tables with the hardened exporter before using them for future
architecture selection. Treat `*_by_config_mean_std.csv` as evidence and `*_by_model_mean_std.csv`
as diagnostic context.

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
