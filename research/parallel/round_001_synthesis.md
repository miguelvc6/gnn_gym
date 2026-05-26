# Parallel Research Round 001 Synthesis

Date: 2026-05-26

## Round Objective

Run a parallel implementation and review round over the remaining graph-theory idea bank entries,
using isolated branches/worktrees, then preserve durable research memory without making unsupported
architecture claims.

## Ideas Attempted

- CycleCutGNN-lite
- BranchSetGNN
- BagAutomatonGNN
- ListColorGNN
- ClassFlowGNN
- DualShadowGNN
- ChordlessCycleMemoryGNN
- RegularPatchGNN
- ObstructionTokenGNN

## Ideas Not Attempted

- SepBottleneckGNN: already tested before this round.
- NormalTreeBackedgeGNN: already tested before this round.
- TreePackGNN: already tested and ablated before this round.

## Branch/Worktree Per Idea

| Idea | Branch | Worktree |
| --- | --- | --- |
| CycleCutGNN-lite | `research/parallel/cycle-cut-gnn-lite` | `/tmp/gnn_gym_parallel/cycle-cut-gnn-lite` |
| BranchSetGNN | `research/parallel/branch-set-gnn` | `/tmp/gnn_gym_parallel/branch-set-gnn` |
| BagAutomatonGNN | `research/parallel/bag-automaton-gnn` | `/tmp/gnn_gym_parallel/bag-automaton-gnn` |
| ListColorGNN | `research/parallel/list-color-gnn` | `/tmp/gnn_gym_parallel/list-color-gnn` |
| ClassFlowGNN | `research/parallel/class-flow-gnn` | `/tmp/gnn_gym_parallel/class-flow-gnn` |
| DualShadowGNN | `research/parallel/dual-shadow-gnn` | `/tmp/gnn_gym_parallel/dual-shadow-gnn` |
| ChordlessCycleMemoryGNN | `research/parallel/chordless-cycle-memory-gnn` | `/tmp/gnn_gym_parallel/chordless-cycle-memory-gnn` |
| RegularPatchGNN | `research/parallel/regular-patch-gnn` | `/tmp/gnn_gym_parallel/regular-patch-gnn` |
| ObstructionTokenGNN | `research/parallel/obstruction-token-gnn` | `/tmp/gnn_gym_parallel/obstruction-token-gnn` |

## Reviewer Outcome Per Idea

| Idea | Reviewer outcome |
| --- | --- |
| CycleCutGNN-lite | Accept for synthesis only as a limited structural-feature diagnostic; mechanism not isolated. |
| DualShadowGNN | Accept for synthesis as synthetic pseudo-face support; shortcut-confounded. |
| BagAutomatonGNN | Implementation blockers mostly fixed; unconfirmed post-fix infrastructure only. |
| ChordlessCycleMemoryGNN | Cap-invariance blocker fixed; conservative negative/inconclusive. |
| ObstructionTokenGNN | Accept as detector/infrastructure evidence; count baselines solve the task. |
| RegularPatchGNN | Accept as archive/negative. |
| BranchSetGNN | Archive/do not merge. |
| ListColorGNN | Archive/do not merge. |
| ClassFlowGNN | Archive/negative. |

## Evidence Quality Per Idea

| Idea | Evidence quality |
| --- | --- |
| CycleCutGNN-lite | Confirmed synthetic signal across seeds `[0,1,2]`, but split cut/cycle mechanism does not beat cycle-only and is close to merged edge-state controls. |
| DualShadowGNN | Confirmed synthetic pseudo-face signal across seeds `[0,1,2]`, but pseudo-face histogram controls are strong and shadow-only is comparable to full. |
| BagAutomatonGNN | Old positive run invalidated by node-id-sensitive decomposition; post-fix invariant edge-bag rewrite has smoke/invariance coverage but no post-fix validation result. |
| ChordlessCycleMemoryGNN | Useful cycle-memory tests and diagnostics, but explicit feature/count baselines beat the model; capped enumeration is now invariant. |
| ObstructionTokenGNN | Detector works on a controlled motif diagnostic, but obstruction-count logistic/MLP controls also solve it. |
| RegularPatchGNN | Hardened diagnostic and controls added; residual degree-stat leakage remains and patch-channel ablations are negative. |
| BranchSetGNN | Weak legacy AP, shortcut baselines stronger, and post-fix implementation changed enough that old metrics are not evidence. |
| ListColorGNN | Harder synthetic heterophily diagnostic is non-saturated, but GPR beats ListColor on validation and suppression is not isolated. |
| ClassFlowGNN | Revised to class-logit flow, but flow-enabled path hurts Cora and does not improve PubMed relative to flow-disabled. |

## Result Summary Table

Validation metrics drive all decisions below. Test metrics are held-out reporting only.

| Idea | Primary dataset/diagnostic | Main baseline | Validation result | Test result | Evidence classification | Reviewer decision | Synthesis decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CycleCutGNN-lite | `cycle_matching_v4` | merged edge-state MPNN / cycle-only ablation | default `0.7568 +/- 0.0466`; cycle-only `0.7686`; merged MPNN `0.7403` | default `0.7306 +/- 0.0545`; merged MPNN `0.7914` | Structural-feature signal, mechanism not isolated | Accept narrow synthesis | Revise mechanism test; do not merge as architecture win |
| DualShadowGNN | `cycle_matching_v4` | pseudo-face histogram MLP | full `0.8802 +/- 0.0194`; histogram MLP `0.8324` | full `0.8280`; histogram MLP `0.8860` | Synthetic pseudo-face signal, shortcut-confounded | Accept narrow synthesis | Revise diagnostic; no planar-dual claim |
| BagAutomatonGNN | `normal-tree-backedge` smoke/post-fix checks | GIN/GCN; old result archived | no post-fix validation result | no post-fix test result | Unconfirmed infrastructure | Minor fixes/no positive claim | Rerun needed before evidence use |
| ChordlessCycleMemoryGNN | chordless-cycle synthetic | explicit-feature GIN / chordless-count logistic | cycle model `0.8828 +/- 0.0052`; explicit-feature GIN `0.9464`; logreg counts `0.8136` | cycle model `0.6679 +/- 0.1075`; explicit-feature GIN `0.8270`; logreg counts `0.8356` | Negative/inconclusive; infra/tests useful | Focused fix passed | Archive as non-win; keep infrastructure |
| ObstructionTokenGNN | triangular prism vs `K3,3` | obstruction-count logreg/MLP | model `1.0`; count logreg/MLP `1.0` | model `1.0`; count logreg/MLP `1.0` | Detector evidence only | Accept infra synthesis | Keep as detector/shortcut-audit infrastructure |
| RegularPatchGNN | planted-patch hardened diagnostic | degree-stat logistic / GIN on old screen | degree-stat logreg `0.6798`; full 2-epoch smoke `0.5074`; old GIN `1.0` | degree-stat logreg `0.7794`; full smoke `0.6546`; old GIN `1.0` | Negative; diagnostic still leaks | Accept negative synthesis | Archive |
| BranchSetGNN | `cycle_matching_v4` / controls | graph-stat shortcut MLP | legacy BranchSet `0.5626 +/- 0.0797`; shortcut MLP `0.6419` | legacy BranchSet `0.4607 +/- 0.0383`; shortcut MLP `0.5192` | Negative; old metrics pre-fix | Archive/do not merge | Archive |
| ListColorGNN | harder bipartite heterophily | GPR-GNN | best ListColor `0.9083`; GPR `0.9167` | best ListColor `0.9250`; selection by validation still fails | Negative | Archive/do not merge | Archive |
| ClassFlowGNN | Cora/PubMed | flow-disabled ablation / GPR-APPNP | Cora flow `0.4260` vs disabled `0.7020`; PubMed flow `0.7600` vs disabled `0.7620` | Cora flow `0.4220`; PubMed flow `0.7470` | Negative | Archive/negative | Archive |

## Merge/Revise/Archive Decision

| Idea | Decision |
| --- | --- |
| CycleCutGNN-lite | Revise. Keep result as structural-feature diagnostic only. |
| DualShadowGNN | Revise. Keep result as pseudo-face synthetic support only. |
| BagAutomatonGNN | Rerun needed. Keep post-fix implementation as infrastructure only. |
| ChordlessCycleMemoryGNN | Archive as architecture win; keep infrastructure/tests. |
| ObstructionTokenGNN | Keep as infrastructure only. |
| RegularPatchGNN | Archive. |
| BranchSetGNN | Archive. |
| ListColorGNN | Archive. |
| ClassFlowGNN | Archive. |

No branch should be merged as a validated architecture improvement from this round.

## Durable Insights Promoted To INSIGHTS.md

- Round 001 produced structural detector signal, not an unconfounded architecture win.
- Shortcut and detector baselines must be matched to the candidate mechanism.
- Relabeling, edge-order, capped-enumeration, and post-fix metric validity checks are required for graph-structural claims.
- GPR remains the node-classification validation target after ListColor and ClassFlow negative screens.

## Negative Results Preserved

- BranchSetGNN: weak synthetic result, shortcuts stronger, old metrics invalid after post-fix changes.
- ListColorGNN: harder diagnostic did not beat GPR on validation; suppression mechanism not isolated.
- ClassFlowGNN: class-flow path hurt Cora and did not improve PubMed.
- RegularPatchGNN: diagnostic remains degree-stat-confounded; patch channel did not help in ablations.
- ChordlessCycleMemoryGNN: explicit feature/count baselines outperform the cycle-memory model.
- ObstructionTokenGNN: detector-count controls solve the diagnostic.

## Open Methodological Risks

- Synthetic tasks can reward detector counts or graph statistics rather than learned message passing.
- Seed-0 and pre-fix metrics are screening/debug evidence only.
- Old metrics cannot support a branch after correctness-changing implementation fixes.
- Several graph models reject or ignore `edge_attr`, so no molecule or edge-feature benchmark claim is justified.
- Small synthetic validation/test splits make AP volatile and prevalence-sensitive.
- Config-level summaries by `architecture_config_hash` remain mandatory for any future claim.

## Recommended Next Parallel Round

1. Build a standard shortcut suite before model work: graph stats, detector counts, histogram MLP/logreg, same-capacity ablations, and exact candidate-feature baselines.
2. Build a standard graph-model audit suite: relabeling, edge-order, batch-composition, capped-enumeration, and cache-bound tests.
3. Continue CycleCut only if the split channel beats cycle-only and merged edge-state MPNN on validation after exact structural-feature controls.
4. Continue DualShadow only on a harder pseudo-face arrangement diagnostic matched on pseudo-face count, cycle-length histogram, and collapsed pseudo-face stats.
5. Treat ObstructionToken as detector/shortcut infrastructure unless a harder diagnostic controls obstruction-count marginals.
6. Archive BranchSet, ListColor, ClassFlow, and RegularPatch until their stated blockers are fixed with new diagnostics.
