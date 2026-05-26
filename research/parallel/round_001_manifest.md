# Parallel Research Round 001 Manifest

Date: 2026-05-26

Base branch: `main`

Base commit: `d6141f80aad11906eceb0eb5a246acf3661aef8b`

Coordinator note: parent checkout had uncommitted prior research changes at orchestration time. Implementation worktrees are isolated Git worktrees branched from the base commit above; this manifest is the only parent-coordinator artifact for this round.

## Ideas Found

- SepBottleneckGNN
- CycleCutGNN
- BranchSetGNN
- BagAutomatonGNN
- TreePackGNN
- NormalTreeBackedgeGNN
- ListColorGNN
- ClassFlowGNN
- DualShadowGNN
- ChordlessCycleMemoryGNN
- RegularPatchGNN
- ObstructionTokenGNN

## Ideas Excluded And Why

- SepBottleneckGNN: already tested; bounded experiment note and status exist.
- NormalTreeBackedgeGNN: already tested and active; experiment note and multiple audit/result artifacts exist.
- TreePackGNN: already tested and active; experiment note, ablations, and audit artifacts exist.

## Ideas Selected For Implementation

- CycleCutGNN-lite
- BranchSetGNN
- BagAutomatonGNN
- ListColorGNN
- ClassFlowGNN
- DualShadowGNN
- ChordlessCycleMemoryGNN
- RegularPatchGNN
- ObstructionTokenGNN

## Branch/Worktree Name Per Idea

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

## Assigned Subagent Per Idea

| Idea | Assigned subagent |
| --- | --- |
| CycleCutGNN-lite | `gnn_implementer` |
| BranchSetGNN | `gnn_implementer` |
| BagAutomatonGNN | `gnn_implementer` |
| ListColorGNN | `gnn_implementer` |
| ClassFlowGNN | `gnn_implementer` |
| DualShadowGNN | `gnn_implementer` |
| ChordlessCycleMemoryGNN | `gnn_implementer` |
| RegularPatchGNN | `gnn_implementer` |
| ObstructionTokenGNN | `gnn_implementer` |

## Expected Output Files

Each implementation branch should write only idea-specific artifacts:

- `research/experiments/<idea_slug>.md`
- `src/gnn_gym/models/<idea_slug>.py`, when a model is implemented
- `configs/models/<idea_slug>.yaml`, when a model is implemented
- `tests/test_<idea_slug>.py`, when a model or diagnostic is implemented
- `results/tables/<idea_slug>_*.csv`
- `results/tables/<idea_slug>_*.json`

Implementation agents must not update global synthesis files in this round:

- `research/INSIGHTS.md`
- `research/AGENT_SCRATCHPAD.md`
- `results/tables/research_all_runs.csv`
