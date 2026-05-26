# GNN-GYM source-library note

This file treats the source as an idea generator, not as a recipe book. Candidate architectures below are research hypotheses for GNN-GYM. They are not claims of literature novelty until a separate prior-art check is done.

Each architecture should be evaluated with the GNN-GYM long-running protocol: toy crash check, seed-0 screen, confirmation on seeds `[0, 1, 2]`, validation-metric selection only, test metric for reporting only, and aggregation by `architecture_config_hash`.

# Source family: pure graph theory and structural decomposition

## Why this family may produce original GNN architectures

Graph theory provides structural objects that ordinary message passing does not explicitly respect: separators, cycle spaces, cut spaces, minors, tree-decompositions, planarity, coloring, flows, and regularity. These objects can become tokens, constraints, pooling units, or synthetic tasks.

## Core uploaded sources

- `graph_theory_book_diestel.pdf`

## Concepts to mine

- Connectivity, blocks, bridges, and articulation points.
- Cycle space and cut space.
- Flows and cuts.
- Planarity and duality.
- Minors and branch sets.
- Tree-decompositions and tree-width.
- Regularity and extremal structure.
- Coloring and list coloring.

## Known GNN/ML prior art to avoid merely reimplementing

- Generic subgraph GNNs.
- Positional encodings from graph distance alone.
- Pooling by communities without a tested structural claim.
- Tree-decomposition GNNs without a new bag-state or interface mechanism.

## Candidate mechanisms

- Separator tokens and interface messages.
- Branch-set contraction modules.
- Cycle/cut edge-state decomposition.
- Bag automata over approximate tree decompositions.
- Planar or pseudo-dual message channels.

## Synthetic tasks this source suggests

1. **Separator bottleneck task:** label depends on information crossing a small separator.
2. **Cycle/cut discrimination:** distinguish circulation from gradient/cut evidence.
3. **Minor stability:** prediction should be invariant under contracting irrelevant branches.
4. **Treewidth task:** success depends on bag-local dynamic programming, not local smoothing.
5. **Planar dual task:** graph-level label depends on face/cycle structure.

## Architecture idea stubs

- `SeparatorDomainGNN`: decompose graph into blocks around separators and pass messages through learned interface states.
- `BranchSetMinorGNN`: learn connected branch-set contractions, run coarse message passing, and lift back to nodes.
- `CycleCutProjectionGNN`: maintain edge states projected into approximate cycle-like and cut-like components.
- `BagAutomatonGNN`: approximate tree-decomposition bag states with learnable introduce/forget/join operations.
- `DualShadowGNN`: build pseudo-face/cycle tokens and pass messages in both primal and dual shadows.

## Rejected or low-novelty translations

- Adding cycle counts as features without a cycle mechanism.
- Running standard pooling and calling it minor contraction.
- Using tree decompositions only as batching order, not as a bag-state architecture.
