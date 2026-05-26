# Original GNN Architecture Ideas: Graph-Theory-Inspired Long-Running Program

Date: 2026-05-26

Purpose: candidate architecture ideas for the GNN Gym agent loop. These are not claims of novelty. Each idea is a falsifiable research hypothesis. The agent must perform a prior-art audit before labeling any result novel.

Current strong baseline to beat for Cora/PubMed node classification: `gpr_gnn`, unless newer confirmed results in `research/INSIGHTS.md` say otherwise.

Primary principle: do not optimize only for a leaderboard. Each experiment should test a mechanism that can teach us something about graph representation learning.

---

## How to use this file

For each idea:

1. Add a short prior-art audit before implementation.
2. Implement the smallest bounded version.
3. Run toy crash checks.
4. Run seed-0 fast screen.
5. Confirm only promising configs with seeds `[0, 1, 2]`.
6. Aggregate by `architecture_config_hash`.
7. Promote only durable conclusions to `research/INSIGHTS.md`.
8. Preserve negative results if they clarify the hypothesis.

Do not claim novelty from:

- simple hyperparameter tuning;
- reimplementing a known baseline;
- concatenating two known models without a new mechanism;
- seed-0 results only;
- test-set selection;
- mixed-config averages.

---

# Idea 1: SepBottleneckGNN — Separator-Aware Bottleneck Routing

## Scientific hypothesis

Many GNN failures are caused by treating all edges as equally safe conduits. Edges and vertices near small separators, articulation points, bridges, or low-connectivity cuts are information bottlenecks. A model that explicitly distinguishes intra-block propagation from cross-separator propagation should reduce oversquashing and harmful overmixing.

## Mechanism

Precompute cheap structural tags:

- bridge edge;
- articulation endpoint;
- biconnected component id;
- whether an edge is intra-block or cross-block;
- optional approximate local edge connectivity for sampled node pairs.

Use two message channels:

```text
m_intra  = message along edges inside the same robust block
m_cross  = message across separator/bottleneck edges
h_v'     = update(h_v, aggregate(m_intra), gate_v * aggregate(m_cross))
```

The cross-separator gate should be initialized conservatively, e.g. near 0 or near the GCN/GPR-equivalent default depending on baseline compatibility.

## Why this is not just a known baseline

GAT learns generic edge attention. SepBottleneckGNN hard-codes a graph-theoretic distinction between robust local regions and separator-mediated communication, then learns how much separator information should cross.

## Closest known related architectures

- GAT/GATv2 edge attention.
- Message passing with structural encodings.
- Oversquashing-aware GNNs.
- Curvature-aware rewiring methods.

## Expected insight if it succeeds

Small separators are not just diagnostics of oversquashing; they can be used as architectural routing primitives.

## Expected insight if it fails

For the tested node tasks, local separator structure may not be the limiting factor, or GPR-style global diffusion may already handle these bottlenecks better than explicit separator routing.

## Target task family

Start with node classification:

- Cora
- PubMed
- heterophily datasets if available

Then try link prediction where separator edges may affect connectivity.

## Primary baseline to beat

- `gpr_gnn` for Cora/PubMed.
- APPNP/GATv2 as secondary baselines.

## Minimal falsifying experiment

A seed-0 fast screen on Cora and PubMed. If SepBottleneckGNN cannot approach GPR within a small margin and shows unstable training, stop the first version.

## Confirmation protocol

Confirm only if seed-0 beats or nearly matches `gpr_gnn` validation. Use seeds `[0, 1, 2]` and compare by validation metric under the same budget.

## Complexity/runtime risk

Low to medium. Biconnected components and bridges are cheap. Approximate local connectivity can be deferred.

## Implementation boundary

Do not rewrite trainers. Implement precomputed graph structural features in dataset transform or inside the model with cached preprocessing.

## Status

2026-05-26: First bounded `SepBottleneckGNN-lite` implementation completed. Cora seed-0 did not
justify confirmation. PubMed seed-0 was promising, but the confirmed config
`architecture_config_hash=0f352f9d` reached validation `0.8000 +/- 0.0053`, slightly below confirmed
`gpr_gnn` PubMed validation `0.8007`. Next useful step is a synthetic separator-bottleneck
diagnostic before additional citation-network variants.

---

# Idea 2: CycleCutGNN — Cycle/Cut Dual Message Passing

## Scientific hypothesis

Standard node-message-passing collapses two different kinds of graph signal: gradient-like information flowing across cuts and circulation-like information living around cycles. A model that explicitly separates cut-space and cycle-space components can learn whether a task depends more on separations, communities, or cyclic/ring structure.

## Mechanism

Maintain both node states and edge states.

Use an oriented incidence matrix `B` conceptually:

```text
edge_gradient = B^T h
node_div      = B e
cycle_residue = e - projection_to_cut_space(e)
```

A lightweight first implementation should avoid expensive exact projections. Use approximations:

- divergence aggregation from edge states to nodes;
- triangle/short-cycle residual features;
- optional sparse least-squares projection later.

Layer sketch:

```text
e_ij' = EdgeMLP(h_i, h_j, e_ij, structural_cycle_features)
h_i'  = NodeMLP(h_i, sum_divergent_edges, sum_cycle_residue_edges)
```

## Why this is not just a known baseline

GINE and gated edge GNNs use edge features, but they do not explicitly factor messages into cut-like and cycle-like components. This idea tests whether the classical cycle/cut decomposition is a useful neural inductive bias.

## Closest known related architectures

- Hodge Laplacian neural networks.
- Simplicial/cellular neural networks.
- GINE and gated edge networks.
- Motif/cycle-aware graph networks.

## Expected insight if it succeeds

Graph tasks may benefit from separating boundary/flow information from cyclic structure, especially molecules and datasets where rings or feedback loops matter.

## Expected insight if it fails

The exact algebraic cycle/cut distinction may be too expensive, too noisy, or already captured by simpler edge-aware GNNs on these benchmarks.

## Target task family

Best after edge-attribute plumbing is fixed:

- MolHIV
- MolPCBA
- LRGB peptides
- synthetic cycle/cut tasks

Can also test Cora/PubMed with synthetic edge states, but graph prediction is the natural fit.

## Primary baseline to beat

- GIN/GINE once edge attributes are available.
- Existing graph baselines in the gym.

## Minimal falsifying experiment

Synthetic graph classification:

- distinguish graphs with similar degree distributions but different cycle structure;
- distinguish tree-like vs ring-rich molecular-style graphs.

Then run MolHIV seed-0.

## Confirmation protocol

Seeds `[0, 1, 2]` only if seed-0 improves validation ROC-AUC over the current graph baseline under the same budget.

## Complexity/runtime risk

Medium. Exact cycle-space projection can be expensive. Start with local short-cycle approximations and edge divergence.

## Implementation boundary

Do not implement full Hodge theory first. Build a minimal edge-state model that exposes divergent vs cyclic channels.

---

# Idea 3: BranchSetGNN — Minor-Contraction Multiscale Message Passing

## Scientific hypothesis

Many graph-level and node-level patterns are stable under contracting locally coherent regions. A GNN that learns over a chain of graph minors may capture structure at multiple resolutions more naturally than arbitrary pooling.

## Mechanism

Construct a minor chain:

```text
G0 -> G1 -> G2 -> ... -> GK
```

Each step contracts connected branch sets. First implementation can use hard, non-learned contractions:

- heavy-edge matching by feature similarity;
- connected clustering;
- biconnected-block-aware contraction;
- molecule ring/functional-group contraction if edge attributes exist.

Run message passing at each level and unpool back:

```text
h_Gk = MP_k(contracted graph)
h_G0 = combine(original states, lifted coarse states)
```

## Why this is not just a known baseline

DiffPool and Graph U-Net learn or choose pooling, but BranchSetGNN constrains coarsening to graph-minor operations with explicit connected branch sets and records the contraction chain as part of the architecture.

## Closest known related architectures

- DiffPool.
- MinCutPool.
- Graph U-Net.
- hierarchical graph pooling.
- graph minor theory.

## Expected insight if it succeeds

Minor-respecting coarsening may preserve task-relevant structure better than unconstrained learned pooling.

## Expected insight if it fails

The chosen contraction heuristic may erase label-relevant information, or benchmark tasks may not reward minor-stable abstractions.

## Target task family

- graph prediction first;
- node classification second, using coarse features lifted back to nodes.

## Primary baseline to beat

- graph tasks: GIN/GINE/graph baselines;
- node tasks: `gpr_gnn` only if the lifted version is competitive.

## Minimal falsifying experiment

Synthetic graph classification where labels are invariant under contracting pendant trees or local cliques. If the model cannot exploit a task designed for minor stability, stop.

## Confirmation protocol

Only run expensive benchmarks if the synthetic task shows the expected behavior.

## Complexity/runtime risk

Medium to high. Coarsening must be deterministic or carefully seeded. Hard contractions are easier than differentiable contractions.

## Implementation boundary

Start with non-learned contractions and cache the minor chain per graph. Do not implement differentiable branch-set learning in the first version.

---

# Idea 4: BagAutomatonGNN — Tree-Decomposition Bag State Network

## Scientific hypothesis

Some graph properties are naturally computed by dynamic programming over tree decompositions. A GNN that processes approximate tree-decomposition bags and their overlaps can learn longer-range structure without relying only on repeated local diffusion.

## Mechanism

Compute an approximate tree decomposition using a heuristic such as min-fill or min-degree elimination.

Create a bag graph:

```text
bag nodes: subsets of original vertices
bag edges: overlap between adjacent bags
bag feature: pooled vertex/edge features inside bag + overlap features
```

Run message passing or a small transformer over the bag tree, then project bag states back to vertices through bag membership.

```text
h_bag' = BagMP(h_bag, overlap_features)
h_v'   = combine(h_v, aggregate_bags_containing_v)
```

## Why this is not just a known baseline

Graph transformers add global attention and positional encodings. BagAutomatonGNN instead uses approximate decomposition bags as computational units, closer to graph-theoretic dynamic programming.

## Closest known related architectures

- Tree Decomposed GNN.
- Junction-tree neural models.
- GraphGPS/Graphormer-style global graph transformers.
- subgraph GNNs.

## Expected insight if it succeeds

Tree-decomposition structure may be a useful neural scaffold for tasks that require reasoning over separators, motifs, or bounded-treewidth substructure.

## Expected insight if it fails

Approximate tree decompositions may be too unstable/noisy, or current benchmarks may not require decomposition-style computation.

## Target task family

Start with synthetic structural tasks:

- treewidth-proxy classification;
- detecting small minors;
- detecting whether two nodes are separated by a small bag.

Then graph prediction.

## Primary baseline to beat

- GIN/GINE on graph tasks;
- subgraph GNN baselines if implemented;
- GraphGPS/Graphormer only as conceptual prior art, not necessarily current gym baseline.

## Minimal falsifying experiment

Synthetic bounded-treewidth dynamic-programming task. If a bag-based model does not beat plain GIN there, it is unlikely to justify benchmark use.

## Confirmation protocol

Synthetic confirmation first; real benchmark confirmation second.

## Complexity/runtime risk

High. Tree decomposition heuristics and bag projection add nontrivial preprocessing.

## Implementation boundary

Use approximate heuristics. Do not try exact treewidth. Limit bag size and number of bags.

---

# Idea 5: TreePackGNN — Low-Overlap Spanning-Tree Ensemble

## Scientific hypothesis

A connected graph can be viewed through multiple spanning trees. Propagation along trees avoids cyclic overmixing and provides long paths with stable routing. An ensemble of low-overlap spanning trees may capture robust connectivity while reducing oversquashing compared with single-path or full-neighborhood propagation.

## Mechanism

Sample or construct `T` spanning trees per graph:

- BFS tree;
- DFS tree;
- random spanning tree;
- maximum-feature-similarity spanning tree;
- low-overlap tree chosen to avoid edges used in previous trees.

Run a shared tree message-passing module on each tree, aggregate tree views, then optionally combine with a base GPR/GCN branch.

```text
h_v_tree[t] = TreeMP_t(G_tree_t)
h_v' = combine(h_v_base, mean_t h_v_tree[t], variance_t h_v_tree[t])
```

## Why this is not just a known baseline

Random walk and diffusion methods average over many walks. TreePackGNN tests whether a small set of globally connected, low-cycle backbones is a better computational substrate than unrestricted local neighborhoods.

## Closest known related architectures

- tree-based GNNs;
- random-walk positional encodings;
- spanning-tree methods;
- DropEdge/graph augmentation;
- ensemble GNNs.

## Expected insight if it succeeds

Connectivity backbones may provide useful global routing without full graph diffusion.

## Expected insight if it fails

Removing non-tree edges loses too much local information, or multiple trees are just a noisy approximation to ordinary message passing.

## Target task family

- Cora/PubMed node classification;
- link prediction;
- graph tasks where global connectivity matters.

## Primary baseline to beat

- `gpr_gnn` for Cora/PubMed.
- GCN/GAT as lower baselines.

## Minimal falsifying experiment

Compare:

```text
GCN on full graph
GCN on one spanning tree
TreePackGNN with 4 trees
```

If multiple trees do not beat one tree or full graph on validation, stop.

## Confirmation protocol

Seeds `[0, 1, 2]` only after seed-0 beats or nearly matches `gpr_gnn`.

## Complexity/runtime risk

Low to medium. Spanning trees are cheap. Multiple tree passes increase runtime linearly in number of trees.

## Implementation boundary

Start with 2 to 4 trees. Cache trees. Do not make tree selection differentiable initially.

---

# Idea 6: NormalTreeBackedgeGNN — DFS Tree Plus Back-Edge Cycle Closure

## Scientific hypothesis

Every connected graph admits a depth-first-search-like normal spanning tree. Such a tree separates hierarchical backbone edges from back edges that close cycles. Standard GNNs treat both edge types similarly. Distinguishing tree propagation from back-edge cycle closure may improve structural reasoning.

## Mechanism

For each connected component:

1. Build a DFS tree.
2. Mark tree edges and non-tree/back edges.
3. Compute node depth, ancestor relation approximations, and back-edge span length.

Use three channels:

```text
downward tree messages
upward tree messages
back-edge cycle-closure messages
```

Layer sketch:

```text
h_tree = TreeUpDown(h, tree_edges, depth)
h_back = BackEdgeMP(h, back_edges, span_length)
h'     = combine(h, h_tree, h_back)
```

## Why this is not just a known baseline

It is not generic edge attention. It imposes a DFS/normal-tree structural decomposition and tests whether cycle-closing edges should be treated differently from backbone edges.

## Closest known related architectures

- TreeLSTM/tree GNNs.
- positional encodings based on depth or random walks.
- cycle-aware GNNs.

## Expected insight if it succeeds

A simple graph-theoretic traversal structure can serve as an architectural scaffold for separating hierarchy from cyclic closure.

## Expected insight if it fails

The DFS tree may be too arbitrary, and performance may depend on root/order choices rather than invariant structure.

## Target task family

- graph prediction;
- node classification;
- synthetic cycle/back-edge tasks.

## Primary baseline to beat

- GIN/GINE for graph tasks.
- `gpr_gnn` for node tasks only if competitive.

## Minimal falsifying experiment

Synthetic classification of graphs with same tree backbone but different back-edge patterns. If the model cannot exploit the channel separation, stop.

## Confirmation protocol

Use multiple DFS roots/orders or deterministic canonical root to test stability.

## Complexity/runtime risk

Low. DFS trees and edge tags are cheap.

## Implementation boundary

Keep the first model invariant enough: aggregate over several rooted DFS trees or use deterministic root rules.

## Status

2026-05-26: First bounded `NormalTreeBackedgeGNN-lite` implementation completed with deterministic
DFS tree/back-edge channels and a small synthetic `normal-tree-backedge` diagnostic. The diagnostic
was not discriminative: final seed-0 runs gave AP `1.0000` for both `gin`
(`architecture_config_hash=bb49e9d0`) and `normal_tree_backedge_gnn`
(`architecture_config_hash=83e3c616`). Do not use this diagnostic as architecture evidence without
hardening it.

2026-05-26: Hardened diagnostic `cycle_matching_v4` added. Graphs are 20-node 3-regular
cycle-plus-matching graphs with constant features and random relabeling. Shortcut graph-stat
baselines did not solve it. Across seeds `[0,1,2]`, GIN and GCN stayed at val/test AP `0.4167`,
while `normal_tree_backedge_gnn` (`architecture_config_hash=f9449a35`) reached validation AP
`0.6635 +/- 0.0788` and test AP `0.5516 +/- 0.0865`. This supports the mechanism on a synthetic
diagnostic but shows weak held-out generalization; next step is multi-root/multi-order DFS averaging.

2026-05-26: Naive four-order DFS averaging was implemented with `model.num_tree_orders=4`, but did
not improve the diagnostic. The four-order config `architecture_config_hash=2fb8a3d0` reached
validation AP `0.6490 +/- 0.0864` and test AP `0.5626 +/- 0.2606`, with about 5x runtime versus the
single-order config. Keep the implementation for future experiments, but prefer a learned order
gate or a different idea-bank direction over unweighted averaging.

---

# Idea 7: ListColorGNN — Dynamic List-Coloring Channel Allocation

## Scientific hypothesis

In heterophilous graphs, neighboring vertices often should not share the same representation channel. A model inspired by list coloring can assign each node a learned list of allowed channels and update channels according to local compatibility constraints.

## Mechanism

Maintain `C` channel states per node:

```text
h_v[c] for c in 1..C
list_logits_v[c] = allowedness of channel c at node v
```

Messages are channel-aware:

```text
m_{u->v,c} = compatibility(c_u, c_v, edge_features) * h_u[c_u]
```

A lightweight first implementation can use:

- channel masks from an MLP;
- neighbor anti-correlation penalty;
- residual ordinary node embedding.

## Why this is not just a known baseline

Multi-head attention has multiple channels but does not impose a coloring/list-constraint interpretation. ListColorGNN tests whether learned incompatibility of adjacent node states helps heterophily.

## Closest known related architectures

- GAT multi-head attention.
- mixture-of-experts GNNs.
- heterophily-specific GNNs.
- signed/compatibility message passing.

## Expected insight if it succeeds

Heterophily may require explicit local state incompatibility, not just higher-order aggregation.

## Expected insight if it fails

The channel-allocation constraint may be too artificial or may collapse to ordinary attention.

## Target task family

- heterophily datasets;
- Cora/PubMed as sanity checks, not primary evidence.

## Primary baseline to beat

- heterophily baselines in the gym if available;
- `gpr_gnn` for homophily citation datasets only as a robustness check.

## Minimal falsifying experiment

Synthetic bipartite or near-bipartite node classification where neighbor labels differ. If ListColorGNN does not beat GCN/GAT there, stop.

## Confirmation protocol

Use heterophily datasets for confirmation, not only Cora/PubMed.

## Complexity/runtime risk

Medium. Channel dimension can explode. Start with `C=4` or `C=8`.

## Implementation boundary

Do not implement discrete coloring. Use differentiable soft channel lists.

---

# Idea 8: ClassFlowGNN — Conservation-Law Class Flow Network

## Scientific hypothesis

Node classification can be interpreted as routing class evidence through a graph. Instead of smoothing logits, learn class-specific flows that are approximately conserved except at evidence-injection nodes and structural sinks/sources.

## Mechanism

For each class `c`, maintain edge flow `f_ij^c` and node potential/logit `p_i^c`.

Use updates inspired by network flow:

```text
f_ij^c = EdgeFlowMLP(p_i^c, p_j^c, h_i, h_j, edge_attr)
div_i^c = sum_in f_ji^c - sum_out f_ij^c
p_i^c' = p_i^c + update(div_i^c, evidence_i^c)
```

Add optional regularizers:

```text
small divergence on unlabeled/train-unobserved nodes
capacity-like penalty on low-reliability edges
cut consistency penalty
```

## Why this is not just a known baseline

APPNP/GPR diffuse class evidence. ClassFlowGNN treats class evidence as a constrained flow with learned capacities and conservation violations.

## Closest known related architectures

- diffusion/label propagation GNNs;
- Hodge/flow neural models;
- PDE-inspired GNNs;
- energy-based graph models.

## Expected insight if it succeeds

Classification may benefit from conservation-style constraints rather than unconstrained smoothing.

## Expected insight if it fails

Conservation may be too restrictive for citation labels or may need better supervision than the gym currently provides.

## Target task family

- semi-supervised node classification;
- link prediction as secondary;
- flow-like synthetic tasks.

## Primary baseline to beat

- `gpr_gnn` for Cora/PubMed.
- APPNP as a diffusion baseline.

## Minimal falsifying experiment

Synthetic source/sink node classification where class evidence must cross cuts. If it cannot outperform GPR-like propagation on the synthetic task, stop.

## Confirmation protocol

Confirm on Cora/PubMed only after synthetic validation.

## Complexity/runtime risk

Medium. Edge states per class can be expensive. Start with hidden flow channels, not one channel per class, then map to logits.

## Implementation boundary

No custom loss at first unless necessary. Add conservation regularization only after the unconstrained version trains.

---

# Idea 9: DualShadowGNN — Primal/Dual Shadow Message Passing

## Scientific hypothesis

For planar or nearly planar graphs, many structural signals are easier to express in the dual graph: cycles in the primal correspond to cuts in the dual, and vice versa. A model that exchanges information between a graph and an approximate dual/shadow graph may better capture ring, face, and barrier structure.

## Mechanism

For planar graphs or graphs with a planarized approximation:

1. Compute a planar embedding if available.
2. Build a dual graph over faces.
3. Run message passing on both primal nodes/edges and dual face nodes.
4. Exchange messages through edge-face incidence.

For non-planar graphs, construct a local cycle-face shadow using short induced cycles as pseudo-faces.

```text
primal node states
edge states
face/pseudo-face states
edge-face incidence messages
```

## Why this is not just a known baseline

It is not merely adding cycle counts. It builds a second graph whose nodes represent faces or pseudo-faces and lets the model learn primal-dual interactions.

## Closest known related architectures

- cellular/simplicial neural networks;
- motif GNNs;
- molecular ring-aware models;
- planar duality methods.

## Expected insight if it succeeds

Dual structures may expose graph-level signals that are hard for ordinary node-edge message passing.

## Expected insight if it fails

The benchmarks may not have reliable planar structure, or pseudo-face construction may be too noisy.

## Target task family

- molecular graph prediction;
- synthetic planar graph tasks;
- ring/cycle-heavy datasets.

## Primary baseline to beat

- GINE/GIN once edge attributes are available.

## Minimal falsifying experiment

Synthetic planar graph classification where labels depend on face/cycle arrangement rather than degree distribution.

## Confirmation protocol

Do not run broad benchmarks until the synthetic planar task validates the mechanism.

## Complexity/runtime risk

High for arbitrary graphs. Medium for planar/small molecular graphs.

## Implementation boundary

First version: pseudo-face graph from chordless cycles up to length 6. Avoid full planar embedding complexity initially.

---

# Idea 10: ChordlessCycleMemoryGNN — Induced-Cycle Memory Network

## Scientific hypothesis

Induced cycles are primitive cyclic structures. Standard message passing may detect short cycles only after several layers and may confuse chorded dense neighborhoods with true rings. A network with explicit induced-cycle memory can better represent molecules, feedback motifs, and ring-rich graphs.

## Mechanism

Precompute induced cycles up to length `L`, e.g. `L <= 6` or `L <= 8`.

Create cycle memory tokens:

```text
c_C = pool({h_v : v in cycle C}, {edge_attr_e : e in C})
```

Exchange messages:

```text
node -> cycle
cycle -> node
edge -> cycle
cycle -> edge
```

Use cycle tokens as graph-level readout features.

## Why this is not just a known baseline

Motif GNNs often count or aggregate motif features. This model maintains recurrent memory states for specific induced cycles and exchanges messages with their participating nodes/edges.

## Closest known related architectures

- motif-aware GNNs;
- ring-aware molecular GNNs;
- subgraph GNNs;
- cellular/simplicial networks.

## Expected insight if it succeeds

Induced cycles may be useful computational objects, not just scalar structural features.

## Expected insight if it fails

Cycle memories may be redundant with edge-aware GIN/GINE or too sparse/noisy on benchmark graphs.

## Target task family

- molecular graph prediction;
- synthetic cycle tasks;
- possibly citation networks as secondary.

## Primary baseline to beat

- GINE/GIN on molecular tasks.

## Minimal falsifying experiment

Synthetic task distinguishing chordless cycles from chorded cliques with the same node/edge counts. If the model cannot solve this better than GIN, stop.

## Confirmation protocol

MolHIV/MolPCBA after synthetic validation and edge-attribute support.

## Complexity/runtime risk

Medium. Cycle enumeration can blow up. Strictly cap cycle length and max cycles per graph.

## Implementation boundary

Do not enumerate all cycles. Enumerate bounded induced cycles only.

---

# Idea 11: RegularPatchGNN — Dense-Regularity Patch Network

## Scientific hypothesis

Dense regions of a graph may behave like approximately regular patches, where individual edges matter less than cross-density patterns between vertex groups. A model that learns patch-level density interactions may capture dense substructures more efficiently than node-level message passing.

## Mechanism

Partition nodes into patches using a learned or heuristic clustering method. Compute pairwise patch statistics:

```text
edge density between patches
degree distribution summaries
feature distribution summaries
label/evidence summaries if allowed by split discipline
```

Run patch-level message passing, then project back to nodes.

```text
patch_state' = PatchMP(patch_state, density_matrix)
h_v' = combine(h_v, patch_state[patch(v)])
```

## Why this is not just a known baseline

Pooling methods coarsen graphs; RegularPatchGNN specifically treats patch-pair edge densities as first-class features and tests a regularity-style view of dense graphs.

## Closest known related architectures

- graph pooling/coarsening;
- stochastic block model neural approaches;
- graph transformers with cluster tokens.

## Expected insight if it succeeds

Some benchmark graphs may benefit from density-level summaries rather than edge-level local propagation.

## Expected insight if it fails

The regularity abstraction may be too coarse for node-level labels or the tested graphs may be too sparse.

## Target task family

- dense synthetic graphs;
- social/collaboration graphs;
- graph-level tasks with dense motifs.

## Primary baseline to beat

- dataset-specific baselines, not necessarily Cora/PubMed.

## Minimal falsifying experiment

Synthetic stochastic block or planted dense-subgraph tasks.

## Confirmation protocol

Only test real datasets after synthetic patch tasks show clear value.

## Complexity/runtime risk

Medium to high depending on clustering.

## Implementation boundary

Start with deterministic clustering and fixed patch count. Do not learn clustering initially.

---

# Idea 12: ObstructionTokenGNN — Forbidden-Minor/Obstruction Token Network

## Scientific hypothesis

Small obstruction patterns such as triangles, K4-like minors, K2,3-like structures, bridges, and low-order separators can act as reusable structural tokens. A GNN that represents these obstructions as tokens may learn graph families and structural complexity more directly than plain message passing.

## Mechanism

Detect a bounded set of small patterns approximately:

- triangles;
- chordless cycles;
- K4 subgraphs or K4-minor proxies;
- K2,3/K3,3-like bipartite motifs for small sizes;
- articulation/bridge obstructions to 2-connectivity.

Create obstruction tokens and connect them to participating nodes/edges. Run a heterogeneous message-passing network:

```text
node tokens
edge tokens
obstruction tokens
```

## Why this is not just a known baseline

Graphlet features count motifs. ObstructionTokenGNN treats structural obstructions as message-passing entities, allowing the model to learn how obstructions interact with node/edge states.

## Closest known related architectures

- graphlet/motif GNNs;
- subgraph GNNs;
- heterogeneous token graph models;
- graph transformer structural tokens.

## Expected insight if it succeeds

Small graph-theoretic obstructions may be useful intermediate objects for neural graph reasoning.

## Expected insight if it fails

The selected obstruction vocabulary may be incomplete, too expensive, or no better than generic subgraph tokens.

## Target task family

- synthetic graph-property tasks;
- molecular graph tasks;
- graph classification tasks involving planarity, connectivity, or ring systems.

## Primary baseline to beat

- GIN/GINE;
- subgraph GNN baseline if implemented.

## Minimal falsifying experiment

Synthetic graph classification with same local degree statistics but different obstruction tokens. If token model does not help, stop.

## Confirmation protocol

Only confirm on real benchmarks after synthetic obstruction tasks succeed.

## Complexity/runtime risk

Medium to high. Pattern detection can be expensive.

## Implementation boundary

Start with a tiny vocabulary: bridges, articulation points, triangles, chordless 4-cycles, K4 subgraphs. Do not attempt arbitrary minor detection first.

---

# Suggested first wave

The first wave should favor bounded, implementable ideas that still test genuine mechanisms:

1. `SepBottleneckGNN`
2. `NormalTreeBackedgeGNN`
3. `TreePackGNN`
4. `ChordlessCycleMemoryGNN`
5. `CycleCutGNN-lite`

The second wave should begin after edge-attribute support and config-level aggregation are stable:

1. `ClassFlowGNN`
2. `BranchSetGNN`
3. `DualShadowGNN`
4. `BagAutomatonGNN`
5. `RegularPatchGNN`
6. `ObstructionTokenGNN`

---

# Additional synthetic tasks to add to the gym

These tasks are useful because many novel mechanisms will not reveal their value on Cora/PubMed first.

## Separator task

Generate graphs with the same degree distribution but different separator structure. Predict whether two marked nodes are separated by a small vertex cut.

Useful for:

- SepBottleneckGNN
- BagAutomatonGNN
- ObstructionTokenGNN

## Cycle/cut task

Generate graphs with matched local degrees but different cycle-space structure. Predict cycle rank bucket, chordless-cycle presence, or whether an edge lies in every s-t cut.

Useful for:

- CycleCutGNN
- ChordlessCycleMemoryGNN
- ClassFlowGNN

## Minor-stability task

Generate graphs where labels are invariant under contraction of pendant trees or local cliques. Test whether a model can ignore contractible noise.

Useful for:

- BranchSetGNN
- ObstructionTokenGNN

## Tree-decomposition task

Generate bounded-treewidth graphs where the label depends on a pattern distributed across bags.

Useful for:

- BagAutomatonGNN
- NormalTreeBackedgeGNN

## Planar-dual task

Generate planar graphs where labels depend on face adjacency or dual cuts, not only primal node neighborhoods.

Useful for:

- DualShadowGNN
- CycleCutGNN

---

# Scoring ideas beyond validation accuracy

For each architecture, record:

```text
Did it beat the primary baseline? yes/no
Did it solve the synthetic mechanism task? yes/no
Did it expose a measurable structural quantity? yes/no
Did it fail cleanly enough to teach something? yes/no
Does it require trainer/evaluator changes? yes/no
Runtime multiplier vs baseline
Parameter multiplier vs baseline
```

An idea can be valuable even if it fails on Cora/PubMed, provided it cleanly tests a graph-theoretic mechanism and updates `INSIGHTS.md` with a durable conclusion.
