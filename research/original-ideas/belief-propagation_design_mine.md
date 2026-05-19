Below are architecture ideas that treat the belief-propagation document as a design mine, not as something to merely imitate. The important shift is: instead of thinking “GNN layer = aggregate neighbors,” think “graph learning = approximate inference, energy minimization, loop correction, uncertainty propagation, region reasoning, and belief revision.” The BP document gives the conceptual ingredients: local messages, beliefs, factor graphs, loopy inference, Bethe/Kikuchi approximations, and generalized message passing over regions rather than only nodes. 

For GNN Gym, the constraint is that these should mostly be implemented as **encoders** compatible with your existing task heads and trainers. Your project spec explicitly separates dataset adapter, model encoder, task head/trainer, and evaluator, and expects new models to be added via a model file, YAML config, and shape test rather than rewriting the pipeline. 

---

# Core architectural direction

The strongest design principle is:

> Replace “node embeddings” with “approximate beliefs,” and replace “one-shot aggregation” with “iterative belief revision under graph constraints.”

GCN, GAT, and GIN are mostly **diffusion / attention / multiset aggregation** models. The BP-inspired models below are instead based on:

* directional messages,
* variable-factor duality,
* region-level reasoning,
* loop correction,
* iterative convergence,
* uncertainty over hidden states,
* energy minimization,
* local-to-global consistency.

These are plausible competitors because they attack known weaknesses of standard GNNs: oversmoothing, heterophily, long-range dependencies, cycles, poor uncertainty representation, and weak higher-order structure.

---

# 1. Cavity Message Network

**Name:** `cavity_gnn`

**Core idea:**
Standard GNNs usually send a transformed version of node (i)'s embedding to all neighbors. Belief propagation instead uses **cavity messages**: the message from (i) to (j) should summarize everything (i) knows **except the information that came from (j)**.

This gives a directional edge-state GNN:

[
m_{i \to j}^{t+1}
=================

\phi_\theta
\left(
x_i,
\sum_{k \in N(i)\setminus j} m_{k \to i}^{t},
e_{ij}
\right)
]

Then node embeddings are read out from incoming messages:

[
h_i^T =
\rho_\theta
\left(
x_i,
\sum_{j \in N(i)} m_{j \to i}^T
\right)
]

**Why it is interesting:**
This is closer to real BP than most message-passing GNNs. It avoids immediate echo effects like:

[
i \to j \to i
]

which contribute to oversmoothing and redundant self-reinforcement.

**Expected strengths:**

* Heterophily datasets: Roman-empire, Amazon-ratings.
* Link prediction: ogbl-collab.
* Graphs where cycles create message echo.

**Main risk:**
Memory cost is edge-level rather than node-level. For ogbn-products this may require neighbor sampling.

**Implementation path in GNN Gym:**

```text
src/gnn_gym/models/cavity_gnn.py
configs/models/cavity_gnn.yaml
tests/test_model_shapes.py
```

Useful config:

```yaml
model:
  name: cavity_gnn
  hidden_channels: 128
  num_steps: 4
  message_mlp_layers: 2
  update: gru
  dropout: 0.2
  residual: true
```

This is one of the best first candidates because it is novel enough, implementable, and directly BP-inspired.

---

# 2. BetheNet: A Free-Energy GNN

**Name:** `bethe_net`

**Core idea:**
Instead of only learning node embeddings, learn **node beliefs** and **edge beliefs** and regularize them to be locally consistent.

Each node has a belief vector:

[
b_i
]

Each edge has a pairwise belief:

[
b_{ij}
]

The model updates both. It is trained not only with the task loss, but also with a **Bethe-style consistency loss**:

[
\mathcal{L}_{cons}
==================

\sum_{(i,j)}
\left|
\operatorname{marginalize}*j(b*{ij}) - b_i
\right|^2
+
\left|
\operatorname{marginalize}*i(b*{ij}) - b_j
\right|^2
]

You are not forced to use literal probability distributions. The “beliefs” can be latent vectors normalized with softmax, sigmoid, or layer norm.

**Why it is interesting:**
GCN/GAT/GIN do not explicitly enforce consistency between node states and edge states. BetheNet would learn representations that must agree locally across graph structure.

**Expected strengths:**

* Molecular graph prediction, because bonds/edges matter.
* Link prediction, because pairwise beliefs are native.
* Long-range tasks, if consistency is propagated iteratively.

**Main risk:**
The consistency loss may hurt if too strong. It should start with a small coefficient.

**Implementation detail:**
Return node embeddings for the usual heads, but internally maintain edge beliefs.

```python
z = model.encoder(x, edge_index, edge_attr, batch)
```

The trainer does not need to know about the consistency loss at first. To keep GNN Gym clean, you can expose an optional `aux_loss()` method and let trainers add it only if present.

---

# 3. Kikuchi Region Network

**Name:** `kikuchi_gnn`

**Core idea:**
The BP document discusses generalized belief propagation, where messages are not only between individual nodes but between **regions**. Translate that into a GNN where the model constructs region nodes:

* triangles,
* 2-hop paths,
* ego-nets,
* cycles,
* molecular rings,
* high-degree neighborhoods,
* sampled subgraphs.

Then run bipartite message passing:

```text
node → region → node
```

Instead of:

```text
node → node
```

A region has an embedding:

[
r_C = \rho_\theta({h_i : i \in C})
]

and sends messages back to its member nodes:

[
h_i^{t+1}
=========

U_\theta
\left(
h_i^t,
\sum_{C \ni i} M_\theta(r_C^t, h_i^t)
\right)
]

**Why it is interesting:**
GIN is powerful because it aggregates multisets, but it is still fundamentally node-neighborhood based. KikuchiGNN gives the model direct access to higher-order objects.

**Expected strengths:**

* Peptides-func and Peptides-struct.
* Molecules with rings and functional groups.
* Heterophily datasets where local edges alone are misleading.
* Any dataset where motifs matter.

**Main risk:**
Region construction can become expensive or dataset-specific. Start with generic regions:

```text
1-hop ego region
2-hop ego region
short cycle region
triangle region if available
```

**Minimal version:**
Do not enumerate all regions. Sample a fixed number of regions per node.

**Config:**

```yaml
model:
  name: kikuchi_gnn
  hidden_channels: 128
  num_layers: 3
  region_types: ["ego1", "ego2", "triangles"]
  max_regions_per_node: 8
  region_pooling: attention
  dropout: 0.2
```

This is the most conceptually ambitious architecture in the list.

---

# 4. Loop-Corrected GNN

**Name:** `loop_corrected_gnn`

**Core idea:**
BP is exact on trees and approximate on loopy graphs. Most real graphs are loopy. So instead of ignoring loops, explicitly model them.

Pipeline:

1. Run a base GNN layer.
2. Detect or sample short cycles.
3. Compute a loop embedding for each cycle.
4. Send correction messages from loops back to nodes/edges.

For a cycle (C):

[
c_C = \operatorname{Pool}_{i \in C}(h_i)
]

Then update nodes:

[
h_i^{t+1}
=========

h_i^{t}
+
\sum_{C \ni i}
\alpha_{iC} W c_C
]

**Why it is interesting:**
Standard GNNs process loops only implicitly through repeated local aggregation. This architecture treats loops as first-class computational objects.

**Expected strengths:**

* Molecular graphs with rings.
* Citation graphs with community cycles.
* Link prediction.
* Datasets where cycles encode semantics.

**Main risk:**
Cycle enumeration is expensive. Use bounded cycle search:

```text
triangles
4-cycles
sampled simple cycles up to length 6
```

**Implementation shortcut:**
Use NetworkX preprocessing for small datasets. For large datasets, approximate with random walks returning to origin.

---

# 5. SurveyGNN: Multi-Belief Graph Network

**Name:** `survey_gnn`

**Core idea:**
In hard graphical models, a node may not have one clean belief; it may have several competing possible states. Survey propagation handles distributions over beliefs. Translate that into a GNN where each node carries (K) belief particles:

[
h_i \in \mathbb{R}^{K \times d}
]

Messages update a set of hypotheses, not a single embedding:

[
H_i^{t+1}
=========

\operatorname{ParticleUpdate}_\theta
\left(
H_i^t,
{H_j^t : j \in N(i)}
\right)
]

Add a diversity term so the particles do not collapse:

[
\mathcal{L}_{div}
=================

*

\sum_i
\operatorname{Var}(H_i)
]

The final embedding is either:

```text
mean over particles
attention over particles
max-confidence particle
```

**Why it is interesting:**
GCN/GAT/GIN compress each node into one vector. SurveyGNN lets a node represent ambiguity.

**Expected strengths:**

* Heterophily.
* Semi-supervised node classification.
* Noisy labels.
* Graphs with multiple local explanations.

**Main risk:**
Can become unstable if particles collapse or diverge. Use small (K), e.g. 4 or 8.

**Config:**

```yaml
model:
  name: survey_gnn
  hidden_channels: 64
  num_particles: 4
  num_layers: 4
  particle_attention: true
  diversity_weight: 0.01
```

This is a good “imaginative but implementable” candidate.

---

# 6. Temperature-Ladder GNN

**Name:** `temp_ladder_gnn`

**Core idea:**
Run multiple parallel GNN streams at different “temperatures.”

High temperature:

* smooth,
* exploratory,
* global,
* less confident.

Low temperature:

* sharp,
* local,
* discriminative,
* confident.

Each layer has (K) replicas:

[
h_i^{t, \tau_1}, \ldots, h_i^{t, \tau_K}
]

The aggregation uses temperature-scaled attention or softmax:

[
\alpha_{ij}^{\tau}
==================

\operatorname{softmax}
\left(
s(h_i,h_j)/\tau
\right)
]

Then replicas exchange information:

[
h_i^{\tau_a}
\leftarrow
h_i^{\tau_a}
+
G_{\theta}
(h_i^{\tau_a}, h_i^{\tau_b})
]

**Why it is interesting:**
This gives the model both smooth and sharp propagation regimes. GCN is often too smooth; GAT can be too locally selective. A temperature ladder can interpolate.

**Expected strengths:**

* Cora/PubMed/ogbn-arxiv.
* Heterophily if low-temperature channels learn to reject bad neighbors.
* Long-range graph tasks if high-temperature channels propagate broader signals.

**Main risk:**
Could just become an expensive ensemble unless exchange gates are well-designed.

**Minimal version:**
Use three temperatures:

```text
τ = [0.5, 1.0, 2.0]
```

---

# 7. Non-Backtracking Belief GNN

**Name:** `nb_belief_gnn`

**Core idea:**
Use directed edge states and forbid immediate backtracking. A message from (i) to (j) aggregates incoming messages from (k \to i) where (k \neq j).

This is similar to CavityGNN, but specifically based on the **non-backtracking operator**.

[
m_{i \to j}^{t+1}
=================

\sum_{k \in N(i), k \neq j}
W m_{k \to i}^{t}
]

Then read out node states from incoming directed messages.

**Why it is interesting:**
This is a clean way to prevent short feedback loops. It may preserve useful high-frequency information better than GCN-style smoothing.

**Expected strengths:**

* Heterophily.
* Link prediction.
* Graphs with many triangles/cycles.

**Main risk:**
Edge memory. But simpler than full CavityGNN.

**Why implement it early:**
It is probably the most controlled BP-inspired baseline: simple, mathematically motivated, and testable.

---

# 8. Belief Revision GNN

**Name:** `revision_gnn`

**Core idea:**
Start with an MLP belief from node features, then iteratively revise it using graph evidence.

Initial belief:

[
b_i^0 = \operatorname{MLP}(x_i)
]

Graph revision:

[
\Delta b_i^t =
F_\theta
\left(
b_i^t,
\operatorname{AGG}*{j \in N(i)} G*\theta(b_i^t,b_j^t,e_{ij})
\right)
]

Update:

[
b_i^{t+1}
=========

b_i^t
+
\lambda_i^t \Delta b_i^t
]

where (\lambda_i^t) is a learned trust gate.

**Why it is interesting:**
Many datasets have strong node features. A naive GNN can corrupt good feature-only predictions by overusing neighbors. RevisionGNN asks: “When should graph evidence revise the feature belief?”

**Expected strengths:**

* Cora, PubMed, ogbn-arxiv.
* Heterophily, if graph evidence is selectively trusted.
* Any dataset where MLP is competitive.

**Main risk:**
May behave like residual GCN unless the trust gate is meaningful.

**Important diagnostic:**
Log average trust gate per dataset. If Roman-empire gets low neighbor trust and Cora gets high neighbor trust, the model is doing something interpretable.

---

# 9. Frustration-Aware GNN

**Name:** `frustration_gnn`

**Core idea:**
In graphical models, neighboring variables may be compatible or incompatible. Standard GNNs often assume neighbors are useful. This fails in heterophily.

Learn an edge frustration score:

[
f_{ij} =
\sigma
\left(
\operatorname{MLP}([h_i,h_j,e_{ij}])
\right)
]

Then split messages into agreement and disagreement channels:

[
m_{ij}^{agree} = (1-f_{ij}) W_a h_j
]

[
m_{ij}^{conflict} = f_{ij} W_c h_j
]

Node update:

[
h_i^{t+1}
=========

U_\theta
\left(
h_i^t,
\sum_j m_{ij}^{agree},
\sum_j m_{ij}^{conflict}
\right)
]

**Why it is interesting:**
This directly targets heterophily. Instead of pretending all neighbors should be smoothed together, the model learns which edges are “supportive” and which are “frustrating.”

**Expected strengths:**

* Roman-empire.
* Amazon-ratings.
* Possibly ogbn-products, where co-purchase edges may not imply same class.

**Main risk:**
The frustration score may become noisy without supervision. Add entropy regularization or initialize conservatively.

---

# 10. Graph Decimation Network

**Name:** `decimation_gnn`

**Core idea:**
Inspired by inference algorithms that progressively fix confident variables. The model alternates between:

1. estimating beliefs,
2. selecting confident nodes/edges,
3. clamping them,
4. propagating again.

For node classification:

```text
round 1: predict all nodes
round 2: clamp high-confidence pseudo-beliefs
round 3: revise uncertain nodes using clamped confident nodes
```

This is not ordinary message passing; it is closer to iterative symbolic/probabilistic solving.

**Why it is interesting:**
Semi-supervised node classification naturally has a small labeled set and many unlabeled nodes. Decimation can act like learned label propagation but with confidence control.

**Expected strengths:**

* Cora/PubMed.
* ogbn-arxiv.
* Maybe link prediction if high-confidence links are clamped.

**Main risk:**
Transductive leakage must be handled carefully. During training, it should not use validation/test labels, only predictions.

**Implementation note:**
This may require trainer support if it uses pseudo-label dynamics, so it violates the “pure encoder” ideal more than other proposals. Keep it as a later experiment.

---

# 11. Walk-Belief Transformer

**Name:** `walk_belief_transformer`

**Core idea:**
Do not pass messages on edges. Sample walks, treat each walk as a sentence, run a transformer over walk tokens, then scatter updated representations back to nodes.

Pipeline:

```text
graph → sampled walks → transformer over walks → node/edge aggregation → task head
```

Each token is:

```text
node feature + degree feature + positional index + return/branch markers
```

**Why it is interesting:**
Message-passing GNNs are local and synchronous. Walk transformers can model sequential dependencies, return events, cycles, and path motifs.

**Expected strengths:**

* Link prediction.
* Long-range graph tasks.
* Peptides.
* Graph-level molecular tasks if walks capture functional structure.

**Main risk:**
Sampling variance and compute cost. It may be weaker than GPS/Graphormer-style models unless carefully tuned.

**Minimal version:**
Use short random walks of length 8 or 16 and a small transformer.

---

# 12. Dual-Primal Factor GNN

**Name:** `dual_primal_gnn`

**Core idea:**
Create a second graph whose nodes are edges, motifs, or constraints of the original graph. Then run coupled message passing on:

1. the primal graph: original nodes,
2. the dual graph: edges/regions/factors,
3. cross-links between primal and dual objects.

For a normal graph:

```text
Primal node: original node
Dual node: original edge or motif
Cross edge: node belongs to edge/motif
```

This generalizes line-graph GNNs and factor graphs.

**Why it is interesting:**
GCN/GAT/GIN mostly represent nodes. GIN is strong for graph-level tasks, but it still lacks explicit factor objects. Dual-primal GNNs can represent interactions as first-class states.

**Expected strengths:**

* Molecules.
* Link prediction.
* Constraint-like graph structures.
* Heterophily, if edge/factor states learn relation types.

**Main risk:**
Graph transformation overhead. But it is highly compatible with PyG using augmented `edge_index`.

---

# 13. Entropy-Gated GNN

**Name:** `entropy_gated_gnn`

**Core idea:**
Each node carries an uncertainty estimate. Uncertain nodes listen more; confident nodes speak more.

For each node:

[
u_i = \operatorname{entropy}(b_i)
]

or learned uncertainty:

[
u_i = \sigma(\operatorname{MLP}(h_i))
]

Message weighting:

[
m_{j \to i}
===========

(1-u_j) \cdot u_i \cdot W h_j
]

Interpretation:

```text
confident sender → uncertain receiver = strong message
uncertain sender → confident receiver = weak message
```

**Why it is interesting:**
Standard GNNs do not distinguish confident and uncertain nodes. BP naturally works with belief strength; this model makes that explicit.

**Expected strengths:**

* Semi-supervised node classification.
* Noisy graphs.
* Datasets where node features are unevenly informative.

**Main risk:**
Uncertainty can collapse. Use calibration diagnostics.

---

# 14. Neural Region Collapse Network

**Name:** `region_collapse_gnn`

**Core idea:**
The model dynamically compresses parts of the graph into temporary supernodes, reasons over the compressed graph, then expands back.

Steps:

1. Learn soft assignments from nodes to regions:

[
S \in \mathbb{R}^{N \times R}
]

2. Pool nodes into region states:

[
R = S^\top H
]

3. Run message passing among regions.
4. Scatter region information back to nodes:

[
H' = H + S R'
]

**Why it is interesting:**
This is a learned approximation to region-based inference and graph coarsening. It may help with long-range dependencies without deep message passing.

**Expected strengths:**

* Peptides-func / Peptides-struct.
* ogbn-arxiv.
* ogbn-products.
* Large graphs where pure local propagation is insufficient.

**Main risk:**
Can become a generic pooling model. The key is to make regions dynamic and task-conditioned.

---

# 15. Equilibrium Belief GNN

**Name:** `equilibrium_belief_gnn`

**Core idea:**
Instead of stacking a fixed number of layers, define one recurrent update and iterate until approximate convergence:

[
H^{t+1} = F_\theta(H^t, X, A)
]

The output is a fixed point:

[
H^* = F_\theta(H^*, X, A)
]

This is closer to BP, where messages are iterated until convergence.

**Why it is interesting:**
Depth becomes adaptive. Easy graphs may converge in 3 steps; hard graphs may require 20. You can also add a convergence penalty.

**Expected strengths:**

* Long-range tasks.
* Graphs with variable structural complexity.
* Datasets where fixed-depth GNNs are too shallow or oversmooth.

**Main risk:**
Training can be unstable. Start with unrolled recurrence before using implicit differentiation.

**Minimal config:**

```yaml
model:
  name: equilibrium_belief_gnn
  hidden_channels: 128
  max_steps: 10
  tolerance: 1e-3
  update: gru
  residual_weight: 0.5
```

---

# Ranking: which ones should you actually implement first?

| Rank | Architecture             |     Novelty | Implementation difficulty | Best benchmark targets       | Why                                         |
| ---: | ------------------------ | ----------: | ------------------------: | ---------------------------- | ------------------------------------------- |
|    1 | `cavity_gnn`             |        High |                    Medium | heterophily, link prediction | Direct BP analogue; clean hypothesis        |
|    2 | `frustration_gnn`        | Medium-high |                Low-medium | Roman-empire, Amazon-ratings | Directly attacks GCN/GAT weakness           |
|    3 | `survey_gnn`             |        High |                    Medium | heterophily, noisy graphs    | Multi-belief states are genuinely different |
|    4 | `kikuchi_gnn`            |   Very high |                      High | molecules, peptides          | Best higher-order architecture              |
|    5 | `nb_belief_gnn`          | Medium-high |                    Medium | link prediction, heterophily | Simpler version of cavity messages          |
|    6 | `bethe_net`              |   Very high |                      High | molecules, links             | Strong theory, but harder to tune           |
|    7 | `revision_gnn`           |      Medium |                       Low | citation graphs              | Strong practical baseline                   |
|    8 | `temp_ladder_gnn`        |        High |                    Medium | broad                        | Interesting but may look ensemble-like      |
|    9 | `loop_corrected_gnn`     |   Very high |                      High | molecules, cycles            | Potentially strong but preprocessing-heavy  |
|   10 | `equilibrium_belief_gnn` |        High |               Medium-high | long-range                   | Good research idea, harder training         |

My recommendation: start with **three architectures**, not fifteen.

## First implementation batch

### 1. `revision_gnn`

This is the easiest strong custom baseline. It tests whether learned belief revision beats ordinary aggregation.

### 2. `frustration_gnn`

This gives you a clear heterophily story and should be easy to compare against GCN/GAT/GIN.

### 3. `cavity_gnn` or `nb_belief_gnn`

This is the most faithful BP-inspired architecture and gives your benchmark a distinctive direction.

These three already cover:

```text
feature-first belief revision
compatibility/conflict-aware propagation
directional cavity-style inference
```

That is a coherent research theme.

---

# How I would position the research contribution

A possible framing:

> We benchmark a family of belief-inspired graph neural architectures that reinterpret graph representation learning as approximate inference. Unlike standard GCN, GAT, and GIN models, which aggregate node neighborhoods directly, these models incorporate cavity messages, local compatibility/frustration, multi-belief node states, and region-level inference inspired by belief propagation and its generalizations.

This is more defensible than claiming “new GNN architecture” one by one. The family has a common thesis.

---

# Minimal GNN Gym integration plan

Given your spec, each architecture should be an encoder compatible with:

```python
z = encoder(x, edge_index, edge_attr=None, batch=None)
```

and then reuse the existing heads:

```text
NodeClassificationHead
GraphClassificationHead
GraphRegressionHead
DotProductLinkPredictionHead
MLPLinkPredictionHead
```

That preserves fair comparison across task families, which your GNN Gym spec explicitly prioritizes. 

Add one model at a time:

```text
src/gnn_gym/models/revision_gnn.py
configs/models/revision_gnn.yaml
tests/test_model_shapes.py

src/gnn_gym/models/frustration_gnn.py
configs/models/frustration_gnn.yaml
tests/test_model_shapes.py

src/gnn_gym/models/cavity_gnn.py
configs/models/cavity_gnn.yaml
tests/test_model_shapes.py
```

Then run:

```bash
uv run gnngym train --model revision_gnn --dataset cora --seed 0
uv run gnngym train --model frustration_gnn --dataset roman-empire --seed 0
uv run gnngym train --model cavity_gnn --dataset ogbl-collab --seed 0
```

---

# Concrete experiment matrix

Do not start with the full benchmark. Use targeted tests.

| Hypothesis                                            | Model             | Dataset                                     |
| ----------------------------------------------------- | ----------------- | ------------------------------------------- |
| Feature beliefs should be revised, not overwritten    | `revision_gnn`    | Cora, PubMed, ogbn-arxiv                    |
| Neighbor conflict should be modeled explicitly        | `frustration_gnn` | Roman-empire, Amazon-ratings                |
| Directional cavity messages reduce echo/oversmoothing | `cavity_gnn`      | Roman-empire, ogbl-collab                   |
| Multi-belief states help ambiguity                    | `survey_gnn`      | Roman-empire, Amazon-ratings                |
| Region reasoning helps graph-level tasks              | `kikuchi_gnn`     | ogbg-molhiv, Peptides-func, Peptides-struct |

---

# The most promising architecture in one sentence

If I had to choose one architecture with the best balance of novelty, implementability, and conceptual clarity, I would choose:

> **Cavity Message Network:** a directed-edge GNN where each message excludes information coming from its recipient, turning BP’s cavity principle into a learnable neural encoder.

This is meaningfully different from GCN, GAT, and GIN, but still implementable inside GNN Gym without changing the whole pipeline.
