# Recursive Improvement Graph Networks (RIGN)

A graph neural network architecture inspired by latent recursive reasoning, hierarchical reasoning, and tiny recursive models.

## 1. Project summary

This project proposes **Recursive Improvement Graph Networks (RIGN)**: a recurrent graph neural network architecture that treats graph reasoning as an iterative process of **latent computation followed by explicit answer correction**.

The starting point is the observation that standard GNNs such as GCN, GAT, and GIN usually perform a fixed number of feed-forward message-passing layers. This works well for many local or moderately structural tasks, but it is limited when the target function requires multi-step reasoning, long-range propagation, constraint satisfaction, or iterative correction.

RIGN adapts three insights from recent recursive reasoning models:

1. **Effective depth can be obtained by reusing a small computation block recurrently.**
2. **A model should separate the current solution state from a latent reasoning state.**
3. **Each recurrent step should be trained and evaluated as an improvement step, not merely as more compute.**

In RIGN, each node maintains two coupled states:

- `y_v`: a **solution state**, directly decoded into the current prediction.
- `z_v`: a **latent reasoning state**, used as a scratchpad for graph computation and message passing.

At each recursive refinement step, the model first updates the latent reasoning state through several message-passing micro-steps. It then updates the solution state through a residual correction. The output is decoded from the solution state at every refinement step, allowing deep supervision and step-wise improvement analysis.

The core claim is not simply that recurrence helps. The more precise claim is:

> A GNN can benefit from recurrence when the recurrence is structured as directed improvement of a current graph-level or node-level solution, with explicit separation between latent reasoning and decoded prediction.

This distinction matters because recurrent computation can easily become dead compute: extra message-passing steps that consume resources without improving the prediction.

---

## 2. Motivation

### 2.1 Limitations of standard GNNs

Most GNNs follow the template:

```text
h_v^{k+1} = UPDATE(h_v^k, AGGREGATE({h_u^k : u in N(v)}))
```

After `K` layers, the final node representation `h_v^K` is decoded.

This design has several known limitations:

- **Fixed computation budget:** every graph receives the same number of message-passing layers.
- **Oversmoothing:** deep message passing can make node representations too similar.
- **Oversquashing:** long-range information may be compressed through narrow graph bottlenecks.
- **Weak iterative correction:** the model does not explicitly maintain and revise a current candidate answer.
- **No direct notion of improvement:** additional layers are not required to improve the output relative to earlier layers.

For many graph reasoning tasks, especially pathfinding, reachability, constraint propagation, algorithmic graph tasks, and graph repair, the desired computation is naturally iterative. A model should be able to revise an answer after seeing the consequences of previous local computations.

### 2.2 Inspiration from HRM and TRM

The **Hierarchical Reasoning Model (HRM)** uses two recurrent modules operating at different update frequencies: a high-level module for slower abstract planning and a low-level module for faster detailed computation. Its broader lesson is that effective reasoning may require many sequential computational steps, and that recurrence can provide such depth without proportionally increasing parameters.

The **Tiny Recursive Model (TRM)** simplifies this idea. Instead of relying on a biologically motivated hierarchy, it interprets the two states more directly:

- one state stores the current solution;
- one state stores latent reasoning information.

This is the more useful abstraction for GNN design. For graphs, the analogous design is to maintain:

- a decoded candidate prediction over nodes, edges, or the whole graph;
- a latent graph reasoning state that performs message passing conditioned on the current prediction.

### 2.3 Core project hypothesis

The project hypothesis is:

> A recurrent GNN with separate solution and reasoning states, trained with deep supervision and an improvement-oriented loss, will outperform standard feed-forward GNNs and naive recurrent GNNs on graph tasks requiring multi-step reasoning, while remaining parameter-efficient.

The hypothesis should be tested carefully. If RIGN only beats shallow GCN/GAT/GIN models but not compute-matched recurrent or deep baselines, then the architecture is not sufficiently justified.

---

## 3. Architecture overview

Let `G = (V, E)` be a graph with node features `x_v`, optional edge features `e_uv`, and task target `t`.

RIGN maintains, for each node `v`:

- `x_v`: fixed input features;
- `y_v^s`: solution state at macro-step `s`;
- `z_v^s`: latent reasoning state at macro-step `s`.

Optionally, for graph-level tasks it also maintains:

- `g^s`: graph-level context state.

The model iterates for `S` macro-steps. Each macro-step contains:

1. **Latent graph reasoning:** update `z` through `R` recurrent message-passing micro-steps.
2. **Solution correction:** update `y` using the final latent state `z`.
3. **Optional graph context update:** update `g` using a readout over `y` and `z`.
4. **Intermediate prediction:** decode the current prediction from `y` or from `g`.

The output head always reads from the solution state, not directly from the latent reasoning state.

---

## 4. State initialization

For each node:

```math
y_v^0 = \phi_y(x_v)
```

```math
z_v^0 = \phi_z(x_v)
```

For graph-level tasks:

```math
g^0 = \operatorname{READOUT}(\{y_v^0 : v \in V\})
```

The two initializers may share a first projection layer, but the final projections should be separate. This encourages `y` and `z` to specialize:

- `y` should become prediction-aligned;
- `z` should remain free to encode latent reasoning features.

---

## 5. Recursive refinement block

### 5.1 Latent message-passing micro-steps

At macro-step `s`, initialize:

```math
z_v^{s,0} = z_v^{s-1}
```

For micro-step `r = 1, ..., R`, compute messages:

```math
m_v^{s,r} = \operatorname{AGG}_{u \in \mathcal{N}(v)}
\phi_m(z_u^{s,r-1}, y_u^{s-1}, x_u, e_{uv})
```

Then update the latent reasoning state:

```math
z_v^{s,r} = \operatorname{GRU}_z
\left(
    z_v^{s,r-1},
    \phi_z(z_v^{s,r-1}, y_v^{s-1}, x_v, m_v^{s,r}, g^{s-1})
\right)
```

At the end of the micro-steps:

```math
\tilde{z}_v^s = z_v^{s,R}
```

This phase is where graph computation happens. The latent state receives neighborhood information while being conditioned on the current solution state. This allows message passing to ask: given the current answer, what information should be propagated to improve it?

### 5.2 Aggregation choices

A first implementation should use a GIN-style sum aggregator with a learned gate:

```math
m_v^{s,r} = \sum_{u \in \mathcal{N}(v)}
\alpha_{uv}^{s,r} \cdot \phi_m(z_u^{s,r-1}, y_u^{s-1}, x_u, e_{uv})
```

where:

```math
\alpha_{uv}^{s,r} = \sigma\left(\phi_\alpha(z_v^{s,r-1}, z_u^{s,r-1}, y_v^{s-1}, y_u^{s-1}, e_{uv})\right)
```

This is a compromise between GIN and GAT:

- Like GIN, it preserves expressive sum aggregation.
- Like GAT, it can select which neighbors matter.
- Unlike full attention, it is simple and cheaper.

A later version can replace the gate with multi-head graph attention.

### 5.3 Solution correction

After the latent micro-steps, update the solution state by a gated residual correction:

```math
\Delta y_v^s = \phi_\Delta(y_v^{s-1}, \tilde{z}_v^s, x_v, g^{s-1})
```

```math
\gamma_v^s = \sigma(\phi_\gamma(y_v^{s-1}, \tilde{z}_v^s, x_v))
```

```math
y_v^s = y_v^{s-1} + \gamma_v^s \odot \Delta y_v^s
```

The residual form is important. The model is not asked to regenerate the full solution at every step. It only needs to learn corrections.

### 5.4 Graph-level context update

For graph-level prediction tasks, maintain a graph state:

```math
c^s = \operatorname{READOUT}(\{[y_v^s \Vert \tilde{z}_v^s] : v \in V\})
```

```math
g^s = \operatorname{GRU}_g(g^{s-1}, c^s)
```

For node-level tasks, `g^s` can be omitted or replaced by global pooling injected back into nodes.

---

## 6. Prediction heads

### 6.1 Node classification

```math
\hat{p}_v^s = \operatorname{softmax}(f_{out}(y_v^s))
```

### 6.2 Graph classification

```math
\hat{p}_G^s = \operatorname{softmax}(f_{out}([g^s \Vert \operatorname{READOUT}(\{y_v^s\})]))
```

### 6.3 Graph regression

```math
\hat{t}_G^s = f_{out}([g^s \Vert \operatorname{READOUT}(\{y_v^s\})])
```

### 6.4 Edge prediction

For edge-level tasks:

```math
\hat{p}_{uv}^s = f_{edge}(y_u^s, y_v^s, \tilde{z}_u^s, \tilde{z}_v^s, e_{uv})
```

---

## 7. Training objective

### 7.1 Deep supervision

Decode at every macro-step and apply task loss at every step:

```math
\mathcal{L}_{task} = \sum_{s=1}^{S} w_s \ell(\hat{t}^s, t)
```

where `w_s` can increase with `s`, for example:

```math
w_s = \frac{s}{\sum_{j=1}^{S} j}
```

This keeps the final step most important while still providing useful gradients to earlier steps.

### 7.2 Improvement loss

A recursive step should not make the prediction worse. Penalize deterioration:

```math
\mathcal{L}_{imp} = \sum_{s=2}^{S} \max(0, \ell(\hat{t}^s, t) - \ell(\hat{t}^{s-1}, t) + \epsilon)
```

where `epsilon >= 0` is a margin. With `epsilon = 0`, the model is only penalized when it worsens. With `epsilon > 0`, it is encouraged to improve by at least a small margin.

### 7.3 Stability regularization

To avoid unstable corrections:

```math
\mathcal{L}_{step} = \sum_{s=1}^{S} \frac{1}{|V|}\sum_{v \in V} \|\Delta y_v^s\|_2^2
```

This discourages unnecessarily large solution jumps.

### 7.4 Full objective

```math
\mathcal{L} = \mathcal{L}_{task} + \lambda_{imp}\mathcal{L}_{imp} + \lambda_{step}\mathcal{L}_{step}
```

Recommended starting values:

```yaml
lambda_imp: 0.1
lambda_step: 1e-4
epsilon: 0.0
```

---

## 8. Pseudocode

```python
class RIGN(nn.Module):
    def __init__(self, dim, num_steps=4, micro_steps=2):
        super().__init__()
        self.S = num_steps
        self.R = micro_steps
        self.y_init = MLP(input_dim, dim)
        self.z_init = MLP(input_dim, dim)
        self.message = MessageMLP(dim)
        self.gate = GateMLP(dim)
        self.z_update = GRUCell(dim, dim)
        self.delta_y = MLP(3 * dim, dim)
        self.gamma_y = MLP(3 * dim, dim)
        self.out = OutputHead(dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        y = self.y_init(x)
        z = self.z_init(x)
        predictions = []

        for s in range(self.S):
            for r in range(self.R):
                msg = self.compute_messages(z, y, x, edge_index, edge_attr)
                z_input = self.z_input_mlp(torch.cat([z, y, x, msg], dim=-1))
                z = self.z_update(z_input, z)

            delta = self.delta_y(torch.cat([y, z, x], dim=-1))
            gamma = torch.sigmoid(self.gamma_y(torch.cat([y, z, x], dim=-1)))
            y = y + gamma * delta

            predictions.append(self.out(y))

        return predictions
```

This pseudocode is intentionally minimal. It omits batching, normalization, graph-level readout, and edge-feature details.

---

## 9. Recommended first configuration

```yaml
model: RIGN-small
hidden_dim: 128
macro_steps_S: 4
latent_micro_steps_R: 2
total_message_passing_steps: 8
aggregator: gated_sum
node_update: GRU
normalization: LayerNorm or GraphNorm
activation: GELU
readout: sum + mean pooling for graph tasks
deep_supervision: true
improvement_loss: true
input_injection: true
dropout: 0.1
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.00001
```

The first version should not use a learned halting module. Halting should be introduced only after the fixed-step architecture has been shown to outperform strong compute-matched baselines.

---

## 10. How RIGN differs from GCN, GAT, and GIN

| Model | Main mechanism | Limitation addressed by RIGN |
|---|---|---|
| GCN | Neighborhood averaging with learned linear maps | Can oversmooth and lacks explicit iterative correction |
| GAT | Attention-weighted neighbor aggregation | Selective but still feed-forward and fixed-depth |
| GIN | Expressive sum aggregation | Strong baseline, but does not separate solution from reasoning state |
| Recurrent GIN | Reuses a GIN block multiple times | Tests recurrence alone, but lacks explicit `y/z` decomposition |
| RIGN | Latent graph reasoning plus solution correction | Structured recurrence trained as improvement |

The most important distinction is the `y/z` split. If removing this split does not hurt performance, the architecture is probably overcomplicated.

---

## 11. Evaluation plan

### 11.1 Main experimental question

The central experimental question is:

> Does RIGN improve graph reasoning because of its recursive improvement design, or merely because it performs more message passing?

The evaluation must therefore control for:

- number of parameters;
- number of message-passing steps;
- training objective;
- recurrent parameter sharing;
- solution/reasoning state separation.

### 11.2 Baselines

Include the following baselines:

1. **GCN** with 2, 4, 8, and 16 layers.
2. **GAT** with 2, 4, 8, and 16 layers.
3. **GIN** with 2, 4, 8, and 16 layers.
4. **Residual GIN** with normalization and dropout.
5. **Recurrent GIN**: one GIN block reused for 8 steps.
6. **RIGN full**.
7. **RIGN without improvement loss**.
8. **RIGN without deep supervision**.
9. **RIGN without `y/z` split**.
10. **RIGN without input injection**.

The key baseline is Recurrent GIN. If RIGN does not outperform Recurrent GIN on reasoning-heavy tasks, the added architectural structure is not justified.

### 11.3 Dataset groups

#### Group A: standard node classification sanity checks

Use:

- Cora;
- Citeseer;
- PubMed;
- optionally ogbn-arxiv.

Purpose:

- verify that RIGN is not broken;
- check whether recurrence hurts simple homophilous benchmarks.

Expected outcome:

- RIGN may not dominate here.
- Comparable performance is sufficient.

#### Group B: graph-level benchmarks

Use:

- ZINC for graph regression;
- ogbg-molhiv for molecular classification;
- PROTEINS or ENZYMES for lightweight graph classification.

Purpose:

- test whether iterative latent refinement helps graph-level prediction.

Expected outcome:

- modest improvements may appear on tasks requiring longer-range structure;
- strong improvements are not guaranteed.

#### Group C: synthetic graph reasoning tasks

This is the critical part of the evaluation.

Use synthetic tasks where multi-step graph reasoning is necessary:

1. **Reachability**
   - Input: graph with marked source and target.
   - Target: whether the target is reachable from the source.

2. **Shortest-path node membership**
   - Input: graph with marked source and target.
   - Target: classify each node as on/off the shortest path.

3. **Distance threshold**
   - Input: graph with two marked nodes.
   - Target: whether their shortest-path distance is at most `k`.

4. **Maze-as-graph pathfinding**
   - Input: grid maze converted to graph.
   - Target: classify path nodes or predict next-step direction.

5. **Long-range parity or counting**
   - Input: graph with binary node labels.
   - Target: global parity, count threshold, or selected aggregate.

Use size generalization splits:

```yaml
train_graph_size: 20-50 nodes
validation_graph_size: 50-80 nodes
test_graph_size: 80-150 nodes
```

Purpose:

- test whether recurrence learns an algorithmic procedure that extrapolates beyond training sizes.

Expected outcome:

- RIGN should show its strongest advantage here.

---

## 12. Fair comparison protocol

### 12.1 Parameter-matched setting

Adjust hidden dimensions so all models have approximately equal parameter counts.

Example:

```text
RIGN-small: hidden_dim = 128
GIN-8: hidden_dim adjusted to match RIGN parameter count
GAT-8: hidden_dim and heads adjusted to match RIGN parameter count
```

### 12.2 Compute-matched setting

Match the number of message-passing applications.

Example:

```text
RIGN: S = 4, R = 2 -> 8 message-passing micro-steps
GCN/GAT/GIN: 8 layers
Recurrent GIN: 8 recurrent steps
```

### 12.3 Depth-generalization setting

Train with fewer recurrent steps and test with more:

```yaml
train_steps: 4 macro-steps
test_steps: [4, 6, 8, 10]
```

This tests whether recurrence can be used as test-time compute.

### 12.4 Hardness-stratified evaluation

For synthetic tasks, report results by graph diameter, shortest-path length, or target distance.

Example:

```text
short paths: distance <= 5
medium paths: 6 <= distance <= 10
long paths: distance > 10
```

If RIGN is genuinely useful, its relative advantage should increase on harder instances.

---

## 13. Metrics

### 13.1 Task metrics

Use task-appropriate metrics:

- accuracy for balanced classification;
- macro-F1 for imbalanced node classification;
- ROC-AUC for molecular binary classification;
- MAE/RMSE for graph regression;
- path validity and path optimality for pathfinding.

### 13.2 Efficiency metrics

Report:

- number of parameters;
- training time per epoch;
- inference time per graph;
- peak GPU memory;
- message-passing steps;
- performance per FLOP or per second.

### 13.3 Recursive improvement metrics

These are central to the project.

For each step `s`, report:

```text
step_s_accuracy
step_s_loss
```

Then compute:

```text
dead_step_rate = fraction of examples where loss_s >= loss_{s-1}
```

```text
mean_improvement = mean(loss_{s-1} - loss_s)
```

```text
overthinking_rate = fraction of examples correct at step s-1 but wrong at step s
```

```text
best_step_oracle = best performance over all steps per example
```

```text
final_minus_best = final step performance - best step oracle performance
```

These metrics reveal whether extra recurrence is useful, redundant, or harmful.

---

## 14. Ablation plan

| Ablation | Purpose | Expected interpretation |
|---|---|---|
| Full RIGN | Main model | Reference point |
| No `y/z` split | Test solution/reasoning decomposition | If no drop, the split is not important |
| No deep supervision | Test intermediate training signal | Drop would support recursive supervision |
| No improvement loss | Test whether steps are shaped as corrections | More dead steps expected |
| No input injection | Test robustness against oversmoothing and forgetting | Worse long-range and deeper-step behavior expected |
| No gated residual correction | Test correction mechanism | May reduce stability |
| Recurrent GIN | Test recurrence alone | Critical baseline |
| Untied RIGN blocks | Test parameter sharing vs depth | Better accuracy but more parameters possible |
| Train 4 steps, test 8 steps | Test test-time compute generalization | Useful only if recurrence is stable |

---

## 15. Success criteria

The project should be considered successful only if at least two of the following hold:

1. RIGN beats 8-layer GCN/GAT/GIN under compute-matched conditions.
2. RIGN beats Recurrent GIN on synthetic reasoning tasks.
3. RIGN shows mostly monotonic improvement across recursive steps.
4. RIGN generalizes better from small graphs to larger graphs.
5. RIGN obtains similar or better performance with fewer parameters than deep feed-forward GNNs.
6. RIGN has a lower overthinking rate than naive recurrent baselines.
7. The `y/z` split and improvement loss both survive ablation.

The strongest result would look like this:

```text
RIGN has roughly the same message-passing compute as 8-layer GIN,
fewer or comparable parameters,
better performance on long-range synthetic reasoning,
better size generalization,
and clear step-wise improvement curves.
```

---

## 16. Failure modes

### 16.1 RIGN is just a complicated recurrent GIN

If Recurrent GIN performs similarly, the architectural contribution is weak. The project would need either a stronger task setting or a sharper mechanism.

### 16.2 Recurrence becomes dead compute

If later steps do not improve predictions, the model is not using recurrence meaningfully. This would show up as high dead-step rate and flat step-wise accuracy curves.

### 16.3 Overthinking

If later steps degrade earlier correct predictions, the model may be over-refining. Potential fixes:

- stronger improvement loss;
- learned halting;
- step dropout;
- smaller correction gates;
- early-exit inference.

### 16.4 Oversmoothing

If node states collapse with more recurrence, use:

- input injection at every step;
- residual correction rather than full replacement;
- normalization;
- pairwise distance monitoring;
- separate `y` and `z` states.

### 16.5 Standard benchmarks do not show gains

This is expected. The architecture is designed for graph reasoning, not necessarily for simple homophilous node classification. Synthetic long-range tasks are therefore necessary.

---

## 17. Implementation milestones

### Milestone 1: Minimal RIGN implementation

Implement:

- node-level RIGN;
- gated sum aggregation;
- GRU latent update;
- residual solution correction;
- deep supervision;
- improvement loss.

Test on a toy synthetic reachability dataset.

### Milestone 2: Baselines

Implement or reuse PyTorch Geometric versions of:

- GCN;
- GAT;
- GIN;
- Recurrent GIN.

Ensure compute-matched and parameter-matched comparison.

### Milestone 3: Synthetic reasoning benchmark

Build synthetic generators for:

- reachability;
- shortest-path node membership;
- distance threshold;
- maze-as-graph pathfinding.

Include small-to-large graph generalization splits.

### Milestone 4: Full evaluation

Run:

- standard benchmarks;
- synthetic benchmarks;
- ablations;
- recursive improvement diagnostics.

### Milestone 5: Analysis

Analyze:

- step-wise improvement curves;
- dead-step rate;
- overthinking rate;
- hidden-state similarity across recurrence;
- effect of input injection;
- effect of `y/z` split.

---

## 18. Possible extensions

### 18.1 Learned halting

After fixed-step RIGN works, add a halting head:

```math
q_v^s = \sigma(f_{halt}(y_v^s))
```

For graph-level tasks:

```math
q_G^s = \sigma(f_{halt}(g^s))
```

This enables adaptive compute. However, it should not be part of the first version because it complicates evaluation.

### 18.2 Constraint-aware RIGN

For constraint-based graph tasks, add factor or constraint nodes. The latent reasoning state can propagate pressure from violated constraints to candidate repair variables.

This would make RIGN especially suitable for graph repair, knowledge graph validation, and symbolic-neural reasoning.

### 18.3 Edge-state RIGN

Maintain edge solution and reasoning states:

```math
y_{uv}^s, z_{uv}^s
```

This is useful for link prediction, pathfinding, and edge classification.

### 18.4 Multi-timescale RIGN

A more HRM-like version could maintain:

- fast node-level latent state;
- slower graph-level planning state.

This should be treated as a later extension, not the first model.

---

## 19. Minimal experiment table

| Experiment | Dataset | Baselines | Main metric | Diagnostic metric |
|---|---|---|---|---|
| Node sanity check | Cora/Citeseer/PubMed | GCN/GAT/GIN | Accuracy | Step-wise accuracy |
| Graph regression | ZINC | GIN/GAT/Recurrent GIN | MAE | Dead-step rate |
| Molecular classification | ogbg-molhiv | GIN/GAT | ROC-AUC | Overthinking rate |
| Reachability | Synthetic | GCN/GAT/GIN/Recurrent GIN | Accuracy | Size generalization |
| Shortest path | Synthetic | GCN/GAT/GIN/Recurrent GIN | Node F1 | Path-length stratification |
| Maze pathfinding | Synthetic | GCN/GAT/GIN/Recurrent GIN | Path validity | Final-minus-best |

---

## 20. Expected contribution

The project contribution would be strongest if framed as follows:

> We introduce a recurrent graph neural architecture that separates decoded solution states from latent reasoning states and trains recurrence as step-wise improvement. Unlike standard deep or recurrent GNNs, the model exposes whether additional message passing is useful, redundant, or harmful. We evaluate it against GCN, GAT, GIN, and recurrent GIN under parameter-matched and compute-matched settings, with particular focus on long-range graph reasoning and size generalization.

The project should avoid overclaiming broad superiority over all GNNs. The intended contribution is narrower and more defensible:

- better recurrent design for graph reasoning tasks;
- explicit improvement diagnostics;
- controlled comparison against standard and recurrent GNN baselines;
- evidence about when recurrence helps in graph neural networks.

---

## 21. Short version

RIGN is a recurrent GNN where each node has:

```text
y = current solution state
z = latent reasoning state
```

Each iteration does:

```text
1. update z by message passing conditioned on x and y
2. update y by a gated residual correction from z
3. decode y
4. supervise every step
5. penalize steps that worsen the prediction
```

The evaluation asks whether this beats:

```text
GCN, GAT, GIN, and Recurrent GIN
```

under fair compute and parameter budgets, especially on synthetic graph reasoning tasks where iterative computation should matter.
