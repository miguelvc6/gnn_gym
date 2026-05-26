# GPR-GNN Model Description

## Summary

`gpr_gnn` is a node-classification encoder that decouples feature transformation from graph
propagation. It first maps raw node features into a hidden representation with a small MLP, then
computes multiple GCN-normalized propagation states and returns a learned weighted sum of those hop
states. The task wrapper then applies the standard node-classification linear head.

The model was introduced in this repository as a bounded experiment after APPNP became the strongest
node baseline. The experiment kept APPNP's stable personalized-PageRank propagation prior, but made
the final hop mixture learnable instead of fixing the propagation weights. On Cora and PubMed this
became the strongest confirmed node-classification model from the current research loop.

## Implementation Location

- Model: `src/gnn_gym/models/gpr_gnn.py`
- Default config: `configs/models/gpr_gnn.yaml`
- Shape coverage: `tests/test_model_shapes.py`
- Main result notes: `research/INSIGHTS.md`
- Aggregated runs: `results/tables/research_all_runs.csv`

## Architecture

The registered model name is `gpr_gnn`. The encoder accepts node features `x` and `edge_index`; it
requires graph structure and raises an error if `edge_index` is missing.

The architecture has three parts:

1. Feature encoder:
   - A stack of `num_layers` linear layers.
   - Each layer is followed by the configured normalization layer, activation, and dropout.
   - The default normalization is batch normalization.
   - The default activation is ReLU.

2. GCN-normalized propagation bank:
   - The input graph is normalized with PyG's `gcn_norm`.
   - Self-loops are added.
   - The normalized operator is the symmetric GCN operator with edge weights equivalent to
     `D_hat^{-1/2} A_hat D_hat^{-1/2}`.
   - Starting from the encoded feature state, the model applies this normalized propagation
     operator `K = propagation_steps` times and stores every intermediate hop state.

3. Learned hop mixture:
   - The model has one scalar trainable coefficient per stored state.
   - Since the stored states are hops `0..K`, the coefficient vector has length `K + 1`.
   - The coefficients are initialized to APPNP-like personalized PageRank weights.
   - In the implementation, the parameter is named `hop_logits`, but it is used directly as a
     coefficient vector. There is no softmax or simplex constraint.
   - This means training can adjust the magnitude and sign of each hop contribution.

The full task model wraps this encoder with the standard `NodeClassificationHead`, a single linear
map from hidden channels to classes.

## Inference Equations

Let the input graph be `G = (V, E)`, with node feature matrix:

```text
X in R^{N x F}
```

where `N` is the number of nodes and `F` is the input feature dimension.

Let `A` be the adjacency matrix. The implementation adds self-loops:

```text
A_hat = A + I
```

and uses the symmetric GCN normalization:

```text
S = D_hat^{-1/2} A_hat D_hat^{-1/2}
```

where `D_hat` is the degree matrix of `A_hat`.

First, the feature encoder produces a hidden node matrix. For `L = num_layers`, hidden width `H`,
activation `sigma`, normalization `Norm_l`, and dropout operator `Dropout_p`:

```text
H_0 = X
H_l = Dropout_p(sigma(Norm_l(H_{l-1} W_l + b_l)))   for l = 1..L
Z_0 = H_L
```

Then the model builds a propagation bank:

```text
Z_k = S Z_{k-1}   for k = 1..K
```

where `K = propagation_steps`.

The encoder output is a learned weighted hop sum:

```text
Z = sum_{k=0}^{K} gamma_k Z_k
```

where `gamma` is the trainable `hop_logits` parameter used directly as propagation coefficients.

For node classification, the repository wrapper applies the standard linear classifier:

```text
Y_logits = Z W_cls + b_cls
```

Predictions are then:

```text
y_hat_i = argmax_c Y_logits[i, c]
```

## Hop Coefficient Initialization

The learned hop coefficients are initialized with an APPNP-like personalized PageRank distribution.
For teleport probability `alpha` and propagation depth `K`, the initialization is:

```text
gamma_k = alpha (1 - alpha)^k   for k = 0..K-1
gamma_K = (1 - alpha)^K
```

This is the finite-depth APPNP/PPR prior: early states retain local feature information, later states
carry progressively more graph-smoothed information, and the final term stores the remaining
probability mass.

The important difference from APPNP is that these coefficients are trainable after initialization.
APPNP fixes the recursive blend between the original encoded features and propagated features. This
model instead explicitly stores all hop states and learns how much to use each one.

## Inspiration

The repository research loop found that APPNP was stronger than the original GCN, GAT, and GIN
baselines on Cora and PubMed. APPNP's advantage suggested that decoupling representation learning
from propagation was more effective than repeatedly interleaving message passing and nonlinear
feature transforms.

`gpr_gnn` follows that observation and makes one conservative change:

```text
Keep the APPNP/PPR propagation prior, but learn the hop mixture.
```

The idea is related to Generalized PageRank GNNs: different datasets and nodes may prefer different
ranges of graph propagation. A fixed PageRank schedule assumes one decay pattern. A learned hop
mixture can emphasize shallow, medium, or deeper propagation states while still starting from a
stable PPR initialization.

Within this repo, the motivation was pragmatic rather than speculative:

- GCN underperformed, likely because repeated nonlinear neighborhood mixing was not the best fit for
  these citation graphs under the fixed harness.
- GAT improved over GCN, but attention still trailed propagation-decoupled models.
- APPNP was strong and stable on both primary datasets.
- GPR-style learned propagation provided a small, bounded extension of APPNP without changing the
  trainer, evaluator, datasets, or aggregation harness.

## Validation-Selected Configurations

The default model YAML is:

```yaml
model:
  name: gpr_gnn
  hidden_channels: 64
  num_layers: 2
  propagation_steps: 10
  alpha: 0.1
  dropout: 0.5
  activation: relu
  norm: batchnorm
```

The best confirmed configurations were selected by validation accuracy only.

| Dataset | hidden_channels | num_layers | dropout | propagation_steps | alpha | lr | weight_decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 128 | 1 | 0.1 | 16 | 0.1 | 0.01 | 0.0005 |
| PubMed | 64 | 1 | 0.2 | 10 | 0.1 | 0.01 | 0.0005 |

The PubMed refinement with `dropout=0.3` and `alpha=0.05` was rejected because its confirmed
validation mean was below the original PubMed GPR configuration.

## Evaluation Comparison Against Baselines

All values below are three-seed means over seeds `0, 1, 2`. Selection decisions were based on
validation accuracy only. Test accuracy is held-out reporting.

### Cora

| Model | Val accuracy | Test accuracy | Params | Val delta vs GPR |
| --- | ---: | ---: | ---: | ---: |
| GPR-GNN | 0.7827 +/- 0.0042 | 0.7757 +/- 0.0006 | 184,728 | - |
| GAT baseline | 0.7313 +/- 0.0110 | 0.7333 +/- 0.0031 | 201,991 | +0.0513 |
| GIN baseline | 0.6613 +/- 0.0115 | 0.6710 +/- 0.0192 | 113,418 | +0.1213 |
| GCN baseline | 0.6380 +/- 0.0330 | 0.6507 +/- 0.0211 | 96,647 | +0.1447 |

### PubMed

| Model | Val accuracy | Test accuracy | Params | Val delta vs GPR |
| --- | ---: | ---: | ---: | ---: |
| GPR-GNN | 0.8007 +/- 0.0050 | 0.7807 +/- 0.0058 | 32,398 | - |
| GAT baseline | 0.7607 +/- 0.0167 | 0.7547 +/- 0.0150 | 82,051 | +0.0400 |
| GIN baseline | 0.7300 +/- 0.0212 | 0.6973 +/- 0.0242 | 53,446 | +0.0707 |
| GCN baseline | 0.6507 +/- 0.1307 | 0.6617 +/- 0.1240 | 36,675 | +0.1500 |

## Interpretation

GPR-GNN is the current validation-selected node-classification target in the repository. It beats the
original GCN, GAT, and GIN baselines on both Cora and PubMed by a clear validation margin.

The main architectural lesson is that these citation benchmarks benefit more from decoupled
propagation than from adding more message-passing complexity. The model does not introduce a new
trainer, auxiliary objective, dataset-specific path, or evaluator change. Its gain comes from a
small increase in propagation flexibility:

```text
fixed APPNP propagation schedule -> trainable hop-weighted propagation bank
```

This also explains why later, more complex belief-style node experiments did not displace it. Many
of those models added edge states, region states, or recurrent update machinery, but their fast Cora
and PubMed validation results stayed below GPR-GNN. For future node experiments, GPR-GNN should be
treated as the baseline to beat under matched seeds and training budgets.

