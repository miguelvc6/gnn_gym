You are working in the GitHub repository:

miguelvc6/gnn_gym

Your task is NOT to implement a new GNN architecture in this pass. Your task is to fix benchmark/research-infrastructure weaknesses and update the research protocol Markdown files so future long-running agent work evaluates architectures correctly and prioritizes scientific novelty, not just incremental leaderboard gains.

## Context

GNN Gym is intended to be a long-running, agent-assisted research environment for designing novel GNN architectures. The goal is not merely to improve validation metrics through tuning or recombination of known architectures. The goal is to generate, test, and accumulate evidence about architecture ideas that may produce novel scientific insight and eventually outperform strong baseline models.

The current implementation has several important weaknesses:

1. Aggregated research summaries group by only:
   - task
   - dataset
   - model
   - metric_name

   This mixes many different hyperparameter/configuration settings into one mean/std row. That makes `research_all_runs_mean_std.csv` unsuitable for selecting the best architecture/config.

2. The current `config_hash` is computed from the whole config after seed insertion, so it is useful for identifying a run but not for grouping the same architecture/hyperparameter configuration across multiple seeds.

3. Graph-prediction models do not receive `edge_attr`. The graph trainer calls the model with only:
   - x
   - edge_index
   - batch

   This makes molecular graph tasks weaker than they should be, because bond/edge features are unavailable to edge-aware architectures.

4. `neighbor_node` is not a proper scalable neighbor-loader trainer. It uses `k_hop_subgraph` manually and evaluates on the full graph. This is acceptable for smoke experiments but not good enough for serious `ogbn-products`-style runs.

5. Model registration is brittle. `ensure_registrations()` manually imports every model module. This can cause new model files to be silently unusable if the import is forgotten.

6. The Markdown research protocol does not yet strongly enforce the evaluation protocol needed for long-running architecture research:
   - selection by validation metric only
   - test metric recorded only for reporting
   - seed-0 fast search
   - seeds `[0, 1, 2]` confirmation
   - config-level aggregation
   - no architecture claims from mixed-config model-level averages
   - novelty-first idea generation
   - explicit scientific hypothesis for each architecture idea
   - durable insight accumulation even from failed experiments

## High-level goal

Fix the benchmark plumbing first, then update the Markdown files so future agents follow a stricter long-running research protocol.

---

# Part A — Fix config-level experiment tracking and aggregation

## Required behavior

Keep the existing run-level `config_hash` behavior if useful, but add a second hash that is seed-independent and suitable for grouping the same architecture/config across seeds.

Use a name such as:

```text
architecture_config_hash
````

This hash should be computed from the resolved config after model/dataset/default/override merging, but with seed-dependent and run-dependent fields removed.

At minimum, remove:

```text
training.seed
config_hash
architecture_config_hash
run_id
device-specific runtime metadata
```

Do not remove actual architecture or training hyperparameters such as:

```text
model.hidden_channels
model.num_layers
model.dropout
training.lr
training.weight_decay
training.max_epochs
training.patience
dataset.name
trainer.name
```

The goal is:

```text
same model/dataset/hyperparameters, different seeds -> same architecture_config_hash
different model/dataset/hyperparameters -> different architecture_config_hash
```

## Implementation guidance

Inspect:

```text
src/gnn_gym/utils/hashing.py
src/gnn_gym/cli.py
src/gnn_gym/training/trainer.py
src/gnn_gym/evaluation/aggregate.py
tests/
```

Add a helper function, probably in `src/gnn_gym/utils/hashing.py`, such as:

```python
def seedless_config_hash(config: Mapping[str, object]) -> str:
    ...
```

or more generally:

```python
def architecture_config_hash(config: Mapping[str, object]) -> str:
    ...
```

Use a deep copy. Do not mutate the original config.

Add the new hash to:

```text
resolved_config.yaml
metadata.json
aggregate output rows
exported tables
```

Update aggregation so summaries can group by exact architecture config. The default research summary should group by:

```text
task
dataset
model
metric_name
architecture_config_hash
```

It is acceptable to keep a separate model-level summary if useful, but make sure it is clearly named so it is not confused with config-level evidence.

Suggested output files or function behavior:

```text
research_all_runs.csv                  # raw run rows
research_all_runs_by_config_mean_std.csv
research_all_runs_by_model_mean_std.csv  # optional, clearly marked as mixed-config
```

If changing existing file names is too disruptive, minimally include `architecture_config_hash` in the summary and update the Markdown protocol to warn against selecting from mixed-config summaries.

## Tests

Add or update tests to verify:

1. Two fake runs with identical configs except seed have the same `architecture_config_hash`.
2. Two fake runs with different model hyperparameters have different `architecture_config_hash`.
3. Aggregation grouped by `architecture_config_hash` produces separate rows for different configs of the same model.
4. The raw aggregate table contains both `config_hash` and `architecture_config_hash`.

---

# Part B — Pass edge attributes through the model stack

## Required behavior

Graph-prediction and link/node trainers should pass `edge_attr` to the model when available.

Existing models that ignore edge attributes must continue working.

Future edge-aware models should be able to implement:

```python
forward(x, edge_index=None, batch=None, edge_attr=None)
```

or another compatible signature, without trainer changes.

## Implementation guidance

Inspect:

```text
src/gnn_gym/models/base.py
src/gnn_gym/models/heads.py
src/gnn_gym/training/node_trainer.py
src/gnn_gym/training/graph_trainer.py
src/gnn_gym/training/link_trainer.py
src/gnn_gym/data/catalog.py
tests/test_model_shapes.py
tests/test_toy_training.py
```

Do not manually update every existing model file unless necessary. A compatibility layer is preferable.

Suggested approach:

1. Update `TaskModel.forward(...)` to accept:

```python
edge_attr: torch.Tensor | None = None
```

while preserving the existing positional calling convention:

```python
model(x, edge_index, batch)
```

So prefer this signature shape:

```python
def forward(
    self,
    x: torch.Tensor,
    edge_index: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

2. Update `TaskModel.encode(...)` similarly.

3. Add a helper that calls the encoder with `edge_attr` only if the encoder supports it. Avoid catching arbitrary internal `TypeError` if possible. Use `inspect.signature` or a small capability check.

4. Update trainers to pass edge attributes by keyword:

```python
edge_attr = getattr(data_or_batch, "edge_attr", None)
self.model(x, edge_index, batch=batch, edge_attr=edge_attr)
```

For node/link trainers, pass `edge_attr` if it exists on the single `Data` object.

5. Keep existing models working without requiring changes.

6. Add a toy graph or synthetic test where `edge_attr` exists and verify the model/trainer path does not crash.

Do not implement a full edge-aware molecular architecture in this task unless needed for a minimal test. The goal is plumbing correctness.

---

# Part C — Improve the neighbor-node trainer

## Required behavior

Make `neighbor_node` a real mini-batch neighbor-sampling trainer when PyG’s `NeighborLoader` is available.

The trainer should support:

```yaml
trainer:
  name: neighbor_node
  batch_size: 1024
  eval_batch_size: 4096
  num_neighbors: [15, 10, 5]
```

Training should use neighborhood batches rooted at train nodes.

Evaluation should support batched evaluation over train/val/test masks instead of always forcing full-graph evaluation. This matters for `ogbn-products`.

## Implementation guidance

Inspect:

```text
src/gnn_gym/training/node_trainer.py
configs/trainers/neighbor_node.yaml
configs/datasets/ogbn_products.yaml
tests/
```

Suggested implementation:

1. Use:

```python
from torch_geometric.loader import NeighborLoader
```

2. Build loaders with:

```python
NeighborLoader(
    data,
    input_nodes=data.train_mask,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=True,
)
```

3. For each sampled batch:

```python
logits = model(batch.x, batch.edge_index, edge_attr=getattr(batch, "edge_attr", None))
root_logits = logits[: batch.batch_size]
root_y = batch.y[: batch.batch_size]
```

4. For evaluation, build non-shuffled loaders for train/val/test masks and compute metrics on the root nodes of each batch.

5. Keep a fallback path for environments where `NeighborLoader` cannot run because optional PyG sampling backends are missing. The fallback can be the existing `k_hop_subgraph` path, but it must be clearly documented and should not be treated as the scalable path.

6. Add or update tests so toy neighbor-node training still passes.

---

# Part D — Make model registration less brittle

## Required behavior

New model modules under `src/gnn_gym/models/` should be automatically imported during registration discovery, so a future agent does not need to manually edit a giant import list in `registry.py`.

## Implementation guidance

Inspect:

```text
src/gnn_gym/registry.py
src/gnn_gym/models/
tests/test_registry.py
```

Suggested approach:

1. Replace the long list of manual model imports with package discovery using:

```python
pkgutil.iter_modules
importlib.import_module
```

2. Import all model modules under `gnn_gym.models`, excluding:

```text
__init__
base
heads
```

3. Keep explicit imports for:

```text
gnn_gym.data.catalog
gnn_gym.evaluation.evaluators
gnn_gym.training.graph_trainer
gnn_gym.training.link_trainer
gnn_gym.training.node_trainer
```

or similarly auto-discover those packages only if safe.

4. Make `ensure_registrations()` idempotent. Calling it multiple times should not re-register models in a way that raises duplicate errors.

5. Add/update tests to ensure important existing models are registered after `ensure_registrations()`, including at least:

```text
mlp
gcn
gat
gin
appnp_net
gpr_gnn
nb_belief_gnn
```

---

# Part E — Update Markdown research protocol

Update the Markdown files so future agents follow a stricter evaluation and novelty protocol.

At minimum inspect and update:

```text
README.md
AGENTS.md
research/PROGRAM.md
research/ARCHITECTURE_IDEAS.md
research/AGENT_SCRATCHPAD.md
research/INSIGHTS.md
```

Create a new Markdown file only if it makes the protocol clearer. If you create one, prefer:

```text
research/LONG_RUNNING_RESEARCH.md
```

## Required protocol content

Add a clearly named section such as:

```text
Evaluation Protocol For Architecture Claims
```

Include these rules:

1. Use validation metric for selection.
2. Record test metric, but do not select architectures or hyperparameters using test performance.
3. A seed-0 result is only a screening signal.
4. Any promising seed-0 architecture/config must be confirmed across seeds `[0, 1, 2]`.
5. Architecture claims must be based on config-level aggregation, not model-level mixed-config averages.
6. The config-level grouping key is `architecture_config_hash`.
7. Do not claim a model beats a baseline unless the exact confirmed config beats the baseline mean validation metric under the same budget.
8. Use task-appropriate baselines:

   * for Cora/PubMed node classification: current confirmed `gpr_gnn`
   * for older node baselines: GAT, APPNP, JK-GCN, GATv2 as relevant comparison points
   * for MolHIV graph prediction: current GIN/GCN/GAT/MLP baselines, but note that edge-aware baselines should be added once edge-attribute plumbing is fixed
9. Test metrics are held-out reporting only.
10. Toy datasets are crash checks only, never architecture evidence.

## Required novelty content

Add a clearly named section such as:

```text
Novelty And Scientific Insight Standard
```

Make the standard stricter than “improve validation accuracy.”

Each architecture idea should state:

```text
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
```

Update the `research/ARCHITECTURE_IDEAS.md` template accordingly.

Make clear that:

* Simple hyperparameter tuning is useful engineering, not a novel architecture.
* Reimplementing known methods like APPNP, GPR-GNN, GATv2, GCNII, GraphSAGE, GINE, GPS, Graphormer, etc. should be labeled as baselines unless there is a genuinely new mechanism.
* Hybridizing two known methods is not automatically novel. It needs a specific mechanism and a falsifiable claim.
* A failed architecture can still be scientifically useful if it tests a clear hypothesis and records what was learned.
* The long-running objective is to accumulate a research memory over months, periodically adding ideas, evidence, negative results, and refined hypotheses.

## Required long-running workflow content

Add a section describing the intended long-running loop:

```text
1. Read the protocol and insights.
2. Pick or add one architecture idea with a scientific hypothesis.
3. Check closest known baselines before calling it novel.
4. Implement the smallest bounded version.
5. Run toy crash checks.
6. Run seed-0 fast screen.
7. Confirm only promising configs over seeds [0,1,2].
8. Aggregate by architecture_config_hash.
9. Promote only durable conclusions to INSIGHTS.md.
10. Keep negative results if they clarify the hypothesis space.
11. Periodically add new ideas, but avoid repeating discarded mechanisms without a new reason.
```

## Required warnings

Add explicit warnings that future agents must not:

* select from `test_metric`
* use toy metrics as evidence
* claim novelty for known architectures
* claim improvement from a mixed-config mean/std table
* edit trainers/evaluators to favor a candidate architecture unless the idea explicitly requires a new training objective
* compare across incompatible metrics as if they were one global score
* hide negative results

---

# Part F — Tests and quality checks

Run:

```bash
uv run ruff check .
uv run pytest
```

Also run minimal smoke checks if feasible:

```bash
uv run gnngym train --model gcn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym train --model gin --dataset toy-graph --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym train --model gcn --dataset toy-link --seed 0 --override training.max_epochs=2 --override training.patience=2
```

If a smoke check cannot be run because of environment limitations, document exactly why in the final response.

---

# Part G — Final response format

At the end, report:

1. Files changed.
2. What was fixed.
3. What tests/checks passed.
4. Any checks that could not be run.
5. Any remaining limitations.
6. Any follow-up recommendations.

Do not claim that a new architecture result was achieved. This task is infrastructure and protocol hardening only.
