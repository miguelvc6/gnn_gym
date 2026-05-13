# GNN Gym: Repository Specification

## 1. Project overview

**GNN Gym** is a reproducible benchmarking framework for evaluating graph neural network architectures across a fixed core suite of graph datasets.

The repository should make it easy to:

1. Add a new GNN architecture.
2. Run it across a standard list of datasets.
3. Store trained checkpoints, predictions, logs, and metrics.
4. Aggregate results across seeds.
5. Compare models fairly across task families.

The initial models are standard baselines:

- MLP
- GCN
- GAT
- GIN

The long-term goal is to support new custom architectures without rewriting the training and evaluation pipeline.

A new model should usually require only:

```text
src/gnn_gym/models/my_model.py
configs/models/my_model.yaml
tests/test_model_shapes.py
```

The training engine, dataset loaders, evaluators, checkpointing logic, and result aggregation should remain reusable.

---

## 2. Core design principle

The repository should separate four concerns:

```text
Dataset adapter  +  Model encoder  +  Task head/trainer  +  Evaluator
```

Do **not** create one script per model or one script per dataset.

Instead, use a registry-based architecture where a single CLI can run:

```bash
uv run gnngym train --model gcn --dataset cora --seed 0
uv run gnngym train --model gat --dataset ogbn-arxiv --seed 1
uv run gnngym run-experiment --config configs/experiments/core_gcn.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/core_mean_std.csv
```

This should allow experiments of the form:

```text
model × dataset × seed × config → checkpoint + metrics + logs + reproducible config
```

---

## 3. Technology stack

Use:

- **Python 3.11**
- **uv** for environment and dependency management
- **PyTorch** as the neural network backend
- **PyTorch Geometric** as the graph learning library
- **OGB** for Open Graph Benchmark datasets and official evaluators
- **Hydra/OmegaConf** or simple YAML loading for configuration
- **Typer** for the command-line interface
- **Pandas/Polars/PyArrow** for result aggregation
- **Ruff**, **pytest**, and optionally **mypy** for quality checks
- **W&B** as an optional tracking backend, but local logging must work without W&B

The repository must work with local-only experiment tracking by default.

---

## 4. Initial setup commands

Expected setup:

```bash
uv init --package gnn-gym
uv python pin 3.11
```

Then install dependencies.

The exact PyTorch installation command may depend on CUDA version. After installing the correct PyTorch build, add the remaining dependencies:

```bash
uv add torch-geometric
uv add ogb
uv add numpy pandas polars pyarrow scikit-learn scipy networkx
uv add pyyaml hydra-core omegaconf pydantic
uv add tqdm rich typer
uv add matplotlib seaborn
uv add wandb tensorboard
uv add pytest ruff mypy pre-commit --dev
```

If dependency groups are used, keep runtime and development dependencies separate.

---

## 5. Repository architecture

Implement the repository with the following structure:

```text
gnn-gym/
├── README.md
├── AGENTS.md
├── PROJECT_SPEC.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── datasets/
│   │   ├── cora.yaml
│   │   ├── pubmed.yaml
│   │   ├── ogbn_arxiv.yaml
│   │   ├── ogbn_products.yaml
│   │   ├── roman_empire.yaml
│   │   ├── amazon_ratings.yaml
│   │   ├── ogbg_molhiv.yaml
│   │   ├── ogbg_molpcba.yaml
│   │   ├── peptides_func.yaml
│   │   ├── peptides_struct.yaml
│   │   └── ogbl_collab.yaml
│   ├── models/
│   │   ├── mlp.yaml
│   │   ├── gcn.yaml
│   │   ├── gat.yaml
│   │   └── gin.yaml
│   ├── trainers/
│   │   ├── full_batch_node.yaml
│   │   ├── neighbor_node.yaml
│   │   ├── graph_prediction.yaml
│   │   └── link_prediction.yaml
│   └── experiments/
│       ├── smoke_test.yaml
│       ├── core_gcn.yaml
│       ├── core_gat.yaml
│       ├── core_gin.yaml
│       └── core_all_baselines.yaml
├── src/
│   └── gnn_gym/
│       ├── __init__.py
│       ├── cli.py
│       ├── registry.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── catalog.py
│       │   ├── adapters.py
│       │   ├── loaders.py
│       │   ├── transforms.py
│       │   └── splits.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── mlp.py
│       │   ├── gcn.py
│       │   ├── gat.py
│       │   ├── gin.py
│       │   └── heads.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── node_trainer.py
│       │   ├── graph_trainer.py
│       │   ├── link_trainer.py
│       │   ├── losses.py
│       │   ├── optimizers.py
│       │   ├── schedulers.py
│       │   └── early_stopping.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── evaluators.py
│       │   ├── ogb_eval.py
│       │   └── aggregate.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── run.py
│       │   ├── sweep.py
│       │   └── resume.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── device.py
│           ├── seed.py
│           ├── logging.py
│           ├── checkpointing.py
│           ├── paths.py
│           └── hashing.py
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_core_benchmark.py
│   ├── aggregate_results.py
│   └── export_tables.py
├── tests/
│   ├── test_registry.py
│   ├── test_config_loading.py
│   ├── test_model_shapes.py
│   ├── test_toy_training.py
│   └── test_result_aggregation.py
├── notebooks/
│   ├── 01_result_analysis.ipynb
│   └── 02_model_diagnostics.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── artifacts/
│   ├── checkpoints/
│   ├── predictions/
│   ├── logs/
│   └── summaries/
└── results/
    ├── runs/
    ├── tables/
    └── figures/
```

---

## 6. Core datasets

The initial benchmark suite should include the following datasets.

### 6.1 Node classification

| Dataset | Trainer | Metric | Notes |
|---|---|---|---|
| Cora | `full_batch_node` | Accuracy | Small debugging/sanity-check dataset |
| PubMed | `full_batch_node` | Accuracy | Larger Planetoid citation graph |
| ogbn-arxiv | `full_batch_node` or `neighbor_node` | OGB accuracy | Medium-scale citation graph with time split |
| ogbn-products | `neighbor_node` | OGB accuracy | Large product co-purchase graph |
| Roman-empire | `full_batch_node` | Accuracy | Heterophily benchmark |
| Amazon-ratings | `full_batch_node` | Accuracy | Heterophily benchmark |

### 6.2 Graph prediction

| Dataset | Trainer | Metric | Notes |
|---|---|---|---|
| ogbg-molhiv | `graph_prediction` | ROC-AUC | Molecular graph classification |
| ogbg-molpcba | `graph_prediction` | Average precision | Molecular multi-label classification |
| Peptides-func | `graph_prediction` | Average precision | Long-range graph classification |
| Peptides-struct | `graph_prediction` | MAE | Long-range graph regression |

### 6.3 Link prediction

| Dataset | Trainer | Metric | Notes |
|---|---|---|---|
| ogbl-collab | `link_prediction` | Hits@50 | Collaboration link prediction |

---

## 7. Phased implementation strategy

Do not implement all datasets at once.

### Phase 1: Infrastructure and smoke tests

Implement:

- Project skeleton
- Registry
- Config loading
- CLI
- Toy synthetic dataset
- MLP
- GCN
- Full-batch node classification trainer
- Local logging
- Checkpoint saving
- Result aggregation over fake runs

Acceptance criteria:

```bash
uv run pytest
uv run gnngym train --model gcn --dataset toy-node --seed 0
uv run gnngym aggregate --runs results/runs
```

### Phase 2: Small real datasets

Add:

- Cora
- PubMed
- GAT
- GIN

Acceptance criteria:

```bash
uv run gnngym train --model gcn --dataset cora --seed 0
uv run gnngym train --model gat --dataset pubmed --seed 0
uv run gnngym train --model gin --dataset cora --seed 0
```

### Phase 3: OGB node datasets

Add:

- ogbn-arxiv
- ogbn-products

Acceptance criteria:

```bash
uv run gnngym train --model gcn --dataset ogbn-arxiv --seed 0
uv run gnngym train --model gat --dataset ogbn-arxiv --seed 0
uv run gnngym train --model gcn --dataset ogbn-products --seed 0
```

For `ogbn-products`, use neighbor sampling by default.

### Phase 4: Graph prediction

Add:

- ogbg-molhiv
- ogbg-molpcba
- Peptides-func
- Peptides-struct

Acceptance criteria:

```bash
uv run gnngym train --model gin --dataset ogbg-molhiv --seed 0
uv run gnngym train --model gcn --dataset peptides-func --seed 0
uv run gnngym train --model gat --dataset peptides-struct --seed 0
```

### Phase 5: Link prediction

Add:

- ogbl-collab

Acceptance criteria:

```bash
uv run gnngym train --model gcn --dataset ogbl-collab --seed 0
uv run gnngym train --model gat --dataset ogbl-collab --seed 0
uv run gnngym train --model gin --dataset ogbl-collab --seed 0
```

### Phase 6: Full benchmark

Run:

```bash
uv run gnngym run-experiment --config configs/experiments/core_all_baselines.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/core_all_baselines.csv
uv run gnngym export-tables --input results/tables/core_all_baselines.csv
```

---

## 8. Registry system

Implement a simple registry for models, datasets, trainers, and evaluators.

```python
# src/gnn_gym/registry.py

from collections.abc import Callable

MODEL_REGISTRY: dict[str, Callable] = {}
DATASET_REGISTRY: dict[str, Callable] = {}
TRAINER_REGISTRY: dict[str, Callable] = {}
EVALUATOR_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model already registered: {name}")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_dataset(name: str):
    def decorator(fn):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset already registered: {name}")
        DATASET_REGISTRY[name] = fn
        return fn

    return decorator


def register_trainer(name: str):
    def decorator(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError(f"Trainer already registered: {name}")
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def register_evaluator(name: str):
    def decorator(cls):
        if name in EVALUATOR_REGISTRY:
            raise ValueError(f"Evaluator already registered: {name}")
        EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator
```

All models, datasets, trainers, and evaluators should be registered explicitly.

---

## 9. Model design

Use encoder/head separation where possible.

### 9.1 Encoder

A GNN encoder should produce node embeddings:

```python
z = encoder(x, edge_index, edge_attr=None, batch=None)
```

### 9.2 Task heads

Use task-specific heads:

```text
NodeClassificationHead
GraphClassificationHead
GraphRegressionHead
DotProductLinkPredictionHead
MLPLinkPredictionHead
```

For graph prediction, use global pooling:

- `global_add_pool`
- `global_mean_pool`
- `global_max_pool`

Make pooling configurable.

### 9.3 Required model interface

Each model should be instantiable from config:

```python
model = build_model(
    name="gcn",
    in_channels=dataset.num_features,
    out_channels=dataset.num_outputs,
    task=dataset.task,
    config=model_config,
)
```

Each model must support a shape test.

---

## 10. Baseline models

Implement the following initial baselines.

### 10.1 MLP

Purpose:

- Non-graph baseline
- Tests whether graph structure is actually useful

Behavior:

- Ignores `edge_index`
- For graph-level tasks, apply node-level MLP followed by graph pooling
- For node-level tasks, apply MLP directly to node features

### 10.2 GCN

Use `torch_geometric.nn.GCNConv`.

Configurable parameters:

```yaml
hidden_channels: 256
num_layers: 3
dropout: 0.5
activation: relu
norm: batchnorm
```

### 10.3 GAT

Use `torch_geometric.nn.GATConv` or `GATv2Conv`.

Configurable parameters:

```yaml
hidden_channels: 128
num_layers: 3
heads: 4
dropout: 0.5
attention_dropout: 0.2
activation: elu
norm: batchnorm
```

Be careful with output dimensions when using multi-head attention.

### 10.4 GIN

Use `torch_geometric.nn.GINConv`.

Configurable parameters:

```yaml
hidden_channels: 256
num_layers: 5
dropout: 0.5
activation: relu
norm: batchnorm
eps_trainable: true
pooling: mean
```

GIN is especially important for graph-level tasks.

---

## 11. Trainers

### 11.1 Base trainer

Create a base trainer with common logic:

```text
setup
train_epoch
validate
test
save_checkpoint
load_checkpoint
log_metrics
run
```

The base trainer should not assume the task type.

### 11.2 Full-batch node trainer

Used for:

- Cora
- PubMed
- ogbn-arxiv if memory allows
- Roman-empire
- Amazon-ratings

Must support:

- train/validation/test masks
- cross-entropy loss
- accuracy metric
- OGB evaluator when applicable

### 11.3 Neighbor-sampling node trainer

Used for:

- ogbn-products
- potentially ogbn-arxiv if full-batch is too slow or memory-heavy

Must support:

- PyG neighbor loaders
- mini-batch training
- full-graph or batched evaluation
- configurable fanout

Example config:

```yaml
trainer:
  name: neighbor_node
  batch_size: 1024
  num_neighbors: [15, 10, 5]
  eval_batch_size: 4096
```

### 11.4 Graph prediction trainer

Used for:

- ogbg-molhiv
- ogbg-molpcba
- Peptides-func
- Peptides-struct

Must support:

- graph mini-batching
- binary classification
- multi-label classification
- regression
- OGB evaluator when applicable
- missing labels for MolPCBA

### 11.5 Link prediction trainer

Used for:

- ogbl-collab

Must support:

- edge split loading
- negative sampling
- link prediction decoders
- Hits@K evaluation
- OGB evaluator when applicable

---

## 12. Evaluation

Use dataset-native metrics.

| Dataset | Metric |
|---|---|
| Cora | Accuracy |
| PubMed | Accuracy |
| ogbn-arxiv | OGB accuracy |
| ogbn-products | OGB accuracy |
| Roman-empire | Accuracy |
| Amazon-ratings | Accuracy |
| ogbg-molhiv | ROC-AUC |
| ogbg-molpcba | Average precision |
| Peptides-func | Average precision |
| Peptides-struct | MAE |
| ogbl-collab | Hits@50 |

Store all metrics in machine-readable files.

Do not collapse all metrics into one global average. Different tasks have incompatible metrics.

Aggregate results by:

```text
task family
dataset
model
seed
```

Recommended summary:

```text
mean ± std over seeds
best validation score
test score at best validation epoch
training time
number of parameters
peak GPU memory if available
```

Optional:

```text
average rank per task family
```

---

## 13. Result and artifact storage

Each run should create a self-contained folder.

```text
results/runs/
└── 2026-05-13_18-30-22__gcn__ogbn-arxiv__seed-0__a8f42c/
    ├── config.yaml
    ├── resolved_config.yaml
    ├── metrics.jsonl
    ├── final_metrics.json
    ├── checkpoint_best.pt
    ├── checkpoint_last.pt
    ├── predictions.pt
    ├── stdout.log
    ├── stderr.log
    └── metadata.json
```

`metadata.json` should contain:

```json
{
  "run_id": "2026-05-13_18-30-22__gcn__ogbn-arxiv__seed-0__a8f42c",
  "model": "gcn",
  "dataset": "ogbn-arxiv",
  "task": "node_classification",
  "seed": 0,
  "git_commit": "...",
  "config_hash": "...",
  "device": "cuda",
  "best_epoch": 142,
  "status": "completed"
}
```

Aggregate files should be stored in:

```text
results/tables/
├── core_all_runs.parquet
├── core_mean_std.csv
├── core_latex_table.tex
└── core_markdown_table.md
```

---

## 14. Configuration examples

### 14.1 Model config: GCN

```yaml
# configs/models/gcn.yaml

model:
  name: gcn
  hidden_channels: 256
  num_layers: 3
  dropout: 0.5
  activation: relu
  norm: batchnorm
```

### 14.2 Model config: GAT

```yaml
# configs/models/gat.yaml

model:
  name: gat
  hidden_channels: 128
  num_layers: 3
  heads: 4
  dropout: 0.5
  attention_dropout: 0.2
  activation: elu
  norm: batchnorm
```

### 14.3 Model config: GIN

```yaml
# configs/models/gin.yaml

model:
  name: gin
  hidden_channels: 256
  num_layers: 5
  dropout: 0.5
  activation: relu
  norm: batchnorm
  eps_trainable: true
  pooling: mean
```

### 14.4 Dataset config: ogbn-arxiv

```yaml
# configs/datasets/ogbn_arxiv.yaml

dataset:
  name: ogbn-arxiv
  task: node_classification
  source: ogb
  root: data/ogb
  metric: accuracy
  evaluator: ogb
  trainer: full_batch_node
```

### 14.5 Dataset config: ogbg-molhiv

```yaml
# configs/datasets/ogbg_molhiv.yaml

dataset:
  name: ogbg-molhiv
  task: graph_binary_classification
  source: ogb
  root: data/ogb
  metric: rocauc
  evaluator: ogb
  trainer: graph_prediction
```

### 14.6 Dataset config: ogbl-collab

```yaml
# configs/datasets/ogbl_collab.yaml

dataset:
  name: ogbl-collab
  task: link_prediction
  source: ogb
  root: data/ogb
  metric: hits@50
  evaluator: ogb
  trainer: link_prediction
```

### 14.7 Experiment config: core all baselines

```yaml
# configs/experiments/core_all_baselines.yaml

experiment:
  name: core_all_baselines
  seeds: [0, 1, 2, 3, 4]

models:
  - mlp
  - gcn
  - gat
  - gin

datasets:
  - cora
  - pubmed
  - ogbn-arxiv
  - ogbn-products
  - roman-empire
  - amazon-ratings
  - ogbg-molhiv
  - ogbg-molpcba
  - peptides-func
  - peptides-struct
  - ogbl-collab

training:
  max_epochs: 300
  patience: 50
  batch_size: 1024
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  grad_clip_norm: 1.0

logging:
  backend: local
  use_wandb: false

artifacts:
  save_best_checkpoint: true
  save_last_checkpoint: true
  save_predictions: false
```

---

## 15. CLI

Implement a Typer-based CLI.

Main commands:

```bash
uv run gnngym train --model gcn --dataset cora --seed 0
uv run gnngym train --model gat --dataset ogbn-arxiv --seed 1
uv run gnngym run-experiment --config configs/experiments/core_all_baselines.yaml
uv run gnngym evaluate --run-dir results/runs/<run_id>
uv run gnngym aggregate --runs results/runs --out results/tables/core_all_runs.parquet
uv run gnngym export-tables --input results/tables/core_all_runs.parquet
```

The CLI should also support overrides:

```bash
uv run gnngym train \
  --model gcn \
  --dataset cora \
  --seed 0 \
  --override training.lr=0.005 \
  --override model.hidden_channels=128
```

---

## 16. Git and GitHub rules

Use GitHub for version control.

Recommended branches:

```text
main
dev
feature/add-gcn
feature/add-gat
feature/add-gin
feature/add-ogb-products
feature/add-link-prediction
feature/add-my-new-architecture
experiment/core-benchmark-v1
```

Do not commit:

```text
data/
artifacts/
results/runs/
checkpoints
wandb logs
large downloaded datasets
```

Do commit:

```text
configs/
src/
tests/
scripts/
README.md
AGENTS.md
PROJECT_SPEC.md
pyproject.toml
uv.lock
```

Only commit exported result summaries under `results/tables/` intentionally.

---

## 17. `.gitignore`

Use this `.gitignore`:

```gitignore
# Python
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data
data/raw/
data/processed/
data/external/
data/ogb/
data/pyg/

# Experiment artifacts
artifacts/
results/runs/
*.pt
*.pth
*.ckpt
*.onnx

# Logs
logs/
*.log
wandb/
runs/

# Notebooks
.ipynb_checkpoints/

# System
.DS_Store
Thumbs.db
```

---

## 18. `AGENTS.md` content for Codex

Create this file at the repository root:

```md
# AGENTS.md

## Project goal

This repository benchmarks graph neural network architectures across a fixed core suite of graph datasets.

The main design principle is that models, datasets, trainers, and evaluators are modular.

## Environment

Use uv.

Common commands:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run gnngym train --model gcn --dataset cora --seed 0
```

## Code rules

- New models go in `src/gnn_gym/models/`.
- Register new models with `@register_model("name")`.
- Do not modify trainers when adding a standard message-passing model unless necessary.
- Keep dataset-specific logic in `src/gnn_gym/data/`.
- Keep task-specific training logic in `src/gnn_gym/training/`.
- Every new model must include a shape test.
- Every new dataset adapter must include a smoke-loading test if feasible.

## Artifact rules

- Do not commit `data/`, `artifacts/`, `results/runs/`, or model checkpoints.
- Save final aggregated tables under `results/tables/`.
- Use deterministic seeds where possible.
```

---

## 19. Tests

Implement lightweight tests that do not require downloading large datasets.

Required tests:

```text
tests/test_registry.py
tests/test_config_loading.py
tests/test_model_shapes.py
tests/test_toy_training.py
tests/test_result_aggregation.py
```

### 19.1 Registry tests

Check that models, datasets, trainers, and evaluators register correctly.

### 19.2 Config tests

Check that configs load and validate.

### 19.3 Model shape tests

For each model:

```text
input: synthetic graph with x and edge_index
output: expected node embedding or logits shape
```

### 19.4 Toy training test

Train GCN or MLP for 2 epochs on a tiny synthetic graph and verify that:

- no crash occurs
- metrics are written
- checkpoint is saved
- final metrics file exists

### 19.5 Aggregation test

Create fake run folders and verify that aggregation creates a valid CSV or Parquet table.

---

## 20. Reproducibility requirements

Every run must save:

- original config
- resolved config
- git commit hash
- seed
- device information
- package versions if feasible
- dataset name and split
- model parameter count
- best validation epoch
- final test metric
- total training time

Set seeds for:

```text
random
numpy
torch
torch.cuda
```

Use deterministic settings where practical, but do not sacrifice excessive speed unless explicitly requested.

---

## 21. Result table format

The aggregated result table should contain at least:

```text
run_id
experiment_name
model
dataset
task
metric_name
seed
best_epoch
val_metric
test_metric
train_time_seconds
num_parameters
device
git_commit
config_hash
status
```

For multi-metric datasets, include either:

```text
metric_name
metric_value
```

in long format, or one column per metric.

Prefer long format internally and wide format for exported tables.

---

## 22. Recommended benchmark execution plan

Start with a compact benchmark before running the full core list.

### Smoke benchmark

```text
Models:
- mlp
- gcn

Datasets:
- cora
- ogbg-molhiv

Seeds:
- 0
```

### Phase 1 benchmark

```text
Models:
- mlp
- gcn
- gat
- gin

Datasets:
- cora
- pubmed
- ogbn-arxiv
- ogbg-molhiv

Seeds:
- 0, 1, 2
```

### Full core benchmark

```text
Models:
- mlp
- gcn
- gat
- gin

Datasets:
- cora
- pubmed
- ogbn-arxiv
- ogbn-products
- roman-empire
- amazon-ratings
- ogbg-molhiv
- ogbg-molpcba
- peptides-func
- peptides-struct
- ogbl-collab

Seeds:
- 0, 1, 2, 3, 4
```

---

## 23. Main design risks

### 23.1 Mixing incompatible task types

Node classification, graph prediction, and link prediction require different heads, losses, and evaluators. Keep trainers separate.

### 23.2 Treating all metrics as comparable

Accuracy, ROC-AUC, average precision, MAE, and Hits@K are not directly comparable. Aggregate by task family and dataset. Use average rank only as an auxiliary summary.

### 23.3 Making GNN architectures task-specific

A new architecture should ideally be an encoder that can be reused by multiple heads. Avoid putting task-specific losses inside the model class.

### 23.4 Starting with too many datasets

Do not start with all datasets. First make the framework robust on toy data, then Cora/PubMed, then OGB, then graph prediction, then link prediction.

### 23.5 Saving too many artifacts

Saving predictions and every epoch checkpoint can consume large storage. Save only:

- best checkpoint
- last checkpoint
- final metrics
- optional predictions

---

## 24. Definition of done for first repository version

The first usable version of GNN Gym is complete when the following works:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run gnngym train --model gcn --dataset cora --seed 0
uv run gnngym train --model gat --dataset pubmed --seed 0
uv run gnngym train --model gin --dataset ogbg-molhiv --seed 0
uv run gnngym run-experiment --config configs/experiments/smoke_test.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/smoke_results.csv
```

The repository should then contain:

```text
src/gnn_gym/
configs/
tests/
scripts/
README.md
AGENTS.md
PROJECT_SPEC.md
pyproject.toml
uv.lock
```

and should not contain committed data, checkpoints, or large experiment artifacts.

---

## 25. Future extensions

After the first version works, consider adding:

- GraphSAGE
- GatedGCN
- GPS / Graph Transformer
- Graphormer-style model
- custom recurrent/recursive GNN architectures
- hyperparameter sweeps
- W&B artifact logging
- Slurm or tmux-based experiment launcher
- automatic LaTeX table export
- model FLOPs/parameter counting
- dataset download size reporting
- OOD and heterophily-specific summaries
