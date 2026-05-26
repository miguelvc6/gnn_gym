# Agent Scratchpad

This file is for working memory across research sessions: what was tried, what happened, what is worth trying next, and what to avoid repeating. It is allowed to be messy, but every entry should be dated and actionable.

Do not use this file for final claims. Promote durable conclusions to `research/INSIGHTS.md`.

## Current Benchmark Discipline

- Keep the harness fixed while evaluating an architecture idea.
- Prefer `toy-node`, `toy-graph`, and `toy-link` for crash checks.
- Toy metrics are never architecture evidence.
- Use Cora/PubMed as the first real node-classification checks.
- Use `ogbg-molhiv` as the first real graph-prediction check.
- Avoid running `ogbn-products`, Peptides, or `ogbl-collab` during casual iteration unless the user
  explicitly wants a long run.
- Select configs by validation metric only; record test metrics for held-out reporting.
- Treat seed `0` as screening only and confirm promising configs across seeds `[0, 1, 2]`.
- Use config-level aggregation by `architecture_config_hash` for claims. Model-level mixed-config
  means are diagnostics only.
- Every architecture idea should state a scientific hypothesis, closest known baselines, and a
  minimal falsifying experiment before implementation.

## Experiment Log Template

```md
## YYYY-MM-DD - Experiment Name

Branch/commit:
Files changed:
Command:
Result:
Keep/discard:
Notes:
Next:
```

## 2026-05-26 - Research Infrastructure Hardening Note

Branch/commit: local uncommitted work
Files changed: benchmark plumbing and research protocol Markdown
Command:

```bash
uv run ruff check .
uv run pytest
```

Result:

- Added seed-independent `architecture_config_hash` protocol for config-level evidence.
- Research claims should use `*_by_config_mean_std.csv`; `*_by_model_mean_std.csv` is mixed-config
  diagnostic output only.
- Edge attributes now belong in the model call path for future edge-aware graph/link models.
- `neighbor_node` is expected to use PyG `NeighborLoader` when available; fallback is for smoke and
  compatibility, not serious `ogbn-products` evidence.

Keep/discard: keep as protocol update

Next:

- Re-export old aggregate tables with the new export command before using them for future claims.
- Add edge-aware graph baselines before making strong MolHIV architecture claims.

## Tried

## 2026-05-22 - Node Baseline Hyperparameter Search Plan

Branch/commit: local worktree, current HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: `research/AGENT_SCRATCHPAD.md`
Command:

```bash
git status --short
uv run gnngym train --model <model> --dataset toy-node --seed 0 --override training.max_epochs=2
uv run gnngym run-sweep --config <compact per-model sweep config>
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- Existing `results/tables/baseline_all_runs_mean_std.csv` shows GAT as the current strongest node
  baseline by validation accuracy: Cora `0.7313 +/- 0.0110`, PubMed `0.7607 +/- 0.0167`.
- Current GCN baseline is weaker and unstable, especially PubMed: Cora `0.6380 +/- 0.0330`,
  PubMed `0.6507 +/- 0.1307`.
- Current GIN baseline is stronger than GCN but below GAT: Cora `0.6613 +/- 0.0115`,
  PubMed `0.7300 +/- 0.0212`.
- Test metrics are recorded in the baseline tables but will not be used for selecting configs.

Keep/discard: keep plan

Notes:

- First search batch will use seed `0`, Cora/PubMed, `training.max_epochs=50`,
  `training.patience=15`, and the fixed trainers/evaluators.
- Compact GCN batch: include default plus modest wider/deeper settings:
  `hidden_channels in [64, 128]`, `num_layers in [2, 3]`, `dropout=0.2`,
  `training.lr in [0.01, 0.005]`.
- Compact GAT batch: include default heads and test attention dropout removal:
  `hidden_channels in [16, 32]`, `heads=4`, `dropout=0.2`,
  `attention_dropout in [0.0, 0.2]`, `training.lr in [0.01, 0.005]`.
- Compact GIN batch: include default plus wider/deeper settings:
  `hidden_channels in [64, 128]`, `num_layers in [3, 5]`, `dropout=0.2`,
  `eps_trainable=true`, `training.lr in [0.01, 0.005]`.
- Ideas mined from `research/original-ideas/`: the bounded architecture candidates after baseline
  tuning are `revision_gnn` or a smaller residual/recurrent-GIN-style model. Cavity/non-backtracking
  edge-state models are interesting but likely a larger first implementation because they carry
  directed edge state. RIGN should be staged later because its deep supervision and improvement
  losses would otherwise require trainer changes.

Next:

- Run toy-node crash checks for GCN/GAT/GIN under representative first-batch overrides.
- Run the compact sweeps and aggregate/export tables.
- Pick confirmation candidates using validation metric only; confirm across seeds `0, 1, 2` with
  `training.max_epochs=200` and `training.patience=50` before changing any default YAML.

## 2026-05-22 - Node Hyperparameter Search Results And RevisionGNN First Pass

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: `configs/experiments/research_2026_05_22_*.yaml`,
`src/gnn_gym/models/revision_gnn.py`, `configs/models/revision_gnn.yaml`,
`src/gnn_gym/registry.py`, `tests/test_model_shapes.py`, research notes
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gcn_batch1.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gat_batch1.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_batch1.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gcn_batch2.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gat_batch2.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_batch2.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gin_pubmed_confirm.yaml
uv run gnngym train --model revision_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_revision_gnn_fast.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
uv run ruff check .
uv run pytest
```

Result:

- Aggregated research table now has `185` rows in `results/tables/research_all_runs.csv`.
- GCN: no fast-search config beat the seed-0 baseline. Best Cora fast result was
  `0.6280` validation for `hidden=64,layers=3,dropout=0.7,lr=0.01,weight_decay=0.0001`; best
  PubMed fast result was `0.6160` validation for `hidden=64,layers=2,dropout=0.5,lr=0.01,
  weight_decay=0.0`.
- GAT: no refinement beat the existing seed-0 baseline. Default-like `hidden=32,heads=4,
  dropout=0.2,attention_dropout=0.2,lr=0.01` remained best on Cora at `0.7240` validation; best
  PubMed fast result was `0.7560`, below the existing seed-0 baseline `0.7660`.
- GIN: `hidden=128,layers=3,dropout=0.5,eps_trainable=true,lr=0.01` confirmed on Cora across seeds
  `0,1,2`: validation mean `0.6773` versus baseline `0.6613`. Test mean was `0.6627` versus
  baseline `0.6710`; recorded only as held-out reporting.
- GIN PubMed candidate `hidden=64,layers=5,dropout=0.2,eps_trainable=true,lr=0.01` did not confirm:
  validation mean `0.7147` versus baseline `0.7300`.
- Architecture idea: implemented bounded `revision_gnn` from the belief-revision design mine. It
  passed toy-node and shape tests. The best fast config was confirmed across seeds `0,1,2` with the
  200-epoch budget and trailed confirmed GAT validation means on both Cora (`0.7020` versus
  `0.7313`) and PubMed (`0.7253` versus `0.7607`). Discard first-pass implementation.
- Validation metric was used for decisions. Test metrics were recorded but not used to choose.
- `uv run ruff check .` passed.
- `uv run pytest` passed: `17 passed, 2 skipped`.

Keep/discard:

- Keep confirmed Cora GIN tuning as a dataset-specific research config/insight.
- Discard GCN batch 1/2 and GAT batch 1/2 refinements.
- Discard PubMed GIN deeper candidate.
- Discard first-pass `revision_gnn` as a performance candidate; keep code only as an explicit
  bounded experiment artifact unless the project wants to remove discarded model code.

Notes:

- Do not update `configs/models/gin.yaml`: the Cora tuning did not transfer to PubMed.
- `research_results.tsv` and `results/runs/` remain local research artifacts.
- The sweep TSV writer initially recorded seed `0` as an empty field because it used a truthiness
  check when serializing metadata. Fixed in `src/gnn_gym/experiments/sweep.py` and covered by
  `tests/test_runtime_features.py`; older rows in the local ledger still show the previous blank.

Next:

- If continuing hyperparameter search, focus GIN Cora around `hidden=128,layers=3,dropout=0.4-0.6`
  or GAT PubMed around the existing default; do not spend more fast budget on the explored GCN
  settings.
- If continuing architecture work, prefer a residual GIN or non-backtracking model with a sharper
  same-budget comparison target rather than expanding `revision_gnn` without diagnostics.

## 2026-05-19 - Repository Scaffold And Multi-Task Trainers

Branch/commit: local uncommitted work
Files changed: `src/gnn_gym/`, `configs/`, `tests/`, `pyproject.toml`, `uv.lock`
Command:

```bash
uv run ruff check .
uv run pytest
uv run gnngym train --model gcn --dataset toy-node --seed 0 --override training.max_epochs=2
uv run gnngym train --model gcn --dataset cora --seed 0 --override training.max_epochs=1
uv run gnngym train --model gat --dataset pubmed --seed 0 --override training.max_epochs=1
uv run gnngym train --model gin --dataset ogbg-molhiv --seed 0 --override training.max_epochs=1
```

Result:

- Lint passes.
- Unit tests pass.
- Toy node, graph, link, neighbor-node smoke paths pass.
- Cora, PubMed, and MolHIV smoke runs completed.

Keep/discard: keep

Notes:

- OGB loaders need a PyTorch 2.6 compatibility wrapper around `torch.load(weights_only=False)`.
- Current graph trainer pools model node outputs directly. Long term, this should become cleaner
  encoder/head separation.

Next:

- Add a formal experiment runner that writes a compact TSV/CSV summary per architecture trial.
- Add architecture-specific result tables with mean/std over seeds.

## Future Queue

- Implement `res_gcn` as first architecture research candidate.
- Add richer graph pooling head.
- Add GraphSAGE as a scalable node-classification baseline.
- Add a small fixed-budget benchmark config analogous to autoresearch's fixed 5-minute run.

## 2026-05-23 - Complete Remaining Markdown Ideas Pass

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: new model/config/sweep files for `nb_light_gnn`, `confidence_appnp_net`,
`bethe_gnn`, `dual_primal_gnn`, `equilibrium_belief_gnn`, `region_collapse_gnn`, `kikuchi_gnn`,
`loop_corrected_gnn`, `decimation_gnn`, and `walk_belief_transformer`; graph pooling head support;
research notes and exported tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_nb_light_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_confidence_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_bethe_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_dual_primal_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_equilibrium_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_region_collapse_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_kikuchi_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_loop_corrected_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_decimation_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_walk_belief_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_concat_pool_toy_graph.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_23_concat_pool_molhiv_fast.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `nb_light_gnn`: implemented generated directed-edge light propagation. Best validation was Cora
  `0.6900`, PubMed `0.7900`; below confirmed GPR and below confirmed `nb_belief_gnn` on PubMed.
- `confidence_appnp_net`: implemented generated confidence-gated APPNP. Best validation was Cora
  `0.6420`, PubMed `0.7180`; discarded.
- `bethe_gnn`: implemented no-auxiliary-loss Bethe-style node/edge consistency. Best validation was
  Cora `0.4580`, PubMed `0.5580`; discarded.
- `dual_primal_gnn`: implemented coupled node/edge factor states. Best validation was Cora
  `0.5020`, PubMed `0.6620`; discarded.
- `equilibrium_belief_gnn`: implemented unrolled recurrent fixed-point-style propagation. Best
  validation was Cora `0.3160`, PubMed `0.4780`; discarded.
- `region_collapse_gnn`: implemented learned soft region coarsening. Best validation was Cora
  `0.6520`, PubMed `0.7520`; discarded.
- `kikuchi_gnn`: implemented ego-region messages. Best validation was Cora `0.5460`, PubMed
  `0.6960`; discarded.
- `loop_corrected_gnn`: implemented capped triangle loop correction. Best validation was Cora
  `0.4060`, PubMed `0.6440`; discarded.
- `decimation_gnn`: implemented trainer-free hidden-state decimation. Best validation was Cora
  `0.6020`, PubMed `0.7520`; discarded.
- `walk_belief_transformer`: implemented deterministic short-walk transformer. Best validation was
  Cora `0.6640`, PubMed `0.7200`; discarded.
- Dataset-conditioned pooling: implemented opt-in `mean_max_add` pooling. Toy-graph smoke passed.
  Short MolHIV seed-0 check gave `mean_max_add` validation `0.7489` versus matched short mean
  pooling `0.7163`, but below existing GIN MolHIV baseline mean `0.7619`; do not promote.
- Aggregated research table now has `1731` rows.

Keep/discard:

- No new model from this pass beats confirmed GPR on Cora/PubMed.
- Keep the implementations as explicit bounded experiment artifacts and negative results.
- Keep `mean_max_add` as an opt-in graph pooling mode only; do not update any default model YAML.

Notes:

- The broad result is consistent with the earlier loop: complex belief/region/recurrent machinery
  under the fixed harness does not beat simple decoupled learnable propagation on Cora/PubMed.
- Ideas that likely require auxiliary losses or trainer changes, especially Bethe/RIGN/decimation,
  should not be reopened as plain encoders.

Next:

- Use `gpr_gnn` as the node benchmark to beat.
- Future work should move to matched `ogbn-arxiv` checks or graph-task-specific architectures
  rather than further Cora/PubMed encoder-only variants.

## 2026-05-22 - Generated Architecture Search: NB-APPNP, Residual APPNP, GPR

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: `research/GENERATED_ARCHITECTURE_IDEAS.md`, new model/config/sweep files for
`nb_appnp_net`, `gated_appnp_net`, `res_appnp_net`, and `gpr_gnn`; research notes and exported
tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_nb_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gated_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_appnp_pubmed_confirm_prop5.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_cora_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_pubmed_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_cora_refined_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gpr_pubmed_refined_confirm.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `nb_appnp_net`: discarded. Best fast validation was Cora `0.6720`, PubMed `0.7240`.
- `gated_appnp_net`: discarded. Best fast validation was Cora `0.7180`, PubMed `0.7300`.
- `res_appnp_net`: not a new best. Cora confirmation was validation mean `0.7733`, below APPNP
  `0.7753`; PubMed confirmation was `0.7900`, above APPNP `0.7880` but below NB-belief `0.7947`.
  A second PubMed config confirmed at `0.7833`.
- `gpr_gnn`: kept. Initial confirmation reached Cora validation mean `0.7807`, above APPNP
  `0.7753`, and PubMed validation mean `0.8007`, above NB-belief `0.7947` and APPNP `0.7880`.
- GPR refinement: keep refined Cora config `dropout=0.1,propagation_steps=16,alpha=0.1`, which
  confirmed validation mean `0.7827`. Discard refined PubMed config `dropout=0.3,alpha=0.05`, which
  confirmed at `0.7967`, below the original GPR PubMed `0.8007`.
- Aggregated research table now has `1318` rows.

Keep/discard:

- Keep `gpr_gnn` as the strongest current Cora/PubMed node architecture from this loop.
  Validation-selected configs: Cora `hidden=128,layers=1,dropout=0.1,K=16,alpha=0.1`; PubMed
  `hidden=64,layers=1,dropout=0.2,K=10,alpha=0.1`.
- Discard direct NB/APPNP fusion and unconstrained APPNP gating.
- Keep `res_appnp_net` only as an informative negative/marginal experiment; do not promote it over
  `gpr_gnn`.

Notes:

- Conservative generated ideas worked better than elaborate hybrids. The successful change was
  learnable hop weights around an APPNP-like prior, not more edge-state machinery.
- Seed-0 lifts are still noisy: residual APPNP looked strong at seed 0 but did not beat the best
  confirmed baselines across seeds.

Next:

- Run a matched `gpr_gnn`/APPNP/GAT scale check on `ogbn-arxiv` only after the Cora/PubMed notes
  are clean and tests pass.

## 2026-05-22 - Belief-Mined Architecture Sweeps: Frustration, NB, Entropy, RIGN, Ladder, Survey

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: new model/config/sweep files for `frustration_gnn`, `nb_belief_gnn`,
`entropy_gated_gnn`, `rign_gnn`, `temp_ladder_gnn`, and `survey_gnn`; research notes and exported
tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_frustration_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_nb_belief_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_nb_belief_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_entropy_gated_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_rign_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_temp_ladder_fast.yaml
uv run gnngym train --model survey_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_survey_fast.yaml
uv run gnngym train --model cavity_gnn --dataset toy-node --seed 0 --override training.max_epochs=2 --override training.patience=2
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_cavity_fast.yaml
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `nb_belief_gnn`: kept for PubMed. Confirmed across seeds `0,1,2` with validation mean `0.7947`,
  above confirmed APPNP PubMed `0.7880`. Held-out test mean was lower than APPNP and is recorded
  only for reporting.
- `frustration_gnn`: discarded. Best fast validation was Cora `0.6840`, PubMed `0.7460`.
- `entropy_gated_gnn`: discarded. Best fast validation was Cora `0.6940`, PubMed `0.7820`.
- `rign_gnn`: discarded as an encoder-only shortcut. Best fast validation was Cora `0.5680`,
  PubMed `0.7060`.
- `temp_ladder_gnn`: discarded. Best fast validation was Cora `0.6100`, PubMed `0.7340`.
- `survey_gnn`: discarded. Best fast validation was Cora `0.6960`, PubMed `0.7760`.
- `cavity_gnn`: discarded. Best fast validation was Cora `0.3340`, PubMed `0.6720`.
- Aggregated research table now has `997` rows.

Keep/discard:

- Keep `nb_belief_gnn` as a PubMed validation improvement, with the caveat that the current Python
  reverse-edge lookup is a small-dataset prototype.
- Discard the first-pass frustration, entropy-gated, RIGN encoder-only, temperature-ladder,
  survey-particle, and GRU-cavity models.

Notes:

- The strongest repeated signal from the original ideas is not generic recurrence or extra latent
  particles; it is directed non-backtracking edge state.
- The full GRU cavity update performed much worse than the simpler `nb_belief_gnn`, so future
  edge-state hybrids should preserve simple stable propagation and add a conservative fusion path.
- Encoder-only versions of ideas that originally relied on auxiliary objectives underperformed.
  Reopening RIGN or SurveyGNN should include the intended supervision/diagnostic rather than simply
  widening the same encoders.

Next:

- Create `research/GENERATED_ARCHITECTURE_IDEAS.md` from the observed insights and start testing a
  newly generated idea that combines APPNP's stable propagation with NB's directed edge signal.

## 2026-05-22 - Continued Architecture Search: ResGIN, JK-GCN, APPNP, GCNII, GATv2

Branch/commit: local uncommitted work on HEAD `46f3954891a4714fd29028c43b70d24bddd37580`
Files changed: new model/config/sweep files for `res_gin`, `jk_gcn`, `appnp_net`, `gcn2_net`,
and `gatv2`; research notes and exported tables
Command:

```bash
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_res_gin_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_jk_gcn_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_jk_gcn_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_pubmed_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_cora_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_pubmed_refine.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_appnp_cora_refined_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gcn2_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gatv2_fast.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gatv2_cora_confirm.yaml
uv run gnngym run-sweep --config configs/experiments/research_2026_05_22_gatv2_pubmed_confirm.yaml
uv run gnngym train --model appnp_net --dataset ogbn-arxiv --seed 0 --override training.max_epochs=50 ...
uv run gnngym aggregate --runs results/runs --out results/tables/research_all_runs.csv
uv run gnngym export-tables --input results/tables/research_all_runs.csv --out-dir results/tables
```

Result:

- `res_gin`: discarded. Best fast validation was Cora `0.5220`, PubMed `0.6980`.
- `jk_gcn`: kept for PubMed. Confirmed validation mean `0.7793` versus GAT PubMed `0.7607`.
- `appnp_net`: current strongest Cora/PubMed model. Confirmed Cora validation mean `0.7753`
  versus GAT `0.7313`; confirmed PubMed validation mean `0.7880` versus GAT `0.7607` and JK-GCN
  `0.7793`.
- APPNP Cora refinement found a seed-0 run at `0.8040`, but the 3-seed refined confirmation
  underperformed the original APPNP Cora confirmation (`0.7700` versus `0.7753`), so keep the
  original Cora APPNP config.
- `gcn2_net`: discarded. Best fast validation was Cora `0.4880`, PubMed `0.6260`.
- `gatv2`: kept as a Cora attention-family improvement. Confirmed Cora validation mean `0.7553`
  versus GAT `0.7313`; PubMed did not confirm versus GAT.
- APPNP `ogbn-arxiv` seed-0 scale smoke completed with validation `0.5810`, test `0.5123`, and
  train time `14.7s`. No arxiv baseline table exists yet, so this is not an improvement claim.
- `results/tables/research_all_runs.csv` now has `603` rows.

Keep/discard:

- Keep `appnp_net`, `jk_gcn`, and Cora `gatv2` as validated architecture results.
- Discard `res_gin`, `gcn2_net`, PubMed `gatv2`, and refined APPNP Cora config.

Notes:

- APPNP should now be the node architecture benchmark to beat on Cora/PubMed.
- For medium-node follow-up, establish matched `gcn`, `gat`, `jk_gcn`, and `appnp_net` baselines on
  `ogbn-arxiv` before making claims.

Next:

- Run lint/tests after the expanded model set.
- If continuing, try APPNP weight decay and alpha refinements, or a graph-specific GIN pooling
  experiment on `ogbg-molhiv` with a small run budget.
