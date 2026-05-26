# Generated Architecture Ideas

These ideas were generated after the 2026-05-22 research loop over baseline tuning and
`research/original-ideas/`. They should be treated as bounded follow-up experiments, not default
model changes.

## Observed Signals

- APPNP is the strongest confirmed Cora/PubMed node baseline so far: Cora validation mean `0.7753`,
  PubMed validation mean `0.7880`.
- Non-backtracking directed edge state is the only original-ideas branch that beat APPNP on a
  primary validation metric: PubMed validation mean `0.7947`.
- More complex belief machinery did not help in the fixed harness. GRU cavity messages,
  multi-particle survey states, temperature ladders, uncertainty gates, and encoder-only RIGN all
  trailed APPNP/NB.
- Promising new ideas should therefore be conservative hybrids: keep APPNP's stable personalized
  propagation path and add a small directed-edge or anti-oversmoothing correction.

## 2026-05-22 - NB-APPNP Hybrid

Status: discarded
Target tasks: node
Expected benefit: Preserve APPNP's strong feature-propagation behavior on Cora/PubMed while adding
the directed non-backtracking signal that improved PubMed.
Risk: The fusion gate can ignore the NB branch or overfit seed-0 validation.
Minimal experiment: Add `nb_appnp_net`, run toy-node, then compact Cora/PubMed fast sweeps.
Success criterion: Beat confirmed APPNP on Cora or confirmed NB-belief on PubMed by validation
metric, then confirm seeds `0,1,2`.

Description:

Encode node features with an MLP. Send one copy through APPNP propagation. Send another copy through
a simple non-backtracking directed-edge message loop. Fuse raw encoded features, APPNP features,
and NB features through a learned gate and projection.

First result:

Fast Cora/PubMed sweeps did not beat APPNP or NB. Best validation was Cora `0.6720` and PubMed
`0.7240`, so this direct fusion is discarded.

## 2026-05-22 - APPNP With Anti-Smoothing Residual

Status: discarded
Target tasks: node
Expected benefit: Keep APPNP's propagation but preserve high-frequency node-feature residuals when
propagation oversmooths.
Risk: May duplicate what the APPNP teleport already does.
Minimal experiment: Add an MLP residual branch and learn a scalar/vector gate between encoded
features and APPNP output.
Success criterion: Improve Cora validation over confirmed APPNP without hurting PubMed validation.

First result:

An unconstrained `gated_appnp_net` was too disruptive: best validation was Cora `0.7180` and PubMed
`0.7300`. Continue this idea only with a conservative residual initialized near plain APPNP.

Second result:

The conservative `res_appnp_net` confirmed Cora validation mean `0.7733`, below APPNP `0.7753`.
Its best PubMed confirmation reached validation mean `0.7900`, slightly above APPNP `0.7880` but
below NB-belief `0.7947`; a second PubMed config confirmed at `0.7833`. Discard as a primary model.

## 2026-05-22 - Directed-Edge Light Propagation

Status: discarded
Target tasks: node
Expected benefit: Replace the Python-heavy NB implementation with a lighter directed-edge
propagation block that can be used on larger citation graphs.
Risk: Engineering improvement may not change validation metrics.
Minimal experiment: Implement vectorized reverse-edge indexing/cache inside a new model only, then
compare runtime and validation to `nb_belief_gnn`.
Success criterion: Match NB PubMed validation within noise while materially reducing runtime.

First result:

Implemented as `nb_light_gnn` with vectorized reverse-edge lookup and a simpler directed update.
Fast Cora/PubMed sweeps completed, but best validation was Cora `0.6900` and PubMed `0.7900`,
below confirmed GPR and below confirmed `nb_belief_gnn` on PubMed. Treat this as a runtime-oriented
negative result, not a replacement.

## 2026-05-22 - Learnable PageRank Propagation

Status: kept
Target tasks: node
Expected benefit: Generalize APPNP by learning the mixture weight for each propagation hop instead
of fixing the personalized PageRank coefficients.
Risk: Extra hop weights can overfit Cora/PubMed seed-0 unless confirmed across seeds.
Minimal experiment: Add `gpr_gnn` as an MLP encoder followed by learned normalized propagation
weights, then run the same fast Cora/PubMed grid as APPNP.
Success criterion: Beat confirmed APPNP on Cora or confirmed NB-belief on PubMed by validation
metric after seeds `0,1,2`.

Description:

Encode features with an MLP, iteratively apply GCN-normalized propagation, and learn a scalar
coefficient for each hop output. Initialize the coefficients to APPNP-like PageRank weights so the
model starts from a stable propagation prior.

First result:

Confirmed across seeds `0,1,2`: Cora validation mean `0.7807`, beating APPNP `0.7753`; PubMed
validation mean `0.8007`, beating NB-belief `0.7947` and APPNP `0.7880`. Keep as the strongest
current generated architecture.

Refinement result:

A narrow GPR refinement confirmed a Cora lift with `dropout=0.1`, `propagation_steps=16`,
`alpha=0.1`: validation mean `0.7827`. The PubMed refinement `dropout=0.3`, `alpha=0.05` confirmed
below the original PubMed GPR config (`0.7967` versus `0.8007`), so keep the original PubMed GPR
setting.

## 2026-05-22 - Confidence-Gated APPNP

Status: discarded
Target tasks: node
Expected benefit: Let nodes with strong MLP predictions retain more feature signal while uncertain
nodes use more propagated evidence.
Risk: Learned confidence can collapse without calibration supervision.
Minimal experiment: Predict a gate from encoded node features and mix encoded features with APPNP
features.
Success criterion: Beat APPNP seed-0 validation on Cora/PubMed before any confirmation run.

First result:

Implemented as `confidence_appnp_net`. Fast Cora/PubMed sweeps were not competitive: best
validation was Cora `0.6420` and PubMed `0.7180`. Discard this confidence-gate formulation.
