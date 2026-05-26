# GNN-GYM source-library note

This file treats the source as an idea generator, not as a recipe book. Candidate architectures below are research hypotheses for GNN-GYM. They are not claims of literature novelty until a separate prior-art check is done.

Each architecture should be evaluated with the GNN-GYM long-running protocol: toy crash check, seed-0 screen, confirmation on seeds `[0, 1, 2]`, validation-metric selection only, test metric for reporting only, and aggregation by `architecture_config_hash`.

# Source family: finite-volume and hyperbolic PDE methods

## Why this family may produce original GNN architectures

Finite-volume methods provide a different computational metaphor from ordinary message passing. Instead of smoothing node embeddings, they maintain cell averages, compute interface fluxes, solve local wave interactions, respect conservation form, and control numerical oscillation through limiters and stability conditions.

For graph learning, this suggests architectures where edges are interfaces, messages are fluxes, and aggregation is constrained by conservation/admissibility instead of unconstrained mean/sum pooling.

## Core uploaded sources

- `Leveque - Finite Volume Methods for Hyperbolic Equations.pdf`
- `urn_ch_slsp_9781009275484_ihv_pdf.pdf`
- `trefethenbook.pdf`

## Concepts to mine

- Conservation laws and conservation form.
- Cell averages and control volumes.
- Numerical flux functions.
- Riemann problems at interfaces.
- Upwind methods and wave speeds.
- CFL conditions and local stability.
- Flux/slope limiters.
- Total variation and oscillation control.
- Entropy conditions and admissible shocks.
- Boundary conditions and ghost cells.
- Variable coefficients and heterogeneous media.
- Homogenization of rapidly varying coefficients.
- Verification/validation and manufactured solutions.

## Known GNN/ML prior art to avoid merely reimplementing

- Generic MPNNs with edge gates.
- Attention-based GNNs where edge weights are unconstrained.
- Neural ODE GNNs without flux conservation.
- PDE-inspired GNNs that do not enforce local conservation or stability.

## Candidate mechanisms

- Edge flux states instead of only node hidden states.
- Antisymmetric pairwise fluxes with node update as graph divergence.
- Learned Riemann solvers at graph interfaces.
- Limiter functions that suppress high-frequency/unstable graph messages.
- Local CFL depth control: choose step size or number of propagation steps from learned edge speeds.
- Ghost boundary tokens for masked, low-degree, separator, or OOD boundary regions.

## Synthetic tasks this source suggests

1. **Graph advection task:** a signal moves along directed/weighted graph edges. The model must predict target labels after `T` steps.
2. **Graph shock task:** features contain discontinuities; the model should avoid oversmoothing across sharp class boundaries.
3. **Heterogeneous medium task:** edge types control propagation speed; the architecture must learn reflection/transmission-like behavior.
4. **Boundary ghost task:** mask boundary nodes and test whether ghost tokens improve extrapolation.
5. **Limiter task:** compare unconstrained aggregation against limiter-based propagation under feature noise.

## Architecture idea stubs

- `RiemannEdgeGNN`: each edge solves a learned two-state interface problem and emits waves with speeds and amplitudes.
- `TVDLimiterGNN`: each node computes limited neighbor increments to reduce oscillatory or oversmoothing updates.
- `EntropyFixGNN`: propagation accepts only messages that reduce a learned graph entropy or satisfy an admissibility gate.
- `CFLDepthGNN`: local edge speeds determine step size/depth; high-speed regions get smaller residual steps.
- `GhostBoundaryGNN`: creates ghost states for graph boundaries, low-degree nodes, split boundaries, or masked subgraphs.

## Rejected or low-novelty translations

- Calling attention weights "fluxes" without conservation form.
- Adding a scalar gate to GCN and calling it a limiter.
- Using fixed K-hop propagation and calling it CFL-aware without any stability mechanism.
