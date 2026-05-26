# GNN-GYM source-library note

This file treats the source as an idea generator, not as a recipe book. Candidate architectures below are research hypotheses for GNN-GYM. They are not claims of literature novelty until a separate prior-art check is done.

Each architecture should be evaluated with the GNN-GYM long-running protocol: toy crash check, seed-0 screen, confirmation on seeds `[0, 1, 2]`, validation-metric selection only, test metric for reporting only, and aggregation by `architecture_config_hash`.

# Source family: spectral methods, stability, pseudospectra, and numerical PDE analysis

## Why this family may produce original GNN architectures

Spectral numerical methods emphasize global bases, smoothness, stability regions, boundary conditions, and the behavior of repeated linear operators. This is closely related to oversmoothing, oversquashing, feature oscillation, non-normal propagation, and depth stability in GNNs.

## Core uploaded sources

- `Trefethenespectral.pdf`
- `trefethenbook.pdf`
- `2010.08895v3.pdf`

## Concepts to mine

- Fourier and Chebyshev differentiation matrices.
- Spectral accuracy for smooth signals.
- Gibbs phenomenon near discontinuities.
- Stability regions and stiffness.
- Dissipation, dispersion, and group velocity.
- Modified equations.
- Pseudospectra and non-normal matrices.
- Boundary conditions.
- Preconditioning and multigrid.

## Known GNN/ML prior art to avoid merely reimplementing

- ChebNet or generic polynomial spectral GNN.
- GPR-GNN/BernNet style global polynomial filters without stability diagnostics.
- Laplacian positional encodings alone.
- Existing graph wavelet/scattering models without new boundary or stability mechanism.

## Candidate mechanisms

- Learn a stable rational/spectral propagation operator with explicit stability-region constraints.
- Detect discontinuities and switch from spectral/global mode to local/limited mode.
- Penalize non-normal amplification via cheap surrogate pseudospectral diagnostics.
- Boundary-aware spectral filters with ghost-node correction.
- Separate dissipative and dispersive propagation channels.

## Synthetic tasks this source suggests

1. **Smooth vs discontinuous graph signal:** test whether spectral/global mode helps smooth tasks but hurts discontinuity tasks.
2. **Non-normal amplification task:** directed graphs where repeated propagation explodes unless stabilized.
3. **Boundary-condition transfer:** same interior graph with different boundary labels/conditions.
4. **Dissipation/dispersion diagnostic:** track whether features smear or travel at wrong effective speed.
5. **Spectral super-resolution:** train on small graph discretizations and evaluate on finer discretizations.

## Architecture idea stubs

- `StabilityRegionGNN`: parameterize propagation coefficients inside a learned stable region.
- `PseudospectralGuardGNN`: block or damp updates with high estimated non-normal amplification.
- `GraphSpectralLimiterGNN`: use spectral/global updates only where the graph signal appears smooth; otherwise switch to local limited flux.
- `DissipationDispersionGNN`: separate smoothing and transport channels with diagnostics over graph frequencies.
- `BoundaryCorrectedSpectralGNN`: spectral operator plus learned ghost-boundary correction.

## Rejected or low-novelty translations

- Any polynomial graph filter without a new stability or boundary principle.
- Using more hops and calling it spectral accuracy.
- Reporting only accuracy without measuring stability, depth, or transfer behavior.
