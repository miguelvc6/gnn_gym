# How to mine uploaded sources for original GNN architectures

## Purpose

This source library is meant to be used the way Diestel's graph theory book was used: mine sources for mechanisms, invariants, operators, stability principles, decompositions, and falsifiable hypotheses. Do not simply reimplement known architectures with a new name.

## Template for each source family

Each source-family file uses this structure:

1. Why this family may produce original GNN architectures.
2. Core uploaded sources.
3. Concepts to mine.
4. Known GNN/ML prior art to avoid merely reimplementing.
5. Candidate mechanisms.
6. Synthetic tasks this source suggests.
7. Architecture idea stubs.
8. Rejected or low-novelty translations.

## Architecture idea standard

Each architecture idea should record:

```text
Scientific hypothesis:
Mechanism:
Source inspiration:
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

## General warning

A model is not a novel architecture merely because it combines two known methods. The idea needs a specific mechanism and a falsifiable claim.

## Source evidence anchors from the uploaded documents

- Diestel's graph theory book motivates mining graph minors, tree-decompositions, regularity, cycle spaces, and cut spaces.
- LeVeque's finite-volume book motivates local fluxes, Riemann problems, limiters, CFL conditions, ghost cells, conservation form, entropy conditions, and shock/contact structure.
- Bridson's fluid simulation book motivates incompressibility, pressure projection, semi-Lagrangian advection, vorticity, local CFL substeps, control-volume reasoning, and multigrid/domain decomposition.
- MeshGraphNets motivates mesh-space/world-space separation, adaptive meshing, learned sizing fields, local remeshing, and resolution-independent dynamics.
- Fourier Neural Operator motivates learning solution operators between function spaces, Fourier-space kernels, mesh invariance, and zero-shot super-resolution.
- Neural ODEs motivate continuous-depth propagation, adaptive solver computation, adjoint-style training, and precision/speed tradeoffs.
- Deep Equilibrium Models motivate fixed-point graph encoders, root finding, implicit differentiation, infinite-depth tied dynamics, and residual monitors.
- Trefethen's spectral/numerical PDE sources motivate spectral accuracy, Fourier/Chebyshev operators, interpolation, boundary conditions, stability regions, dissipation/dispersion, pseudospectra, and multigrid/preconditioning.
- Reddy/Anand/Roy's finite-element/finite-volume source motivates verification/validation, manufactured solutions, residual convergence, control volumes, pressure-velocity coupling, source terms, grid transfer, and multigrid cycles.
