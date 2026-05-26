# GNN-GYM source-library note

This file treats the source as an idea generator, not as a recipe book. Candidate architectures below are research hypotheses for GNN-GYM. They are not claims of literature novelty until a separate prior-art check is done.

Each architecture should be evaluated with the GNN-GYM long-running protocol: toy crash check, seed-0 screen, confirmation on seeds `[0, 1, 2]`, validation-metric selection only, test metric for reporting only, and aggregation by `architecture_config_hash`.

# Source family: fluid simulation, projection, vorticity, and multigrid

## Why this family may produce original GNN architectures

Fluid simulation separates physical update steps: advection, forcing, projection, boundary handling, and sometimes vorticity/turbulence correction. That decomposition is more structured than a standard GNN layer. It suggests GNNs that predict provisional states and then project them onto graph-defined constraint manifolds.

## Core uploaded sources

- `Fluid Simulation for Computer Graphics, Second Edition.pdf`
- `urn_ch_slsp_9781009275484_ihv_pdf.pdf`
- `Leveque - Finite Volume Methods for Hyperbolic Equations.pdf`

## Concepts to mine

- Incompressibility and divergence-free velocity fields.
- Pressure projection.
- Control-volume enforcement of constraints.
- Semi-Lagrangian advection.
- Vorticity and vorticity confinement.
- Boundary conditions and solid obstacles.
- Local CFL substeps.
- Multigrid and domain decomposition.
- Residuals and declaring convergence.
- Source terms and pressure-velocity correction.

## Known GNN/ML prior art to avoid merely reimplementing

- Hodge-Laplacian or sheaf-inspired GNNs without a new projection objective.
- Generic residual GNNs with no conserved quantity.
- Hierarchical pooling methods that do not implement a coarse-grid correction or residual cycle.

## Candidate mechanisms

- Predict-provisional-then-project GNN layers.
- Graph Poisson correction for node or edge states.
- Edge-flux states decomposed into gradient-like and circulation-like parts.
- Coarse-grid residual correction in a graph hierarchy.
- Vorticity/cycle memory channels for ring-rich graphs.

## Synthetic tasks this source suggests

1. **Divergence-free routing:** edge flows must satisfy zero node imbalance except at source/sink nodes.
2. **Cycle circulation:** labels depend on circulation around cycles, not node-local features.
3. **Projection benefit:** compare raw GNN update vs projected update on noisy conservation tasks.
4. **Multigrid long-range:** graphs with bottlenecks where coarse correction should improve long-range propagation.
5. **Obstacle boundary:** remove/mark obstacle nodes and test learned boundary handling.

## Architecture idea stubs

- `DivergenceProjectionGNN`: compute provisional edge fluxes, then project them toward a constraint satisfying low graph divergence.
- `GraphPressureCorrectionNet`: solve an approximate graph Poisson correction from node residuals and apply it to messages.
- `VorticityMemoryGNN`: maintain cycle/curl-like edge memory alongside node embeddings.
- `MultigridResidualGNN`: alternate fine updates with coarse residual correction and prolongation.
- `ObstacleGhostGNN`: introduce learned boundary/obstacle ghost states for masked or separator nodes.

## Rejected or low-novelty translations

- A standard graph pooling/unpooling stack without residual equation semantics.
- A GNN with edge features named "pressure" but no projection step.
- Using cycles as positional encodings without testing a cycle/circulation hypothesis.
