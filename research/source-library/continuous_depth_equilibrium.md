# GNN-GYM source-library note

This file treats the source as an idea generator, not as a recipe book. Candidate architectures below are research hypotheses for GNN-GYM. They are not claims of literature novelty until a separate prior-art check is done.

Each architecture should be evaluated with the GNN-GYM long-running protocol: toy crash check, seed-0 screen, confirmation on seeds `[0, 1, 2]`, validation-metric selection only, test metric for reporting only, and aggregation by `architecture_config_hash`.

# Source family: continuous-depth and equilibrium models

## Why this family may produce original GNN architectures

Neural ODEs and Deep Equilibrium Models suggest replacing a manually chosen stack depth with either continuous dynamics or an implicit fixed point. For GNNs, this creates a way to study propagation as a dynamical system: convergence, stiffness, local solver work, contraction, and residual error can become scientific signals, not just training details.

## Core uploaded sources

- `1806.07366v5.pdf` (Neural ODEs)
- `1909.01377v2.pdf` (Deep Equilibrium Models)
- `trefethenbook.pdf`

## Concepts to mine

- Continuous-depth hidden-state dynamics.
- Adaptive ODE solver computation.
- Precision/speed tradeoffs.
- Adjoint-style training.
- Continuous normalizing flows.
- Fixed points and root finding.
- Broyden/quasi-Newton methods.
- Implicit differentiation.
- Constant memory with respect to effective depth.
- Stiffness and stability.

## Known GNN/ML prior art to avoid merely reimplementing

- Standard continuous-depth GNNs.
- GRAND-style graph neural diffusion without a new graph mechanism.
- Implicit GNNs that simply wrap an existing layer in a fixed-point solver.
- DEQ reimplementation without residual/stability scientific instrumentation.

## Candidate mechanisms

- Solver work as a graph diagnostic: hard nodes/edges require more evaluations.
- Local time-step fields from learned graph stiffness.
- Fixed-point residual as uncertainty or rejection score.
- Contraction-constrained message operator.
- Hybrid explicit-local / implicit-global propagation.

## Synthetic tasks this source suggests

1. **Variable-depth dependency:** different nodes require different propagation lengths.
2. **Stiff graph dynamics:** some regions change fast, others slow; fixed-depth GNNs under/over compute.
3. **Convergence-as-confidence:** fixed-point residual should correlate with uncertainty or OOD structure.
4. **Implicit long-range task:** label depends on equilibrium of a graph process, not finite K-hop smoothing.
5. **Solver-budget task:** evaluate accuracy vs number of function evaluations.

## Architecture idea stubs

- `EventAdaptiveGraphODE`: continuous graph dynamics with event-triggered solver steps per graph or region.
- `GraphDEQResidualNet`: fixed-point GNN with residual monitor and contraction regularization.
- `HybridExplicitImplicitGNN`: local explicit flux updates plus global implicit correction.
- `SolverWorkReadoutGNN`: use solver iteration counts/residuals as auxiliary features for uncertainty-aware prediction.
- `StiffnessAwarePropagationGNN`: learn local stiffness estimates that control step size or damping.

## Rejected or low-novelty translations

- Wrap GCN in an ODE solver and claim novelty.
- Wrap GPR-GNN in a DEQ and claim novelty without a new equilibrium hypothesis.
- Ignore solver diagnostics; the solver behavior should be part of the insight.
