# Original GNN architecture ideas from uploaded graph-theory, PDE, fluid, spectral, ODE, DEQ, FNO, and MeshGraphNet sources

Date: 2026-05-26

Intended location:

```text
research/original-ideas/2026-05-26_uploaded_docs_original_gnn_ideas.md
```

## Status and novelty warning

These are **candidate original GNN-GYM research hypotheses**, not literature novelty claims. Before a paper-style claim, run a prior-art search for each specific mechanism. The objective is to give the long-running agent loop a diverse backlog of architecture ideas that are source-grounded, falsifiable, and capable of producing scientific insight even when they fail.

## Evaluation protocol reminder

Use the long-running GNN-GYM protocol:

1. Toy crash check only; toy metrics are not evidence.
2. Seed-0 fast screen.
3. Confirm promising configs with seeds `[0, 1, 2]`.
4. Select by validation metric only.
5. Record test metric only for held-out reporting.
6. Aggregate by `architecture_config_hash`.
7. Compare to task-appropriate baselines under the same budget.
8. Log negative results in `research/INSIGHTS.md` if they clarify the hypothesis.

## Source map used

- Diestel: graph minors, tree-decompositions, cycle/cut spaces, structural decomposition.
- LeVeque: conservation laws, Riemann problems, Godunov methods, CFL conditions, limiters, ghost cells, entropy conditions.
- Bridson: incompressibility, pressure projection, vorticity, semi-Lagrangian advection, local CFL substeps, multigrid.
- MeshGraphNets: mesh-space/world-space messages, adaptive mesh, learned sizing fields, resolution-independent dynamics.
- FNO: neural operators between function spaces, Fourier kernels, mesh invariance, super-resolution.
- Neural ODE: continuous-depth hidden dynamics, adaptive solver work, adjoint-style training.
- DEQ: fixed points, root finding, implicit differentiation, constant memory effective depth.
- Trefethen sources: spectral accuracy, Gibbs behavior, stability regions, pseudospectra, dissipation/dispersion, boundary conditions.
- Reddy/Anand/Roy: verification/validation, manufactured solutions, residual convergence, control volumes, multigrid cycles, pressure-velocity coupling.

---

# Wave 1: bounded implementations likely to fit the current gym

## 1. RiemannEdgeGNN

```text
Scientific hypothesis:
Standard GNN aggregation fails near class or feature discontinuities because it averages across interfaces. A learned edge-local Riemann solver can preserve sharp boundaries while still propagating useful information along compatible directions.

Mechanism:
For each directed edge i -> j, compute left/right states h_i and h_j, a learned wave speed s_ij, a wave amplitude a_ij, and an admissible flux F_ij. Update nodes by graph divergence: h_i <- h_i - sum_j F_ij + sum_k F_ki. Use antisymmetric or conservative parameterization where possible.

Source inspiration:
LeVeque's finite-volume treatment of conservation laws, Riemann problems, Godunov-style fluxes, upwinding, and limiters.

Why this is not just a known baseline:
The message is not attention or mean aggregation. It is an interface solver with wave speed, flux, and divergence-form update.

Closest known related architectures:
MPNNs with edge gates, PDE-inspired GNNs, continuous-depth GNNs, MeshGraphNets. Must check current PDE-GNN literature before claiming novelty.

Expected insight if it succeeds:
Class boundaries in graphs may behave like discontinuities. Directional interface solvers may reduce oversmoothing across heterophilous or noisy edges.

Expected insight if it fails:
Finite-volume structure may be too strong for citation-style node classification, or edge discontinuities may not be the right abstraction for these benchmarks.

Target task family:
Node classification, heterophily datasets, synthetic graph advection/shock tasks.

Primary baseline to beat:
gpr_gnn on Cora/PubMed; GATv2/APPNP as secondary comparisons; synthetic finite-volume baselines for toy tasks.

Minimal falsifying experiment:
A synthetic graph shock task where labels are separated by sharp graph interfaces. If RiemannEdgeGNN oversmooths like GCN/GPR or underperforms a simple MLP+edge gate, reject the mechanism.

Confirmation protocol:
Fast screen on synthetic shock + Cora/PubMed seed 0. Confirm on seeds [0,1,2] only if seed-0 validation improves over the best relevant baseline.

Complexity/runtime risk:
Moderate. Edge-wise MLPs can be expensive on large graphs.

Implementation boundary:
Start with scalar/channelwise flux and one residual step. Avoid full systems of conservation laws initially.
```

## 2. TVDLimiterGNN

```text
Scientific hypothesis:
GNN failures often come from uncontrolled feature variation: propagation either creates oscillatory features around graph boundaries or over-damps meaningful discontinuities. A graph total-variation limiter can learn when to suppress, pass, or clip messages.

Mechanism:
Compute candidate edge increments delta_ij = h_j - h_i. Estimate local graph slopes from multiple neighbors. Apply a differentiable limiter phi(r_ij) to each message, where r_ij compares incoming and outgoing increments. Aggregate limited messages only.

Source inspiration:
High-resolution finite-volume methods, slope/flux limiters, total variation, and the principle of reducing numerical oscillations near discontinuities.

Why this is not just a known baseline:
The gate depends on local variation ratios and is designed as a limiter, not arbitrary attention.

Closest known related architectures:
Attention GNNs, robust aggregation, median/trimmed GNNs, PDE-inspired limiters.

Expected insight if it succeeds:
Oversmoothing and heterophily can be reframed as a limiter problem: propagate only when local graph variation says it is safe.

Expected insight if it fails:
Graph feature variation may not correspond to meaningful discontinuities, or learned limiters may collapse to ordinary attention.

Target task family:
Heterophily node classification, noisy-edge robustness, synthetic discontinuity tasks.

Primary baseline to beat:
gpr_gnn and APPNP on node tasks; robust GCN/GAT variants if added.

Minimal falsifying experiment:
Generate a graph with two adjacent regions and noisy cross edges. If limiter messages do not reduce cross-boundary contamination compared with GPR/GAT, reject.

Confirmation protocol:
Measure validation accuracy plus a diagnostic: average limiter strength across same-label vs different-label edges.

Complexity/runtime risk:
Low to moderate.

Implementation boundary:
Use simple differentiable minmod/van-Leer-like MLP limiter. Do not implement a complex finite-volume stack.
```

## 3. CFLDepthGNN

```text
Scientific hypothesis:
Fixed propagation depth K is a poor proxy for stable graph computation. Some graph regions require small cautious steps, while others can propagate faster. A Courant-like local stability controller can improve depth/compute allocation and reveal graph stiffness.

Mechanism:
Learn nonnegative edge speeds c_ij and node capacities cap_i. Compute a local stability score rho_i = sum_j c_ij / cap_i. Use rho_i to damp residual updates or choose local substeps. Log effective steps as a diagnostic.

Source inspiration:
CFL conditions, local substeps in fluid simulation, and stability regions in numerical ODE/PDE methods.

Why this is not just a known baseline:
Depth/damping is controlled by a learned stability estimate tied to graph connectivity and edge speeds, not by a global K or attention weights.

Closest known related architectures:
Adaptive computation GNNs, Neural ODE GNNs, Jumping Knowledge, GPR-GNN.

Expected insight if it succeeds:
Graph regions have different propagation stiffness; adaptive local compute can help without adding arbitrary depth.

Expected insight if it fails:
CFL-like stability may not correlate with supervised task difficulty.

Target task family:
Node classification, long-range synthetic tasks, ogbn-products if scalable trainer is fixed.

Primary baseline to beat:
gpr_gnn for node classification; GCN/GAT/APPNP for diagnostics.

Minimal falsifying experiment:
A variable-speed graph advection task. If fixed-depth GPR matches or beats CFLDepthGNN with less compute, reject or narrow scope.

Confirmation protocol:
Report validation accuracy, effective average steps, and correlation between high rho_i and graph bottlenecks/degree.

Complexity/runtime risk:
Moderate if using actual substeps; low if using damping only.

Implementation boundary:
First version should use learned damping and optional repeated residual blocks, not a full ODE solver.
```

## 4. GhostBoundaryGNN

```text
Scientific hypothesis:
Graph nodes at boundaries, separators, low-degree regions, or train/test mask interfaces need special boundary conditions. Standard GNNs treat them as ordinary nodes, which may cause brittle extrapolation.

Mechanism:
Attach learned ghost states to boundary-like nodes. Boundary criteria can include low degree, articulation/separator membership, train/test mask frontier, or missing-feature status. Messages interact with ghost states before ordinary aggregation.

Source inspiration:
Ghost cells and boundary conditions from finite-volume methods; boundary and obstacle treatment from fluid simulation.

Why this is not just a known baseline:
The new object is a boundary-condition state, not a global virtual node or ordinary positional encoding.

Closest known related architectures:
Virtual node GNNs, boundary-aware graph models, positional encodings.

Expected insight if it succeeds:
Some GNN failures are boundary-condition failures rather than lack of expressivity.

Expected insight if it fails:
Boundary tokens may add noise on citation graphs or collapse into degree embeddings.

Target task family:
Node classification with sparse labels; graph prediction with molecular boundary/substructure tokens; synthetic boundary tasks.

Primary baseline to beat:
gpr_gnn on node tasks; GIN/GINE on graph tasks once edge_attr plumbing is fixed.

Minimal falsifying experiment:
Synthetic graph with masked boundary nodes. If ghost tokens do not improve boundary extrapolation, reject.

Confirmation protocol:
Report validation overall and boundary-node subset performance.

Complexity/runtime risk:
Low.

Implementation boundary:
Start with one ghost token per boundary node type; avoid expensive graph decomposition at first.
```

## 5. DivergenceProjectionGNN

```text
Scientific hypothesis:
Some graph tasks require conserving or balancing evidence across edges. Ordinary GNNs can create arbitrary node evidence. A projection step that enforces low graph divergence in edge messages can improve stability and interpretability.

Mechanism:
Compute provisional edge fluxes q_ij. Compute node residual/divergence b_i = sum_in q - sum_out q. Apply a small learned graph-Poisson-like correction using a few Jacobi/GPR steps, then update corrected fluxes and node states.

Source inspiration:
Incompressibility, pressure projection, control-volume constraints, and velocity-pressure correction from fluid simulation and finite-volume methods.

Why this is not just a known baseline:
It has an explicit projective correction of edge fluxes before node update.

Closest known related architectures:
Hodge GNNs, sheaf diffusion, physics-informed GNNs, flow networks.

Expected insight if it succeeds:
Projection layers can stabilize graph evidence propagation and reveal when constraints matter.

Expected insight if it fails:
Conservation constraints may be inappropriate for classification tasks unless the synthetic task requires them.

Target task family:
Synthetic flow/circulation tasks, link prediction, molecular graph tasks with edge_attr.

Primary baseline to beat:
GCN/GAT/GPR on node tasks; GIN/GINE on graph tasks; simple edge-flow MPNN synthetic baseline.

Minimal falsifying experiment:
Divergence-free routing task. If the projection does not reduce divergence error and improve prediction, reject.

Confirmation protocol:
Track both supervised validation and divergence residual.

Complexity/runtime risk:
Moderate because of inner correction iterations.

Implementation boundary:
Use 2-5 cheap graph smoothing steps as an approximate Poisson correction, not a sparse linear solve.
```

## 6. VorticityMemoryGNN

```text
Scientific hypothesis:
Cycle/ring information is not well represented by node-only smoothing. Maintaining a circulation-like edge memory can help tasks where labels depend on cycles, rings, or feedback loops.

Mechanism:
Maintain node states h_i and oriented edge memory omega_ij. Update omega through local cycle closures, triangle/chordless-cycle tokens, or approximate cycle basis messages. Node updates receive both gradient-like neighbor evidence and circulation-like memory evidence.

Source inspiration:
Vorticity in fluid simulation; Diestel cycle spaces; spectral/Hodge intuition.

Why this is not just a known baseline:
The architecture stores an oriented cycle/circulation state instead of only aggregating node embeddings.

Closest known related architectures:
Hodge-Laplacian neural networks, simplicial/cellular GNNs, ring-aware molecular GNNs.

Expected insight if it succeeds:
Cycle-space memory is useful on molecular/ring-heavy tasks and synthetic cycle/cut discrimination.

Expected insight if it fails:
Cycle features may be sufficient as static features; dynamic cycle memory may not add value.

Target task family:
MolHIV/MolPCBA after edge_attr support, synthetic cycle/cut tasks, graph classification.

Primary baseline to beat:
GIN/GINE, GCN, GAT, and any ring-feature baseline.

Minimal falsifying experiment:
Cycle/cut synthetic labels. If static cycle counts plus GIN beat VorticityMemoryGNN, demote the mechanism.

Confirmation protocol:
Graph task seed-0 screen, then seeds [0,1,2]. Track cycle-heavy subset if available.

Complexity/runtime risk:
Moderate to high if cycle enumeration is expensive.

Implementation boundary:
Start with triangles and short chordless cycles up to length 6, or use a sampled cycle basis.
```

## 7. MultigridResidualGNN

```text
Scientific hypothesis:
Long-range graph dependencies are hard because local message passing slowly communicates low-frequency error. A coarse-grid residual correction can transmit global information without dense attention.

Mechanism:
Run a fine update, compute a residual/error signal, restrict it to a coarsened graph, update on the coarse graph, then prolongate correction back to nodes. Coarsening can start with Graclus/METIS/simple pooling or learned branch sets.

Source inspiration:
Multigrid methods, residual correction, grid-transfer operators, and domain decomposition from fluid/PDE numerical methods.

Why this is not just a known baseline:
The hierarchy is used for residual correction, not only representation pooling.

Closest known related architectures:
DiffPool, Graph U-Net, hierarchical GNNs, algebraic multigrid-inspired networks.

Expected insight if it succeeds:
Oversquashing can be addressed as low-frequency residual correction.

Expected insight if it fails:
Generic graph coarsening may destroy task-relevant structure, or baseline GPR already handles the needed low frequencies.

Target task family:
Long-range node tasks, LRGB-style graph tasks, synthetic bottleneck tasks.

Primary baseline to beat:
gpr_gnn, APPNP, GPS/transformer baselines if added.

Minimal falsifying experiment:
A long-range dependency task where local GNN fails. If multigrid residual does not beat simple virtual-node/global-token baselines, reject or revise.

Confirmation protocol:
Report validation plus communication-distance diagnostics.

Complexity/runtime risk:
Moderate.

Implementation boundary:
One coarse level first. Avoid elaborate learned coarsening until the residual correction helps.
```

## 8. DualSpaceContactGNN

```text
Scientific hypothesis:
Graph topology edges and feature/geometry proximity edges encode different interaction types. Separating mesh-space and world-space messages can help graphs where important interactions are not adjacent in the given topology.

Mechanism:
Maintain two edge sets: original graph edges and learned contact edges from feature-space, coordinate-space, or embedding-space kNN. Use separate message functions and a gate that decides whether internal/topological or external/contact dynamics dominate.

Source inspiration:
MeshGraphNets' separation of mesh-space and world-space messages.

Why this is not just a known baseline:
The architecture explicitly distinguishes internal graph adjacency from external contact/similarity interactions and logs their relative contribution.

Closest known related architectures:
Graph transformers, kNN-augmented GNNs, MeshGraphNets, geometric GNNs.

Expected insight if it succeeds:
Some benchmark graphs are missing important non-edge interactions, and treating them as a separate channel is better than densifying all attention.

Expected insight if it fails:
Feature-space contact edges may be spurious or already captured by MLP/GPR.

Target task family:
Node classification, molecular graph tasks, synthetic contact tasks.

Primary baseline to beat:
gpr_gnn on node tasks; GIN/GINE on graph tasks.

Minimal falsifying experiment:
Synthetic graph where labels depend on non-topological contact edges. If contact channel does not help, reject.

Confirmation protocol:
Ablate topology-only, contact-only, and dual-space models.

Complexity/runtime risk:
Moderate; kNN/contact construction costs matter.

Implementation boundary:
Start with fixed contact edges from input features or positions, not dynamic all-pairs attention.
```

## 9. AdaptiveVirtualMeshGNN

```text
Scientific hypothesis:
Uniform graph resolution wastes compute and misses rare high-gradient structures. A learned sizing field that inserts virtual refinement tokens can improve accuracy on graphs with localized complexity.

Mechanism:
Compute graph-gradient or uncertainty scores. For high-score edges/subgraphs, insert virtual nodes or edge tokens for one or more processor steps. Merge virtual states back into endpoints before the task head.

Source inspiration:
Adaptive meshing, learned sizing fields, local remeshing, and resolution-independent dynamics from MeshGraphNets.

Why this is not just a known baseline:
The graph computation dynamically refines local resolution rather than merely pooling/coarsening existing nodes.

Closest known related architectures:
Virtual nodes, edge-token transformers, adaptive computation, MeshGraphNets.

Expected insight if it succeeds:
Graph complexity is spatially localized; adaptive virtual resolution is a useful alternative to global depth or dense attention.

Expected insight if it fails:
Virtual nodes may be unnecessary for benchmark graphs or too hard to train from supervision.

Target task family:
Synthetic high-gradient graph tasks, molecular graphs with functional groups/rings, long-range tasks.

Primary baseline to beat:
GIN/GINE for graph tasks; gpr_gnn/APPNP for node tasks.

Minimal falsifying experiment:
A synthetic graph where only a small region determines the label. If adaptive refinement does not improve compute/accuracy tradeoff, reject.

Confirmation protocol:
Report validation, number of virtual tokens, and accuracy vs compute.

Complexity/runtime risk:
Moderate to high due to dynamic graph construction.

Implementation boundary:
First version: edge tokens on selected top-k high-gradient edges, no persistent remeshing.
```

## 10. GraphFourierOperatorGNN

```text
Scientific hypothesis:
Some graph tasks are better viewed as learning an operator over graph signals rather than a fixed-depth classifier. A graph neural operator should generalize across graph resolutions/coarsenings better than ordinary message passing.

Mechanism:
Combine a local MPNN branch with a low-rank spectral/operator branch. The operator branch uses a small set of graph basis functions or randomized low-rank features and learns mode mixing. Train with optional consistency across graph coarsenings/refinements.

Source inspiration:
Fourier Neural Operator, Trefethen spectral methods, and mesh-invariant neural operator motivation.

Why this is not just a known baseline:
The claim is operator transfer across graph discretizations, not just spectral filtering on one fixed graph.

Closest known related architectures:
ChebNet, BernNet, GPR-GNN, graph neural operators, spectral GNNs.

Expected insight if it succeeds:
Resolution-transfer is a distinct graph generalization axis missing from standard node-classification benchmarks.

Expected insight if it fails:
Citation-style fixed-graph tasks may not reward operator learning; synthetic resolution tasks are necessary.

Target task family:
Synthetic resolution-transfer tasks, graph signal tasks, possibly LRGB.

Primary baseline to beat:
gpr_gnn and BernNet-style filters if present; local MPNN on synthetic operator tasks.

Minimal falsifying experiment:
Train on coarse graphs and evaluate on refined graphs. If no transfer advantage appears, reject or revise.

Confirmation protocol:
Require cross-resolution validation, not only same-graph validation.

Complexity/runtime risk:
Moderate; eigenvectors are costly, randomized bases may be needed.

Implementation boundary:
Use truncated Laplacian eigenvectors only for small graphs; provide randomized/anchor basis fallback.
```

## 11. GraphSpectralLimiterGNN

```text
Scientific hypothesis:
Global spectral propagation is powerful on smooth graph signals but harmful near discontinuities. A hybrid spectral-limiter layer can use global modes where smoothness is detected and local finite-volume-style updates near graph shocks.

Mechanism:
Estimate local graph smoothness. Use a spectral/global operator branch for smooth regions and a local limiter/flux branch for nonsmooth regions. A learned but regularized switch controls the mixture.

Source inspiration:
Spectral accuracy for smooth problems, Gibbs phenomenon around discontinuities, and finite-volume limiters.

Why this is not just a known baseline:
The architecture explicitly couples a smoothness detector to the choice of spectral vs limited local propagation.

Closest known related architectures:
Spectral GNNs, GPR-GNN, BernNet, attention gates, wavelet GNNs.

Expected insight if it succeeds:
The right propagation operator depends on graph signal regularity.

Expected insight if it fails:
Smoothness diagnostics may not be reliable from node features or may be learned implicitly by GPR.

Target task family:
Node classification with heterophily, synthetic smooth/discontinuous graph signals.

Primary baseline to beat:
gpr_gnn, APPNP, GATv2.

Minimal falsifying experiment:
Mixed synthetic benchmark with smooth regions and discontinuities. If hybrid switch does not outperform either branch alone, reject.

Confirmation protocol:
Ablate spectral-only, limiter-only, and hybrid.

Complexity/runtime risk:
Low to moderate.

Implementation boundary:
Use GPR-like polynomial branch for spectral/global part; do not require expensive eigendecomposition initially.
```

## 12. PseudospectralGuardGNN

```text
Scientific hypothesis:
Repeated graph propagation can be unstable even when eigenvalues appear safe, especially on directed or non-normal operators. A cheap pseudospectral/non-normality guard can prevent hidden-state amplification and improve deep GNN reliability.

Mechanism:
Parameterize propagation with a learned operator plus a diagnostic estimate of transient amplification, such as norms of repeated random probes or commutator-like asymmetry. Damp updates when estimated amplification exceeds a threshold.

Source inspiration:
Pseudospectra, stability regions, and numerical stability analysis in Trefethen's materials.

Why this is not just a known baseline:
It targets non-normal transient amplification, not just spectral radius or dropout.

Closest known related architectures:
Stability-constrained GNNs, deep residual GNNs, directed spectral GNNs.

Expected insight if it succeeds:
Some deep GNN failures are non-normal transient-growth failures.

Expected insight if it fails:
Non-normal amplification may not be a dominant issue in the current benchmark graphs.

Target task family:
Directed graphs, link prediction, synthetic non-normal propagation tasks.

Primary baseline to beat:
Deep GCN/APPNP/GPR variants and directed GNN baselines if available.

Minimal falsifying experiment:
Synthetic directed non-normal graph where unconstrained propagation explodes. If guard does not stabilize and improve accuracy, reject.

Confirmation protocol:
Report validation and amplification diagnostics over depth.

Complexity/runtime risk:
Low to moderate depending on diagnostic probes.

Implementation boundary:
Start with 1-3 random probe vectors per batch; no full pseudospectrum computation.
```

## 13. EventAdaptiveGraphODE

```text
Scientific hypothesis:
Different graphs and graph regions need different effective propagation depths. An adaptive ODE solver can expose this as number of function evaluations and improve compute allocation.

Mechanism:
Define dh/dt = f_theta(h, graph, t). Use an ODE solver or a lightweight adaptive Euler/Runge-Kutta controller. Let local residual/error estimates choose additional steps. Save solver work diagnostics.

Source inspiration:
Neural ODEs, adaptive numerical ODE solvers, and stability/precision tradeoffs.

Why this is not just a known baseline:
The scientific signal is adaptive solver work and stiffness on graph regions, not merely continuous depth.

Closest known related architectures:
Continuous-depth GNNs, GRAND, ODE-GNNs.

Expected insight if it succeeds:
Solver work identifies hard graph regions and improves performance/compute tradeoff.

Expected insight if it fails:
Adaptive continuous depth may be overkill; fixed GPR-style propagation may suffice.

Target task family:
Node classification, long-range synthetic tasks, variable-depth tasks.

Primary baseline to beat:
gpr_gnn and APPNP.

Minimal falsifying experiment:
A variable-depth synthetic dependency task. If fixed-depth GPR beats adaptive ODE at equal compute, reject.

Confirmation protocol:
Report validation, number of function evaluations, and error-estimate histograms.

Complexity/runtime risk:
High if using torchdiffeq; moderate with custom adaptive residual stepping.

Implementation boundary:
Begin with custom adaptive residual steps to avoid heavy dependencies.
```

## 14. GraphDEQResidualNet

```text
Scientific hypothesis:
Some graph predictions correspond to an equilibrium of iterative evidence propagation. Directly solving for a fixed point can give deeper reasoning without choosing a finite depth, while residual size provides uncertainty.

Mechanism:
Define z = f_theta(z, x, edge_index). Solve z* with Broyden or Anderson-like iterations. Add contraction/residual regularization. Use final residual norm as an auxiliary uncertainty feature or diagnostic.

Source inspiration:
Deep Equilibrium Models: root finding, implicit differentiation, fixed-point modeling, and constant-memory effective depth.

Why this is not just a known baseline:
The architecture treats graph propagation as an implicit equilibrium and uses residual/contraction diagnostics as part of the scientific hypothesis.

Closest known related architectures:
Implicit GNNs, DEQ-GNNs, recurrent GNNs.

Expected insight if it succeeds:
Equilibrium residuals can predict confidence or OOD graph regions, and implicit depth can outperform finite propagation.

Expected insight if it fails:
Implicit propagation may converge to oversmoothed equilibria or be unstable on benchmark graphs.

Target task family:
Node classification, link prediction, synthetic equilibrium tasks.

Primary baseline to beat:
gpr_gnn, APPNP, GCNII if present.

Minimal falsifying experiment:
A synthetic equilibrium-label task. If DEQ does not beat a finite unrolled recurrent GNN with similar compute, reject.

Confirmation protocol:
Report validation, convergence rate, residual norm distributions, and failed-solve rate.

Complexity/runtime risk:
High due to root finding and backward pass.

Implementation boundary:
Start with forward fixed-point iterations and truncated backward if necessary. Do not over-engineer implicit differentiation in the first prototype.
```

## 15. HybridExplicitImplicitFluxGNN

```text
Scientific hypothesis:
Local graph discontinuities need explicit flux-limited updates, while long-range low-frequency errors need implicit/global correction. A hybrid explicit-implicit solver can combine the strengths of finite-volume and DEQ/multigrid thinking.

Mechanism:
Apply one or more explicit Riemann/limiter edge-flux steps, compute a residual, then apply an implicit correction layer solved approximately as a fixed point or coarse-grid correction.

Source inspiration:
Explicit/implicit numerical solvers, finite-volume flux updates, DEQ fixed-point correction, and multigrid residual correction.

Why this is not just a known baseline:
It has a solver split: explicit local interface dynamics plus implicit global residual correction.

Closest known related architectures:
Residual GNNs, implicit GNNs, multigrid GNNs, PDE-GNNs.

Expected insight if it succeeds:
GNN propagation benefits from a numerical-solver decomposition rather than a homogeneous stack of layers.

Expected insight if it fails:
The split may be too complex for current data sizes and overfit or destabilize training.

Target task family:
Synthetic shock + long-range tasks first; then Cora/PubMed.

Primary baseline to beat:
RiemannEdgeGNN, MultigridResidualGNN, gpr_gnn.

Minimal falsifying experiment:
A combined discontinuity/long-range synthetic task. If either component alone beats the hybrid, reject or simplify.

Confirmation protocol:
Ablate explicit-only, implicit-only, and hybrid.

Complexity/runtime risk:
High.

Implementation boundary:
Wave 2 only. Do not implement before RiemannEdgeGNN and MultigridResidualGNN have individual evidence.
```

## 16. SeparatorDomainGNN

```text
Scientific hypothesis:
Oversquashing often occurs at graph separators. Treating separators as interfaces between domains, with special boundary states, can improve long-range evidence transfer and expose when graph structure bottlenecks matter.

Mechanism:
Detect bridges, articulation points, or approximate separators. Run local message passing inside components and pass compressed interface messages through separator tokens. Optional ghost boundary states model missing external context.

Source inspiration:
Diestel's structural graph theory, separators, connectivity, blocks, and tree-decomposition intuition; finite-volume domain decomposition and boundary conditions.

Why this is not just a known baseline:
The architecture explicitly decomposes the graph into domains and separator interfaces, rather than adding a single virtual node or global attention.

Closest known related architectures:
Cluster-GCN, hierarchical GNNs, subgraph GNNs, separator-based algorithms.

Expected insight if it succeeds:
Graph bottlenecks need interface-specific computation, not merely more hops.

Expected insight if it fails:
Separator preprocessing may be too crude or labels may not depend on bottleneck communication.

Target task family:
Long-range synthetic tasks, node classification, graph classification with modular graphs.

Primary baseline to beat:
gpr_gnn, APPNP, virtual-node GNN, hierarchical pooling baseline.

Minimal falsifying experiment:
Synthetic barbell/separator task. If separator tokens do not improve over virtual node/GPR, reject.

Confirmation protocol:
Report validation and separator-node accuracy/attention diagnostics.

Complexity/runtime risk:
Moderate.

Implementation boundary:
Start with bridges/articulation points and simple component decomposition.
```

## 17. BranchSetMinorGNN

```text
Scientific hypothesis:
Useful graph abstractions often arise from contracting connected branch sets. A GNN that learns minor-like coarsenings can preserve structural signal better than arbitrary pooling.

Mechanism:
Learn connected branch sets using local assignment constrained by adjacency. Contract each branch set to a supernode, propagate on the minor graph, and lift corrections back to original nodes.

Source inspiration:
Graph minors and branch-set thinking from Diestel; multigrid/coarse-grid correction from numerical PDEs.

Why this is not just a known baseline:
Pooling units must be connected branch sets and are used as minor contractions, not arbitrary clusters.

Closest known related architectures:
DiffPool, MinCutPool, hierarchical GNNs, graph coarsening models.

Expected insight if it succeeds:
Minor-like contractions are a useful inductive bias for graph abstraction and long-range reasoning.

Expected insight if it fails:
Learned connected pooling may be too hard or current benchmarks may not reward minor stability.

Target task family:
Graph classification, long-range graph tasks, synthetic minor-invariance tasks.

Primary baseline to beat:
GIN/GINE, DiffPool/Graph U-Net if added, gpr_gnn for node variants.

Minimal falsifying experiment:
Minor-invariance synthetic task: labels unchanged under contraction of irrelevant branches. If BranchSetMinorGNN does not transfer across contractions, reject.

Confirmation protocol:
Train/test across graph contractions and expansions.

Complexity/runtime risk:
High if connected assignments are hard.

Implementation boundary:
Prototype with deterministic connected clusters first; add learned branch-set assignment later.
```

## 18. ManufacturedSolutionProbeGNN

```text
Scientific hypothesis:
A long-running architecture gym needs diagnostic tasks with known solutions, analogous to manufactured solutions in numerical PDE verification. These probes can reveal whether a candidate has learned the intended mechanism before real benchmarks.

Mechanism:
This is not a standalone model. It is a test harness: generate graph PDE/process tasks with known target functions, such as graph advection, diffusion, divergence-free flows, cycle circulation, and boundary conditions. Use them as pre-benchmark falsifying experiments.

Source inspiration:
Verification/validation and manufactured solutions from finite-element/finite-volume methodology.

Why this is not just a known baseline:
It is infrastructure for scientific falsification, not a model.

Closest known related architectures:
Synthetic graph benchmark generation, algorithmic reasoning benchmarks.

Expected insight if it succeeds:
The gym can detect whether an architecture mechanism works before noisy real-data comparisons.

Expected insight if it fails:
The synthetic tasks may be too artificial or not predictive of real benchmark performance.

Target task family:
All source-inspired architectures.

Primary baseline to beat:
Not applicable; use to screen mechanisms.

Minimal falsifying experiment:
For each architecture, define one source-aligned manufactured task. If the architecture fails its own task, do not promote it to real-data runs.

Confirmation protocol:
Use as mandatory precondition for Wave 2 architecture claims.

Complexity/runtime risk:
Low to moderate.

Implementation boundary:
Add datasets/tests, not a model file.
```

---

# Wave 2: more ambitious ideas after source-aligned synthetic tasks exist

## 19. EntropyAdmissibleGNN

Finite-volume methods for nonlinear conservation laws distinguish weak solutions from admissible ones using entropy criteria. Translate this into graph learning by making propagation choose among multiple possible message states and penalizing updates that increase a learned graph entropy unless supported by supervised evidence. Use on heterophily and noisy-edge tasks. This should wait until `RiemannEdgeGNN` and `TVDLimiterGNN` establish whether finite-volume metaphors help.

## 20. GraphNormalizingFlowODE

Neural ODEs include continuous normalizing flows. A graph version could model invertible transformations of node embeddings under graph-conditioned dynamics. Use likelihood or reconstruction as an auxiliary objective, then evaluate whether invertibility prevents overcompression/oversmoothing. Prior-art risk is high; treat as Wave 2.

## 21. BoundaryCorrectedGraphOperator

Combine `GraphFourierOperatorGNN` with `GhostBoundaryGNN`: a global low-rank operator handles smooth interior graph signals, while ghost-boundary corrections handle boundary/separator conditions. This is meant to test whether boundary errors are the main weakness of graph neural operators on irregular graphs.

## 22. ResidualAsUncertaintyDEQGNN

A DEQ-style GNN where the fixed-point residual is trained/calibrated as an uncertainty signal. Evaluate whether high residual identifies mislabeled, OOD, boundary, or heterophilous nodes. This requires careful calibration metrics, not only accuracy.

## 23. OperatorRemeshGNN

Unify neural operators and adaptive mesh ideas: learn an operator whose computation graph changes resolution through virtual refinement, but whose output is constrained to be consistent under graph refinement/coarsening. This is ambitious and should only be attempted after `AdaptiveVirtualMeshGNN` and `GraphFourierOperatorGNN` have separate evidence.

---

# Implementation priority

Recommended first five implementations:

```text
1. TVDLimiterGNN
2. RiemannEdgeGNN
3. DivergenceProjectionGNN
4. SeparatorDomainGNN
5. GraphSpectralLimiterGNN
```

Why these first:

- They are source-grounded.
- They are implementable without a full new trainer.
- They test different scientific hypotheses.
- They can each have a clear synthetic falsifying task.
- They are less likely to be trivial variants of GPR-GNN than another hop-gating architecture.

Recommended infrastructure before Wave 2:

```text
1. Add source-aligned synthetic tasks.
2. Add edge_attr plumbing for graph prediction.
3. Add config-level aggregation with architecture_config_hash.
4. Add diagnostics logging: limiter strength, divergence residual, solver residual, effective depth, separator interface usage.
```

# Diagnostics table

| Architecture | Required diagnostic |
|---|---|
| RiemannEdgeGNN | edge speeds, flux norms, cross-boundary flux |
| TVDLimiterGNN | limiter values by same-label vs different-label edges |
| CFLDepthGNN | effective step size/depth by degree and bottleneck |
| GhostBoundaryGNN | boundary-node performance and ghost-message norms |
| DivergenceProjectionGNN | divergence residual before/after projection |
| VorticityMemoryGNN | cycle-memory norms and cycle-heavy subset accuracy |
| MultigridResidualGNN | fine residual before/after coarse correction |
| DualSpaceContactGNN | topology vs contact channel contribution |
| AdaptiveVirtualMeshGNN | virtual token count and high-gradient coverage |
| GraphFourierOperatorGNN | cross-resolution validation and mode weights |
| GraphSpectralLimiterGNN | smoothness switch statistics |
| PseudospectralGuardGNN | transient amplification estimate |
| EventAdaptiveGraphODE | number of function evaluations |
| GraphDEQResidualNet | fixed-point residual and failed-solve rate |
| SeparatorDomainGNN | separator interface usage and bottleneck accuracy |
| BranchSetMinorGNN | contraction consistency |

# Synthetic benchmark backlog

```text
synthetic_graph_advection
synthetic_graph_shock_boundary
synthetic_variable_speed_cfl
synthetic_divergence_free_flow
synthetic_cycle_cut_discrimination
synthetic_barbell_separator
synthetic_minor_invariance
synthetic_resolution_transfer
synthetic_contact_edges
synthetic_non_normal_stability
synthetic_fixed_point_equilibrium
synthetic_boundary_condition_transfer
```

# Promotion rule

A source-inspired architecture should be promoted to `research/INSIGHTS.md` only if at least one of the following is true:

1. It beats the relevant baseline under confirmed config-level aggregation.
2. It fails on real benchmarks but succeeds on its source-aligned synthetic task, giving a clear scope condition.
3. It fails both real and synthetic tasks, but the failure falsifies a precise hypothesis and prevents future repeated attempts.

Do not promote an idea merely because it has an interesting name or one lucky seed.
