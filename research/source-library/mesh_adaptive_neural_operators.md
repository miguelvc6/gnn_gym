# GNN-GYM source-library note

This file treats the source as an idea generator, not as a recipe book. Candidate architectures below are research hypotheses for GNN-GYM. They are not claims of literature novelty until a separate prior-art check is done.

Each architecture should be evaluated with the GNN-GYM long-running protocol: toy crash check, seed-0 screen, confirmation on seeds `[0, 1, 2]`, validation-metric selection only, test metric for reporting only, and aggregation by `architecture_config_hash`.

# Source family: mesh adaptivity and neural operators

## Why this family may produce original GNN architectures

MeshGraphNets and Fourier Neural Operators point to two architecture principles that standard GNN benchmarks rarely test: resolution independence and operator learning. Instead of learning a classifier tied to one graph, an architecture can learn a graph-indexed operator intended to transfer across graph resolutions, coarsenings, perturbations, and discretizations.

## Core uploaded sources

- `2010.03409v4.pdf` (MeshGraphNets)
- `2010.08895v3.pdf` (Fourier Neural Operator)
- `Trefethenespectral.pdf`

## Concepts to mine

- Mesh-space and world-space message passing.
- Adaptive mesh resolution and learned sizing fields.
- Local remeshing.
- Resolution-independent dynamics.
- Neural operators between function spaces.
- Fourier-space kernels and mode truncation.
- Zero-shot super-resolution.
- Spectral filtering and global-local operator splits.

## Known GNN/ML prior art to avoid merely reimplementing

- Standard graph transformers with structural encodings.
- MeshGraphNets reimplementation.
- Fourier Neural Operator reimplementation on grid data.
- Positional-encoding GNNs that use Laplacian eigenvectors but do not learn operator transfer.

## Candidate mechanisms

- Learn a graph sizing field that chooses where to add virtual refinement tokens.
- Split edges into topology-space edges and learned world/similarity-space contact edges.
- Train the same operator on coarse and fine versions of synthetic graph tasks.
- Low-rank spectral operator with local boundary correction.
- Resolution-transfer loss: predictions should be consistent after graph refinement/coarsening.

## Synthetic tasks this source suggests

1. **Graph super-resolution:** train on coarse graphs, evaluate on refined graphs.
2. **Resolution-invariant diffusion/advection:** same underlying continuum graph sampled at different densities.
3. **Adaptive refinement task:** labels depend on rare high-gradient regions, not uniform graph neighborhoods.
4. **Contact-edge task:** distant nodes in graph topology interact through feature/coordinate proximity.
5. **Operator-transfer task:** learn a map from input graph signal to output signal across graph sizes.

## Architecture idea stubs

- `AdaptiveVirtualMeshGNN`: insert virtual refinement nodes where learned graph gradients are high, then pool them away before task output.
- `DualSpaceContactGNN`: pass messages on both original graph edges and learned contact/similarity edges.
- `GraphFourierOperatorGNN`: combine local MPNN updates with low-rank graph spectral operator modes.
- `ResolutionConsistencyGNN`: architecture plus loss enforcing consistency across graph coarsenings/refinements.
- `SizingFieldPoolingGNN`: pooling guided by a learned sizing field rather than generic top-k scores.

## Rejected or low-novelty translations

- Add Laplacian eigenvectors to GCN and call it a neural operator.
- Add kNN feature edges and call it world-space simulation without a contact/interior distinction.
- Train only on one fixed graph and claim resolution independence.
