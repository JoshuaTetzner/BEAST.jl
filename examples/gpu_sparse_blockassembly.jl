# Batched GPU assembly of the near interactions of a fast method (H-matrix / ACA)
# with the sparse block assembler.
#
# Instead of looping a per-block assembler over `values[i]` / `nearvalues[i]`,
# all near-interaction blocks are assembled in one batched pass and returned as a
# sparse matrix `Z` (single device) or as a vector of partial sparse matrices,
# one per device (multi device). This is unrelated to the radiated near field.
# Load CUDA before using the extension.

using CUDA
using BEAST
using CompScienceMeshes
using LinearAlgebra
using ParallelKMeans, H2Trees, AdaptiveCrossApproximation

const ext = Base.get_extension(BEAST, :BEASTCUDAExt)
@assert ext !== nothing "Load CUDA in an environment where it is available."

k = 2π
Γ = meshsphere(1.0, 0.1)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
qstrat = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)  # Sauter-capable: required

# near-interaction block list from the cluster tree
tree = KMeansTree(space.pos, 2; minvalues=100)
values, nearvalues = AdaptiveCrossApproximation.nearinteractions(BlockTree(tree, tree))

# ---- single device ---------------------------------------------------------
# Returns one CuSparseMatrixCSR; block (i) is Z[values[i], nearvalues[i]].
gpuasm = ext.gpu_blockassembler(op, space, space; quadstrat=qstrat, device=0)
Z = ext.gpu_sparse_blockassemble(gpuasm, values, nearvalues)
println("Z: ", size(Z), "  nnz = ", nnz(Z))

# `maxchunkpairs` streams the element-pair work list to bound peak memory
# (default keeps the intermediate ~1 GiB); smaller = leaner GPU footprint.
Z = ext.gpu_sparse_blockassemble(gpuasm, values, nearvalues; maxchunkpairs=2_000_000)

# ---- multiple devices ------------------------------------------------------
# Build one functor per GPU, then assemble: the deduplicated element-pair list
# is split disjointly across devices, so the full result is the exact SUM of the
# returned per-device partials. Start Julia with enough threads, e.g. `-t 8`.
devices = [0, 1, 2, 3]
functors = ext.gpu_sparse_blockassemble_functors(op, space, space; quadstrat=qstrat, devices=devices)
partials = ext.gpu_sparse_blockassemble(functors, values, nearvalues)

# apply additively (never materialise the global matrix), e.g. a matvec:
x = rand(scalartype(op), numfunctions(space))
y = zeros(scalartype(op), numfunctions(space))
for (f, P) in zip(functors, partials)
    CUDA.device!(f.device)
    y .+= Array(P * CuArray(x))
end
println("‖Z x‖ = ", norm(y))
