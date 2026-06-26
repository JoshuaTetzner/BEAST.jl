using CUDA
using BEAST
using CompScienceMeshes
using LinearAlgebra
using SparseArrays
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using OhMyThreads

λ = 1.0
k = 2 * pi / λ
Γ = meshicosphere(40, 1.0)

op = Helmholtz3D.singlelayer(; wavenumber=k)
space = lagrangec0d1(Γ)
println("NumRT: $(length(space))")
qstrat = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)

ext = Base.get_extension(BEAST, :BEASTCUDAExt)

device_id = begin
    value = get(ENV, "BEAST_GPU_DEVICE", "")
    isempty(value) ? 0 : parse(Int, value)
end
CUDA.device!(device_id)

qstrat = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)
tiling = ext.TilingStrategy(ext.WorksizeTiling(5000), ext.WorksizeTiling(5000))
##
@time gpu = assemble(op, space, space; threading=:gpu, tilingstrat=tiling, quadstrat=qstrat);

##
@time cpu = assemble(op, space, space; threading=:cellcoloring, quadstrat=qstrat);

norm(gpu - cpu) / norm(cpu)
##
λ = 1.0
k = 2 * pi / λ
Γ = meshicosphere(40, 1.0)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
println("NumRT: $(length(space))")
tree = KMeansTree(space.pos, 2; minvalues=100)
blktree = BlockTree(tree, tree)

values, nearvalues = AdaptiveCrossApproximation.nearinteractions(blktree)

blkassembler = blockassembler(op, space, space; quadstrat=qstrat);
gpublkassembler = ext.gpu_blockassembler(op, space, space; quadstrat=qstrat);
##
values
nearvalues
##
println("near blocks  = ", length(values))

# element-pair mask: which (test_el, trial_el) integrals the near-field needs
mask = ext.nearblock_element_mask(space, values, space, nearvalues)
println("element mask = ", size(mask, 1), " x ", size(mask, 2),
    ", nnz = ", nnz(mask),
    " (", round(100 * nnz(mask) / length(mask); digits=4), "% dense)")

# --- CPU block assembly (reference) ---------------------------------------
cpu_blks = [zeros(scalartype(op), length(values[i]), length(nearvalues[i]))
            for i in eachindex(values)]

@time for i in eachindex(values)
    blk = cpu_blks[i]
    store(v, m, n) = (@inbounds blk[m, n] += v)
    blkassembler(values[i], nearvalues[i], store)
end

##
# --- GPU block assembly (device-allocated blocks) -------------------------
gpu_blks = [CUDA.zeros(scalartype(op), length(values[i]), length(nearvalues[i]))
            for i in eachindex(values)]

# warmup: first call compiles the kernels
gpublkassembler(gpu_blks[1], values[1], nearvalues[1])
CUDA.synchronize()

CUDA.@time begin
    for i in eachindex(values)
        gpublkassembler(gpu_blks[i], values[i], nearvalues[i])
    end
    CUDA.synchronize()
end

##
# --- correctness: GPU vs CPU ----------------------------------------------
maxrel = 0.0
for i in eachindex(values)
    gpu_blk = Array(gpu_blks[i])
    denom = norm(cpu_blks[i])
    rel = iszero(denom) ? norm(gpu_blk - cpu_blks[i]) : norm(gpu_blk - cpu_blks[i]) / denom
    global maxrel = max(maxrel, rel)
end
println("max relative block error (GPU vs CPU) = ", maxrel)
