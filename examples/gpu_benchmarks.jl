using CUDA
using BEAST
using CompScienceMeshes
using LinearAlgebra
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

blkassembler = blockassembler(op, space, space);
gpublkassembler = ext.gpu_blockassembler(op, space, space; quadstrat=qstrat);
##
values
nearvalues
##
@time begin
    blks = [zeros(
        scalartype(op),
        length(values[i]),
        length(nearvalues[i])
    ) for i in eachindex(values)]

    for i in eachindex(values)
        blkassembler(blks[i], values[i], nearvalues[i])
    end
end
