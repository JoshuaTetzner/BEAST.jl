using CompScienceMeshes
using BEAST

using Test

m = meshrectangle(1.0, 1.0, 0.5, 3)
nodes = skeleton(m,0)
srt_bnd_nodes = sort.(skeleton(boundary(m),0))
int_nodes = submesh(nodes) do node
    !(sort(node) in srt_bnd_nodes)
end

@test length(int_nodes) == 1

L0 = BEAST.lagrangec0d1(m, int_nodes)
@test numfunctions(L0) == 1
@test length(L0.fns[1]) == 6
@test length(m) == 8

Lx = BEAST.lagrangecxd0(m)
@test numfunctions(Lx) == 8

Id = BEAST.Identity()
Q1, st1 = BEAST.allocatestorage(Id, L0, Lx, Val{:bandedstorage}, BEAST.LongDelays{:ignore})
BEAST.assemble_local_matched!(Id, L0, Lx, st1)

Q2, st2 = BEAST.allocatestorage(Id, L0, Lx, Val{:bandedstorage}, BEAST.LongDelays{:ignore})
BEAST.assemble_local_mixed!(Id, L0, Lx, st2)

@test isapprox(Q1, Q2, atol=1e-8)
