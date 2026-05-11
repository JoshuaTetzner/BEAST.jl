using CompScienceMeshes
using BEAST

using LinearAlgebra
# using SparseArrays
# using Profile

using CUDA


Γ = meshcuboid(1.0,1.0,1.0,1.0)
Γ = meshsphere(1.0,0.5;generator=:gmsh)


X = raviartthomas(Γ)
Y = buffachristiansen(Γ)
Z = lagrangec0d2(Γ)
W = lagrangecxd0(Γ)
L = lagrangec0(Γ,order=2)
G = BEAST.gwpdiv(Γ; order=3)



@show numcells(Γ)
@show numcells(geometry(Y))

@show numfunctions(X)
@show numfunctions(Y)
@show numfunctions(Z)
@show numfunctions(G)

κ, η = 1.0, 1.0
T = Maxwell3D.singlelayer(wavenumber=κ)

V = Helmholtz3D.singlelayer(wavenumber=κ)


qstrat = BEAST.DoubleNumSauterQstrat(4, 4, 6, 6, 6, 6)


CUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)

#gpu_tstrat = CUDAExt.TilingStrategy(CUDAExt.EqualTiling(5), CUDAExt.EqualTiling(5))
gpu_tstrat = CUDAExt.TilingStrategy(CUDAExt.WorksizeTiling(4096), CUDAExt.WorksizeTiling(4096))


CUDA.@time Th_gpu = assemble(V,L,L;threading=:gpu,tilingstrat=gpu_tstrat,quadstrat=qstrat)


cpu_tstrat = CUDAExt.TilingStrategy(CUDAExt.WorksizeTiling(128), CUDAExt.WorksizeTiling(128))

@time Th_cpu2 = assemble(V,L,L;threading=:cellsplitting,tilingstrat=cpu_tstrat,quadstrat=qstrat)

@time Th_cpu3 = assemble(V,L,L;threading=:dofsplitting,quadstrat=qstrat)

@time Th_cpu = assemble(V,L,L;threading=:cellcoloring,quadstrat=qstrat)

@time Th_cpu = assemble(V,Z,Z;threading=:cellcoloring,quadstrat=qstrat)

@show Threads.nthreads()
@show eps(real(eltype(Th_cpu))) maximum(norm.(Th_gpu-Th_cpu)) 

