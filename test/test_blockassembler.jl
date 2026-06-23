using BEAST
using CompScienceMeshes
using LinearAlgebra
using Test

r = 10.0
λ = 20 * r
k = 2 * π / λ

sphere = readmesh(joinpath(dirname(@__FILE__),"assets","sphere5.in"), T=Float64)

D = Maxwell3D.doublelayer(wavenumber=k)
X = raviartthomas(sphere)
Y = buffachristiansen(sphere)

A = assemble(D, X, X)

@views blkasm = BEAST.blockassembler(D, X, X)

@views function assembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end

A_blk = zeros(ComplexF64, length(X.fns), length(Y.fns))
assembler(A_blk, [1:length(X.fns);], [1:length(Y.fns);])

@test norm(A - A_blk) ≈ 0 atol=eps(Float64)

n = length(X.fns)
allids = [1:n;]

Arows = zeros(ComplexF64, n, n)
for t in 1:n
    blkasm([t], allids, (v,m,o) -> (Arows[t,o] += v))
end
@test norm(A - Arows) ≈ 0 atol=eps(Float64)

Acols = zeros(ComplexF64, n, n)
for s in 1:n
    blkasm(allids, [s], (v,m,o) -> (Acols[m,s] += v))
end
@test norm(A - Acols) ≈ 0 atol=eps(Float64)

Apar = zeros(ComplexF64, n, n)
Threads.@threads for t in 1:n
    blkasm([t], allids, (v,m,o) -> (Apar[t,o] += v))
end
@test norm(A - Apar) ≈ 0 atol=eps(Float64)
