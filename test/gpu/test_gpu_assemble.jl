# Full-matrix GPU assembly. Ground truth is the CPU assembly of the same
# operator/space/quadrature, which the rest of the BEAST test suite validates.

@testitem "GPU full assembly vs CPU" tags = [:gpu] begin
    using CUDA
    using CompScienceMeshes
    using LinearAlgebra
    using Test

    CUDA.device!(parse(Int, get(ENV, "BEAST_GPU_DEVICE", "0")))
    qs = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)

    cuboid = meshcuboid(1.0, 1.0, 1.0, 0.5)
    sphere = readmesh(joinpath(@__DIR__, "..", "assets", "sphere5.in"); T=Float64)

    configs = [
        ("Helmholtz3D single layer / Lagrange", Helmholtz3D.singlelayer(wavenumber=1.0), lagrangec0(cuboid; order=1)),
        ("Maxwell3D single layer / RaviartThomas", Maxwell3D.singlelayer(wavenumber=1.0), raviartthomas(sphere)),
        ("Maxwell3D single layer / BuffaChristiansen", Maxwell3D.singlelayer(wavenumber=1.0), buffachristiansen(sphere)),
    ]

    for (name, op, X) in configs
        @testset "$name" begin
            cpu = assemble(op, X, X; threading=:single, quadstrat=qs)
            gpu = assemble(op, X, X; threading=:gpu, quadstrat=qs)
            @test isapprox(Array(gpu), cpu; atol=1e-10, rtol=1e-10)
        end
    end
end

@testitem "GPU full assembly with tiling" tags = [:gpu] begin
    using CUDA
    using CompScienceMeshes
    using LinearAlgebra
    using Test

    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    CUDA.device!(parse(Int, get(ENV, "BEAST_GPU_DEVICE", "0")))

    op = Helmholtz3D.singlelayer(wavenumber=1.0)
    X = lagrangec0(meshcuboid(1.0, 1.0, 1.0, 0.5); order=1)
    qs = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)
    tiling = ext.TilingStrategy(ext.WorksizeTiling(8), ext.WorksizeTiling(8))

    cpu = assemble(op, X, X; threading=:single, quadstrat=qs)
    gpu = assemble(op, X, X; threading=:gpu, quadstrat=qs, tilingstrat=tiling)
    @test isapprox(Array(gpu), cpu; atol=1e-10, rtol=1e-10)
end

@testitem "GPU assembly with outer_rule != inner_rule" tags = [:gpu] begin
    using CUDA
    using CompScienceMeshes
    using LinearAlgebra
    using Test

    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    CUDA.device!(parse(Int, get(ENV, "BEAST_GPU_DEVICE", "0")))

    op = Helmholtz3D.singlelayer(wavenumber=1.0)
    X = lagrangec0(meshcuboid(1.0, 1.0, 1.0, 0.5); order=1)

    # the GPU double-num kernels integrate test and trial with their own rule,
    # so asymmetric outer/inner rules must still match the CPU assembly
    for (outer, inner) in ((4, 2), (2, 4), (5, 3))
        @testset "outer=$outer inner=$inner" begin
            qs = BEAST.DoubleNumSauterQstrat(outer, inner, 2, 2, 2, 2)
            cpu = assemble(op, X, X; threading=:single, quadstrat=qs)

            gpu = assemble(op, X, X; threading=:gpu, quadstrat=qs)
            @test isapprox(Array(gpu), cpu; atol=1e-10, rtol=1e-10)

            asm = ext.gpu_blockassembler(op, X, X; quadstrat=qs)
            tids = collect(1:min(6, numfunctions(X)))
            sids = collect(max(1, numfunctions(X) - 5):numfunctions(X))
            blk = CUDA.zeros(scalartype(op, X, X), length(tids), length(sids))
            asm(blk, tids, sids)
            @test isapprox(Array(blk), cpu[tids, sids]; atol=1e-10, rtol=1e-10)
        end
    end
end
