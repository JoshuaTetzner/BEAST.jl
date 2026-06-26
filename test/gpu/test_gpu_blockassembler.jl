# Per-block GPU assembler (`gpu_blockassembler`). The functor writes a dense
# block into a device matrix; ground truth is the matching slice of the CPU
# assembly.

@testitem "GPU block functor vs CPU assembly slice" tags = [:gpu] begin
    using CUDA
    using CompScienceMeshes
    using LinearAlgebra
    using Test

    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
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
            ref = assemble(op, X, X; threading=:single, quadstrat=qs)
            asm = ext.gpu_blockassembler(op, X, X; quadstrat=qs)
            tids = collect(1:min(8, numfunctions(X)))
            sids = collect(max(1, numfunctions(X) - 7):numfunctions(X))
            blk = CUDA.zeros(scalartype(op, X, X), length(tids), length(sids))
            @test asm(blk, tids, sids) === blk
            @test isapprox(Array(blk), ref[tids, sids]; atol=1e-10, rtol=1e-10)
        end
    end
end

@testitem "GPU block functor: subset space, far block, view destination" tags = [:gpu] begin
    using CUDA
    using CompScienceMeshes
    using LinearAlgebra
    using Test

    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    CUDA.device!(parse(Int, get(ENV, "BEAST_GPU_DEVICE", "0")))
    op = Helmholtz3D.singlelayer(wavenumber=1.0)
    cuboid = meshcuboid(1.0, 1.0, 1.0, 0.5)

    @testset "subset space" begin
        qs = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)
        Xfull = lagrangec0(cuboid; order=1)
        X = subset(Xfull, collect(1:2:numfunctions(Xfull)))
        ref = assemble(op, X, X; threading=:single, quadstrat=qs)
        asm = ext.gpu_blockassembler(op, X, X; quadstrat=qs)
        ids = collect(1:numfunctions(X))
        blk = CUDA.zeros(scalartype(op, X, X), length(ids), length(ids))
        asm(blk, ids, ids)
        @test isapprox(Array(blk), ref; atol=1e-10, rtol=1e-10)
    end

    @testset "far block (DoubleNumQStrat)" begin
        qfar = BEAST.DoubleNumQStrat(2, 2)
        X = lagrangec0(cuboid; order=1)
        Y = lagrangec0(CompScienceMeshes.translate(cuboid, point(5.0, 0.0, 0.0)); order=1)
        ref = assemble(op, X, Y; threading=:single, quadstrat=qfar)
        asm = ext.gpu_blockassembler(op, X, Y; quadstrat=qfar)
        tids = collect(1:4); sids = collect(1:4)
        blk = CUDA.zeros(scalartype(op, X, Y), length(tids), length(sids))
        asm(blk, tids, sids)
        @test isapprox(Array(blk), ref[tids, sids]; atol=1e-10, rtol=1e-10)
    end

    @testset "view destination" begin
        qs = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)
        X = lagrangec0(cuboid; order=1)
        ref = assemble(op, X, X; threading=:single, quadstrat=qs)
        asm = ext.gpu_blockassembler(op, X, X; quadstrat=qs)
        tids = [1, 3, 5, 8]; sids = [2, 4, 6, 7]
        workspace = CUDA.zeros(scalartype(op, X, X), length(tids) + 2, length(sids) + 3)
        dst = @view workspace[2:length(tids)+1, 3:length(sids)+2]
        @test asm(dst, tids, sids) === dst
        @test isapprox(Array(dst), ref[tids, sids]; atol=1e-10, rtol=1e-10)
    end
end
