# Sparse block assembler: the batched form of the block assembler for the near
# interactions of a fast method. Given a list of dof blocks (`values[i]` test
# dofs, `nearvalues[i]` trial dofs) all blocks are assembled into one sparse
# matrix; block i is `Z[values[i], nearvalues[i]]`. This is unrelated to the
# radiated near field. Ground truth is the matching slice of the CPU assembly.

@testitem "GPU sparse block assembler vs CPU" tags = [:gpu] begin
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

            # a few overlapping dof blocks, so the element-pair deduplication is exercised
            m = min(numfunctions(X), 24)
            a = collect(1:m÷2)
            b = collect(m÷2+1:m)
            values = [a, a, b]
            nearvalues = [a, b, b]

            dense = Array(ext.gpu_sparse_blockassemble(asm, values, nearvalues))
            @test size(dense) == size(ref)
            for i in eachindex(values)
                @test isapprox(dense[values[i], nearvalues[i]], ref[values[i], nearvalues[i]];
                    atol=1e-10, rtol=1e-10)
            end
        end
    end
end

@testitem "GPU sparse block assembler is chunk-invariant" tags = [:gpu] begin
    using CUDA
    using CompScienceMeshes
    using LinearAlgebra
    using Test

    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    CUDA.device!(parse(Int, get(ENV, "BEAST_GPU_DEVICE", "0")))

    op = Maxwell3D.singlelayer(wavenumber=1.0)
    X = raviartthomas(readmesh(joinpath(@__DIR__, "..", "assets", "sphere5.in"); T=Float64))
    qs = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)
    asm = ext.gpu_blockassembler(op, X, X; quadstrat=qs)

    m = min(numfunctions(X), 24)
    values = [collect(1:m)]
    nearvalues = [collect(1:m)]

    full = Array(ext.gpu_sparse_blockassemble(asm, values, nearvalues))
    chunked = Array(ext.gpu_sparse_blockassemble(asm, values, nearvalues; maxchunkpairs=7))
    @test isapprox(full, chunked; atol=1e-12, rtol=1e-12)
end

@testitem "GPU sparse block assembler multi-device" tags = [:gpu] begin
    using CUDA
    using CompScienceMeshes
    using LinearAlgebra
    using Test

    if length(CUDA.devices()) < 2
        @test_skip "needs at least two CUDA devices"
    else
        ext = Base.get_extension(BEAST, :BEASTCUDAExt)
        op = Maxwell3D.singlelayer(wavenumber=1.0)
        X = raviartthomas(readmesh(joinpath(@__DIR__, "..", "assets", "sphere5.in"); T=Float64))
        qs = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)

        CUDA.device!(0)
        ref = assemble(op, X, X; threading=:single, quadstrat=qs)

        m = min(numfunctions(X), 24)
        a = collect(1:m÷2); b = collect(m÷2+1:m)
        values = [a, a, b]
        nearvalues = [a, b, b]

        functors = ext.gpu_sparse_blockassemble_functors(op, X, X; quadstrat=qs, devices=[0, 1])
        partials = ext.gpu_sparse_blockassemble(functors, values, nearvalues)

        # the full result is the exact sum of the per-device partials
        dense = zeros(scalartype(op, X, X), size(ref))
        for (f, P) in zip(functors, partials)
            CUDA.device!(f.device)
            dense .+= Array(P)
        end
        for i in eachindex(values)
            @test isapprox(dense[values[i], nearvalues[i]], ref[values[i], nearvalues[i]];
                atol=1e-10, rtol=1e-10)
        end
    end
end
