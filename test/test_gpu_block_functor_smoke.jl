using Test
using CUDA
using BEAST
using CompScienceMeshes
using LinearAlgebra
##
@testset "GPU block functor inactive cells regression" begin
    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    @test ext !== nothing

    constructor = if isdefined(ext, :gpu_blockassembler)
        getfield(ext, :gpu_blockassembler)
    elseif isdefined(ext, :blockassembler_gpu)
        getfield(ext, :blockassembler_gpu)
    else
        println("GPU block functor regression skipped: no gpu_blockassembler/blockassembler_gpu yet.")
        nothing
    end

    if constructor !== nothing
        device_id = begin
            value = get(ENV, "BEAST_GPU_DEVICE", "")
            isempty(value) ? 0 : parse(Int, value)
        end
        CUDA.device!(device_id)

        mesh = meshcuboid(1.0, 1.0, 1.0, 1.0)
        space_full = lagrangec0(mesh; order=1)
        n = numfunctions(space_full)
        keep = collect(max(1, n - 2):n)
        space = subset(space_full, keep)

        op = Helmholtz3D.singlelayer(gamma=1.0)
        qstrat = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)

        test_ids = collect(1:numfunctions(space))
        trial_ids = test_ids

        cpu_assembler = BEAST.blockassembler(op, space, space; quadstrat=qstrat)
        cpu_block = zeros(scalartype(op, space, space), length(test_ids), length(trial_ids))
        store(v, m, n) = (@inbounds cpu_block[m, n] += v)
        cpu_assembler(test_ids, trial_ids, store)

        gpu_assembler = constructor(op, space, space; quadstrat=qstrat)
        gpu_block_prealloc_d = CUDA.zeros(scalartype(op, space, space), length(test_ids), length(trial_ids))
        @test eltype(gpu_block_prealloc_d) === Float64
        returned_d = gpu_assembler(gpu_block_prealloc_d, test_ids, trial_ids)
        @test returned_d === gpu_block_prealloc_d
        gpu_block = Array(gpu_block_prealloc_d)
        max_abs = maximum(abs.(gpu_block .- cpu_block))
        denom = norm(cpu_block)
        rel = iszero(denom) ? norm(gpu_block - cpu_block) : norm(gpu_block - cpu_block) / denom
        println("GPU block functor inactive cells")
        println("  size         = ", size(gpu_block))
        println("  max_abs_diff = ", max_abs)
        println("  rel_fro_diff = ", rel)
        @test isapprox(gpu_block, cpu_block; atol=1e-11, rtol=1e-11)
    end
end

@testset "CUDA device smoke" begin
    @test CUDA.functional()
    @test length(CUDA.devices()) >= 1

    println("CUDA.functional() = ", CUDA.functional())
    println("CUDA runtime      = ", CUDA.runtime_version())
    println("CUDA driver       = ", CUDA.driver_version())
    println("CUDA devices      = ", length(CUDA.devices()))

    for dev in CUDA.devices()
        CUDA.device!(dev)
        println("  device ", CUDA.deviceid(dev), ": ", CUDA.name(dev),
            ", capability=", CUDA.capability(dev),
            ", totalmem_GiB=", round(CUDA.totalmem(dev) / 2.0^30; digits=2))
    end

    device_id = begin
        value = get(ENV, "BEAST_GPU_DEVICE", "")
        isempty(value) ? 0 : parse(Int, value)
    end
    CUDA.device!(device_id)
    x = CUDA.fill(Float32(device_id + 1), 1_000_000)
    y = similar(x)
    CUDA.@sync y .= 3f0 .* x .- 1f0
    expected = Float32(3 * (device_id + 1) - 1) * length(y)
    @test sum(y) ≈ expected rtol = 1f-5
    CUDA.unsafe_free!(x)
    CUDA.unsafe_free!(y)
end

@testset "CPU AssembleblockbodyFunctor smoke" begin
    meshfile = get(ENV, "BEAST_BLOCK_MESH", joinpath(@__DIR__, "assets", "sphere35.in"))
    mesh = readmesh(meshfile)
    space = raviartthomas(mesh)
    op = Maxwell3D.singlelayer(wavenumber=1.0)

    n = numfunctions(space)
    block_size = begin
        value = get(ENV, "BEAST_BLOCK_SIZE", "")
        isempty(value) ? 12 : parse(Int, value)
    end
    block_size = min(block_size, n)
    ids = block_size == n ? collect(1:n) : unique!(sort!(round.(Int, range(1, n; length=block_size))))

    println("CPU block functor smoke")
    println("  mesh         = ", meshfile)
    println("  numcells     = ", numcells(mesh))
    println("  numfunctions = ", n)
    println("  block ids    = ", ids)

    assembler = BEAST.blockassembler(op, space, space)
    block = zeros(scalartype(op, space, space), length(ids), length(ids))
    store(v, m, n) = (@inbounds block[m, n] += v)
    assembler(ids, ids, store)

    subspace = subset(space, ids)
    reference = assemble(op, subspace, subspace)
    tol = sqrt(eps(real(eltype(reference))))
    max_abs = maximum(abs.(block .- reference))
    denom = norm(reference)
    rel = iszero(denom) ? norm(block - reference) : norm(block - reference) / denom
    println("CPU block functor vs subset assembly")
    println("  size         = ", size(block))
    println("  max_abs_diff = ", max_abs)
    println("  rel_fro_diff = ", rel)
    @test isapprox(block, reference; atol=tol, rtol=tol)

    full = assemble(op, space, space)
    tol = sqrt(eps(real(eltype(full))))
    slice = full[ids, ids]
    max_abs = maximum(abs.(block .- slice))
    denom = norm(slice)
    rel = iszero(denom) ? norm(block - slice) : norm(block - slice) / denom
    println("CPU block functor vs dense matrix slice")
    println("  size         = ", size(block))
    println("  max_abs_diff = ", max_abs)
    println("  rel_fro_diff = ", rel)
    @test isapprox(block, slice; atol=tol, rtol=tol)
end

@testset "Current GPU full assembly smoke" begin
    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    @test ext !== nothing

    device_id = begin
        value = get(ENV, "BEAST_GPU_DEVICE", "")
        isempty(value) ? 0 : parse(Int, value)
    end
    CUDA.device!(device_id)

    mesh = meshcuboid(1.0, 1.0, 1.0, 1.0)
    space = lagrangec0(mesh; order=1)
    op = Helmholtz3D.singlelayer(wavenumber=1.0)
    qstrat = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)
    tiling = ext.TilingStrategy(ext.WorksizeTiling(16), ext.WorksizeTiling(16))

    println("GPU full assembly smoke")
    println("  numcells     = ", numcells(mesh))
    println("  numfunctions = ", numfunctions(space))

    gpu = assemble(op, space, space; threading=:gpu, tilingstrat=tiling, quadstrat=qstrat)
    cpu = assemble(op, space, space; threading=:single, quadstrat=qstrat)
    max_abs = maximum(abs.(gpu .- cpu))
    denom = norm(cpu)
    rel = iszero(denom) ? norm(gpu - cpu) : norm(gpu - cpu) / denom
    println("GPU full assembly vs CPU single-thread")
    println("  size         = ", size(gpu))
    println("  max_abs_diff = ", max_abs)
    println("  rel_fro_diff = ", rel)
    @test isapprox(gpu, cpu; atol=1e-11, rtol=1e-11)
end

@testset "GPU block functor smoke" begin
    ext = Base.get_extension(BEAST, :BEASTCUDAExt)
    @test ext !== nothing

    constructor = if isdefined(ext, :gpu_blockassembler)
        getfield(ext, :gpu_blockassembler)
    elseif isdefined(ext, :blockassembler_gpu)
        getfield(ext, :blockassembler_gpu)
    else
        println("GPU block functor smoke skipped: no gpu_blockassembler/blockassembler_gpu yet.")
        nothing
    end

    if constructor !== nothing
        device_id = begin
            value = get(ENV, "BEAST_GPU_DEVICE", "")
            isempty(value) ? 0 : parse(Int, value)
        end
        CUDA.device!(device_id)

        mesh = meshcuboid(1.0, 1.0, 1.0, 1.0)
        space = lagrangec0(mesh; order=1)
        op = Helmholtz3D.singlelayer(wavenumber=1.0)
        qstrat = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)
        test_ids = [1, 3, 5, 8]
        trial_ids = [2, 4, 6, 7]

        gpu_assembler = constructor(op, space, space; quadstrat=qstrat)

        cpu_assembler = BEAST.blockassembler(op, space, space; quadstrat=qstrat)
        cpu_block = zeros(scalartype(op, space, space), length(test_ids), length(trial_ids))
        store(v, m, n) = (@inbounds cpu_block[m, n] += v)
        cpu_assembler(test_ids, trial_ids, store)

        gpu_block_prealloc_d = CUDA.zeros(scalartype(op, space, space), length(test_ids), length(trial_ids))
        @test eltype(gpu_block_prealloc_d) === ComplexF64
        returned_d = gpu_assembler(gpu_block_prealloc_d, test_ids, trial_ids)
        @test returned_d === gpu_block_prealloc_d
        gpu_block = Array(gpu_block_prealloc_d)
        max_abs = maximum(abs.(gpu_block .- cpu_block))
        denom = norm(cpu_block)
        rel = iszero(denom) ? norm(gpu_block - cpu_block) : norm(gpu_block - cpu_block) / denom
        println("GPU block functor preallocated CuMatrix vs CPU block functor")
        println("  size         = ", size(gpu_block))
        println("  max_abs_diff = ", max_abs)
        println("  rel_fro_diff = ", rel)
        @test isapprox(gpu_block, cpu_block; atol=1e-11, rtol=1e-11)

        gpu_workspace_d = CUDA.zeros(scalartype(op, space, space), length(test_ids) + 2, length(trial_ids) + 3)
        gpu_block_view_d = @view gpu_workspace_d[2:length(test_ids)+1, 3:length(trial_ids)+2]
        returned_view_d = gpu_assembler(gpu_block_view_d, test_ids, trial_ids)
        @test returned_view_d === gpu_block_view_d
        gpu_block_view = Array(gpu_block_view_d)
        max_abs = maximum(abs.(gpu_block_view .- cpu_block))
        denom = norm(cpu_block)
        rel = iszero(denom) ? norm(gpu_block_view - cpu_block) : norm(gpu_block_view - cpu_block) / denom
        println("GPU block functor CuMatrix view vs CPU block functor")
        println("  size         = ", size(gpu_block_view))
        println("  max_abs_diff = ", max_abs)
        println("  rel_fro_diff = ", rel)
        @test isapprox(gpu_block_view, cpu_block; atol=1e-11, rtol=1e-11)

        full_ids = collect(1:numfunctions(space))
        cpu_full_block = zeros(scalartype(op, space, space), length(full_ids), length(full_ids))
        store_full(v, m, n) = (@inbounds cpu_full_block[m, n] += v)
        cpu_assembler(full_ids, full_ids, store_full)

        gpu_full_block_d = CUDA.zeros(scalartype(op, space, space), length(full_ids), length(full_ids))
        returned_full_d = gpu_assembler(gpu_full_block_d, full_ids, full_ids)
        @test returned_full_d === gpu_full_block_d
        gpu_full_block = Array(gpu_full_block_d)
        max_abs = maximum(abs.(gpu_full_block .- cpu_full_block))
        denom = norm(cpu_full_block)
        rel = iszero(denom) ? norm(gpu_full_block - cpu_full_block) : norm(gpu_full_block - cpu_full_block) / denom
        println("GPU block functor full tiny block vs CPU block functor")
        println("  size         = ", size(gpu_full_block))
        println("  max_abs_diff = ", max_abs)
        println("  rel_fro_diff = ", rel)
        @test isapprox(gpu_full_block, cpu_full_block; atol=1e-11, rtol=1e-11)

        far_mesh = CompScienceMeshes.translate(mesh, point(3.0, 0.0, 0.0))
        far_space = lagrangec0(far_mesh; order=1)
        far_qstrat = BEAST.DoubleNumQStrat(2, 2)
        far_test_ids = [1, 2]
        far_trial_ids = [1, 2]
        far_gpu_assembler = constructor(op, space, far_space; quadstrat=far_qstrat)
        far_cpu_assembler = BEAST.blockassembler(op, space, far_space; quadstrat=far_qstrat)

        far_cpu_block = zeros(scalartype(op, space, space), length(far_test_ids), length(far_trial_ids))
        far_store(v, m, n) = (@inbounds far_cpu_block[m, n] += v)
        far_cpu_assembler(far_test_ids, far_trial_ids, far_store)

        far_gpu_block_d = CUDA.zeros(scalartype(op, space, space), length(far_test_ids), length(far_trial_ids))
        returned_far_d = far_gpu_assembler(far_gpu_block_d, far_test_ids, far_trial_ids)
        @test returned_far_d === far_gpu_block_d
        far_gpu_block = Array(far_gpu_block_d)
        max_abs = maximum(abs.(far_gpu_block .- far_cpu_block))
        denom = norm(far_cpu_block)
        rel = iszero(denom) ? norm(far_gpu_block - far_cpu_block) : norm(far_gpu_block - far_cpu_block) / denom
        println("GPU block functor DoubleNumQStrat far block vs CPU block functor")
        println("  size         = ", size(far_gpu_block))
        println("  max_abs_diff = ", max_abs)
        println("  rel_fro_diff = ", rel)
        @test isapprox(far_gpu_block, far_cpu_block; atol=1e-11, rtol=1e-11)
    end
end
