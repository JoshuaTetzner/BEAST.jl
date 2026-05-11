using CUDA
using BEAST
using CompScienceMeshes
using LinearAlgebra

device = 0
blocksize = 8
repeats = 5
block_sizes = [blocksize]

function block_ids(n, blocksize; offset=1)
    blocksize = min(blocksize, n)
    start = min(max(1, offset), n - blocksize + 1)
    return collect(start:start + blocksize - 1)
end

function timed_gpu_block_call(ext, gpu_asm, block_d, test_ids, trial_ids)
    request = nothing
    request_time = @elapsed begin
        request = ext._gpu_block_request(gpu_asm.tfs, test_ids, gpu_asm.bfs, trial_ids)
    end

    test_el_d = test_el_ids_d = test_ad_d = test_qd = nothing
    trial_el_d = trial_el_ids_d = trial_ad_d = trial_qd = nothing
    active_data_time = @elapsed begin
        test_el_d, test_el_ids_d, test_ad_d, test_qd = ext._active_data(
            gpu_asm.testdata, request.testids, request.active_test_el_ids)
        trial_el_d, trial_el_ids_d, trial_ad_d, trial_qd = ext._active_data(
            gpu_asm.trialdata, request.trialids, request.active_trial_el_ids)
        CUDA.synchronize()
    end

    assemble_time = @elapsed begin
        if gpu_asm.singularrules === nothing
            matrix_d, zlocal_d, trial_proj_d =
                ext._prepare_far_workspace!(
                    gpu_asm.workspace,
                    gpu_asm.biop,
                    gpu_asm.testdata.numshapes, test_el_ids_d, test_ad_d,
                    gpu_asm.trialdata.numshapes, trial_el_ids_d, trial_ad_d)
            ext.assemblechunk_body_gpu_device_far_indexed!(
                matrix_d, zlocal_d, trial_proj_d,
                gpu_asm.biop,
                refspace(gpu_asm.tfs), test_el_d, test_el_ids_d, test_ad_d,
                refspace(gpu_asm.bfs), trial_el_d, trial_el_ids_d, trial_ad_d,
                (test_qd, trial_qd))
        else
            qd_d = (
                test_qd,
                trial_qd,
                gpu_asm.singularrules.common_vert,
                gpu_asm.singularrules.common_edge,
                gpu_asm.singularrules.common_face,
            )
            matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d =
                ext._prepare_workspace!(
                    gpu_asm.workspace,
                    gpu_asm.biop,
                    gpu_asm.testdata.numshapes, test_el_ids_d, test_ad_d,
                    gpu_asm.trialdata.numshapes, trial_el_ids_d, trial_ad_d)
            ext.assemblechunk_body_gpu_device_indexed!(
                matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d,
                gpu_asm.biop,
                refspace(gpu_asm.tfs), test_el_d, test_el_ids_d, test_ad_d,
                refspace(gpu_asm.bfs), trial_el_d, trial_el_ids_d, trial_ad_d,
                qd_d)
        end
        CUDA.synchronize()
    end

    copy_time = @elapsed begin
        copyto!(block_d, gpu_asm.workspace.matrix_d)
        CUDA.synchronize()
    end

    return (;
        request=request_time,
        active_data=active_data_time,
        assemble=assemble_time,
        copy=copy_time,
        total=request_time + active_data_time + assemble_time + copy_time,
        active_test_elements=length(request.active_test_el_ids),
        active_trial_elements=length(request.active_trial_el_ids),
    )
end

function print_timings(label, timings)
    println(label)
    println("  request/setup CPU = ", round(1e3 * timings.request; digits=3), " ms")
    println("  active data/maps  = ", round(1e3 * timings.active_data; digits=3), " ms")
    println("  GPU assemble      = ", round(1e3 * timings.assemble; digits=3), " ms")
    println("  copy to output    = ", round(1e3 * timings.copy; digits=3), " ms")
    println("  total             = ", round(1e3 * timings.total; digits=3), " ms")
    println("  active elements   = ",
        timings.active_test_elements, " x ", timings.active_trial_elements)
end

function mean_timings(timings)
    n = length(timings)
    return (;
        request=sum(t.request for t in timings) / n,
        active_data=sum(t.active_data for t in timings) / n,
        assemble=sum(t.assemble for t in timings) / n,
        copy=sum(t.copy for t in timings) / n,
        total=sum(t.total for t in timings) / n,
        active_test_elements=last(timings).active_test_elements,
        active_trial_elements=last(timings).active_trial_elements,
    )
end

CUDA.device!(device)
println("device      = ", device, " / ", CUDA.name(CUDA.device()))
println("block sizes = ", block_sizes)
println("repeats     = ", repeats)

mesh = meshcuboid(1.0, 1.0, 1.0, 1.0)
space = lagrangec0(mesh; order=1)
operator = Helmholtz3D.singlelayer(wavenumber=1.0)
qstrat = BEAST.DoubleNumSauterQstrat(2, 2, 2, 2, 2, 2)

ext = Base.get_extension(BEAST, :BEASTCUDAExt)
@assert ext !== nothing "Load CUDA before BEAST or use an environment with CUDA available."

println("numcells    = ", numcells(mesh))
println("numfunctions= ", numfunctions(space))

println()
println("constructing persistent GPU blockassembler")
CUDA.@time gpu_asm = ext.gpu_blockassembler(operator, space, space;
    quadstrat=qstrat,
    device=device)
cpu_asm = BEAST.blockassembler(operator, space, space; quadstrat=qstrat)

for blocksize in block_sizes
    test_ids = block_ids(numfunctions(space), blocksize; offset=1)
    trial_ids = block_ids(numfunctions(space), blocksize;
        offset=max(1, numfunctions(space) - blocksize + 1))

    block_d = CUDA.zeros(scalartype(operator, space, space), length(test_ids), length(trial_ids))
    println()
    println("============================================================")
    println("block size  = ", blocksize)
    println("test ids    = ", test_ids)
    println("trial ids   = ", trial_ids)
    println("block eltype= ", eltype(block_d))

    println()
    first_timings = timed_gpu_block_call(ext, gpu_asm, block_d, test_ids, trial_ids)
    print_timings("first GPU block call", first_timings)

    println()
    repeated_timings = [
        timed_gpu_block_call(ext, gpu_asm, block_d, test_ids, trial_ids)
        for _ in 1:repeats
    ]
    print_timings("mean repeated GPU block call", mean_timings(repeated_timings))

    cpu_block = zeros(scalartype(operator, space, space), length(test_ids), length(trial_ids))
    store(v, m, n) = (@inbounds cpu_block[m, n] += v)

    println()
    println("CPU block reference")
    @time cpu_asm(test_ids, trial_ids, store)

    gpu_block = Array(block_d)
    println()
    println("max_abs_diff= ", maximum(abs.(gpu_block .- cpu_block)))
    println("rel_fro_diff= ", norm(gpu_block - cpu_block) / norm(cpu_block))
end
