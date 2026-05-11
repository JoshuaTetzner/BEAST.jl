using CUDA
using BEAST
using CompScienceMeshes
using LinearAlgebra

device = 0
h = 0.1
distance = 3.0
block_sizes = [64, 128, 256, 512]
repeats = 3
qstrat = BEAST.DoubleNumQStrat(2, 2)

function block_ids(n, blocksize; offset=1)
    blocksize = min(blocksize, n)
    start = min(max(1, offset), n - blocksize + 1)
    return collect(start:start + blocksize - 1)
end

function mean_namedtuples(ts)
    n = length(ts)
    names = keys(first(ts))
    return NamedTuple{names}(map(name -> sum(getfield(t, name) for t in ts) / n, names))
end

function timed_active_data(ext, data, ids, active_element_ids)
    data_element_ids = nothing
    index_time = @elapsed begin
        data_element_ids = [data.active_cell_to_data_id[id] for id in active_element_ids]
    end

    element_ids_d = nothing
    id_upload_time = @elapsed begin
        element_ids_d = CuArray(data_element_ids)
        CUDA.synchronize()
    end

    assembly_d = nothing
    assembly_time = @elapsed begin
        assembly_d = ext._block_assembly_gpu(data, ids, active_element_ids)
        CUDA.synchronize()
    end

    return (; elements_d=data.elements_d, element_ids_d, assembly_d,
        qd=(data.quadrule_d, data.shapes_d), index_time, id_upload_time, assembly_time)
end

function time_far_call(ext, gpu_asm, block_d, test_ids, trial_ids)
    request = nothing
    request_time = @elapsed begin
        request = ext._gpu_block_request(gpu_asm.tfs, test_ids, gpu_asm.bfs, trial_ids)
    end

    test_active = timed_active_data(ext, gpu_asm.testdata, request.testids, request.active_test_el_ids)
    trial_active = timed_active_data(ext, gpu_asm.trialdata, request.trialids, request.active_trial_el_ids)
    active_index_time = test_active.index_time + trial_active.index_time
    active_id_upload_time = test_active.id_upload_time + trial_active.id_upload_time
    active_assembly_time = test_active.assembly_time + trial_active.assembly_time

    test_el_d, test_el_ids_d, test_ad_d, test_qd =
        test_active.elements_d, test_active.element_ids_d, test_active.assembly_d, test_active.qd
    trial_el_d, trial_el_ids_d, trial_ad_d, trial_qd =
        trial_active.elements_d, trial_active.element_ids_d, trial_active.assembly_d, trial_active.qd

    matrix_d = zlocal_d = trial_proj_d = nothing
    workspace_time = @elapsed begin
        matrix_d, zlocal_d, trial_proj_d = ext._prepare_far_workspace!(
            gpu_asm.workspace,
            gpu_asm.biop,
            gpu_asm.testdata.numshapes, test_el_ids_d, test_ad_d,
            gpu_asm.trialdata.numshapes, trial_el_ids_d, trial_ad_d)
        CUDA.synchronize()
    end

    (quadrule_d, test_shapes_d), (_, trial_shapes_d) = (test_qd, trial_qd)
    fill_time = @elapsed begin
        fill!(zlocal_d, zero(eltype(zlocal_d)))
        fill!(matrix_d, zero(eltype(matrix_d)))
        CUDA.synchronize()
    end

    integral_time = @elapsed begin
        npairs = length(test_el_ids_d) * length(trial_el_ids_d)
        ext.launch_gpu_kernel!(ext.gpu_momintegral_doublenum_allpairs_indexed!, zlocal_d, gpu_asm.biop, npairs,
            test_el_d, test_el_ids_d, trial_el_d, trial_el_ids_d, test_shapes_d, trial_shapes_d,
            refspace(gpu_asm.tfs), refspace(gpu_asm.bfs), quadrule_d;
            gpu_blocksize=(256), problem_size=npairs)
        CUDA.synchronize()
    end

    projection_time = @elapsed begin
        ext.build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d, trial_proj_d)
        CUDA.synchronize()
    end

    copy_time = @elapsed begin
        copyto!(block_d, matrix_d)
        CUDA.synchronize()
    end

    return (;
        request=request_time,
        active_index=active_index_time,
        active_id_upload=active_id_upload_time,
        active_assembly=active_assembly_time,
        workspace=workspace_time,
        fill=fill_time,
        integral=integral_time,
        projection=projection_time,
        copy=copy_time,
        active_test_elements=length(request.active_test_el_ids),
        active_trial_elements=length(request.active_trial_el_ids),
    )
end

function print_timing(blocksize, t)
    total = t.request + t.active_index + t.active_id_upload + t.active_assembly +
        t.workspace + t.fill + t.integral + t.projection + t.copy
    println()
    println("block size        = ", blocksize)
    println("active elements   = ", round(Int, t.active_test_elements), " x ", round(Int, t.active_trial_elements))
    println("request           = ", round(1e3 * t.request; digits=3), " ms")
    println("active index map  = ", round(1e3 * t.active_index; digits=3), " ms")
    println("active id upload  = ", round(1e3 * t.active_id_upload; digits=3), " ms")
    println("sparse map upload = ", round(1e3 * t.active_assembly; digits=3), " ms")
    println("workspace         = ", round(1e3 * t.workspace; digits=3), " ms")
    println("fill              = ", round(1e3 * t.fill; digits=3), " ms")
    println("integrals         = ", round(1e3 * t.integral; digits=3), " ms")
    println("CUSPARSE project  = ", round(1e3 * t.projection; digits=3), " ms")
    println("copy              = ", round(1e3 * t.copy; digits=3), " ms")
    println("total             = ", round(1e3 * total; digits=3), " ms")
end

CUDA.device!(device)
println("device      = ", device, " / ", CUDA.name(CUDA.device()))
println("h           = ", h)
println("distance    = ", distance)
println("block sizes = ", block_sizes)
println("repeats     = ", repeats)

mesh_test = meshcuboid(1.0, 1.0, 1.0, h)
mesh_trial = CompScienceMeshes.translate(mesh_test, point(distance, 0.0, 0.0))
space_test = lagrangec0(mesh_test; order=1)
space_trial = lagrangec0(mesh_trial; order=1)
operator = Helmholtz3D.singlelayer(wavenumber=1.0)

ext = Base.get_extension(BEAST, :BEASTCUDAExt)
@assert ext !== nothing "Load CUDA before BEAST or use an environment with CUDA available."

println("test cells  = ", numcells(mesh_test), ", dofs = ", numfunctions(space_test))
println("trial cells = ", numcells(mesh_trial), ", dofs = ", numfunctions(space_trial))

println()
println("constructing far GPU blockassembler")
CUDA.@time gpu_asm = ext.gpu_blockassembler(operator, space_test, space_trial;
    quadstrat=qstrat,
    device=device)

for blocksize in block_sizes
    test_ids = block_ids(numfunctions(space_test), blocksize; offset=1)
    trial_ids = block_ids(numfunctions(space_trial), blocksize; offset=1)
    block_d = CUDA.zeros(scalartype(operator, space_test, space_trial), length(test_ids), length(trial_ids))

    time_far_call(ext, gpu_asm, block_d, test_ids, trial_ids)
    timings = [time_far_call(ext, gpu_asm, block_d, test_ids, trial_ids) for _ in 1:repeats]
    print_timing(blocksize, mean_namedtuples(timings))
end
