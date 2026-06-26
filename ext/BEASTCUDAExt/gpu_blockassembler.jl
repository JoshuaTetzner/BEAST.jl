"""
    GPUAssembleblockbodyFunctor

Callable produced by [`gpu_blockassembler`](@ref). Holds the GPU-resident
per-space assembly data (elements, dof→shape matrices, precomputed shape values)
and quadrature rules for one `(operator, test space, trial space, device)`, and
assembles arbitrary dof sub-blocks when called — see its call method,
`(f::GPUAssembleblockbodyFunctor)(block_d, testids, trialids)`. The same functor
also backs the batched [`gpu_sparse_blockassemble`](@ref).
"""
struct GPUAssembleblockbodyFunctor{B,T1,T2,T3,T4,T5,T6}
    biop::B
    tfs::T1
    bfs::T2
    testdata::T3
    trialdata::T4
    singularrules::T5
    workspace::T6
    device::Int
    verbose::Bool
end

mutable struct GPUBlockWorkspace
    quadstrat_d::Any
    singularity_map_d::Any
    zlocal_d::Any
    matrix_d::Any
    trial_proj_d::Any
end

GPUBlockWorkspace() = GPUBlockWorkspace(nothing, nothing, nothing, nothing, nothing)

struct GPUSauterSchwabRules{V,E,F}
    common_vert::V
    common_edge::E
    common_face::F
end

struct GPUSpaceAssemblyData{S,E,A,Q,Sh,C,M,N}
    space::S
    elements_d::E
    assembly_d::A
    quadrule_d::Q
    shapes_d::Sh
    activecells::C
    active_cell_to_data_id::M
    numshapes::N
end

struct GPUBlockRequest{T1,T2,T3,T4}
    testids::T1
    trialids::T2
    active_test_el_ids::T3
    active_trial_el_ids::T4
end

function _resolved_quadstrat(quadstrat, biop, tfs, bfs)
    qs = applicable(quadstrat, biop, tfs, bfs) ? quadstrat(biop, tfs, bfs) : quadstrat

    if CompScienceMeshes.refines(geometry(tfs), geometry(bfs))
        return TestRefinesTrialQStrat(qs)
    elseif CompScienceMeshes.refines(geometry(bfs), geometry(tfs))
        return TrialRefinesTestQStrat(qs)
    else
        return qs
    end
end

_conforming_quadstrat(qs) = qs
_conforming_quadstrat(qs::BEAST.TestRefinesTrialQStrat) = qs.conforming_qstrat
_conforming_quadstrat(qs::BEAST.TrialRefinesTestQStrat) = qs.conforming_qstrat

function _check_supported_gpu_quadstrat(qs, conforming_qs)
    if qs !== conforming_qs
        throw(ArgumentError(
            "gpu_blockassembler currently supports conforming test/trial meshes only. " *
            "Nonconforming/refinement quad strategies still need a GPU implementation."))
    end
end

function _gpu_block_quad_config(qs::BEAST.DoubleNumQStrat)
    return qs.outer_rule, qs.inner_rule, nothing
end

function _gpu_block_quad_config(qs::BEAST.DoubleNumSauterQstrat)
    return qs.outer_rule, qs.inner_rule, GPUSauterSchwabRules(qs)
end

function _gpu_block_quad_config(qs)
    throw(ArgumentError(
        "gpu_blockassembler currently supports DoubleNumQStrat for far blocks " *
        "and DoubleNumSauterQstrat for Sauter near-capable blocks. " *
        "Got $(typeof(qs))."))
end

function _active_cell_lookup(activecells)
    lookup = Dict{Int,Int}()
    for (data_id, cell_id) in enumerate(activecells)
        lookup[cell_id] = data_id
    end
    return lookup
end

function GPUSpaceAssemblyData(operator::Operator, space::Space, quadrule, ::Type{T};
    verbose=false) where T

    _, ((elements_d, assembly_d), (quadrule_d, shapes_d), activecells) =
        assemble_primer_gpu(operator, space, quadrule, T; verbose=verbose)
    element_domain = CUDA.@allowscalar domain(elements_d[1])
    numshapes = numfunctions(refspace(space), element_domain)

    return GPUSpaceAssemblyData(
        space, elements_d, assembly_d, quadrule_d, shapes_d,
        activecells, _active_cell_lookup(activecells), numshapes)
end

function GPUSauterSchwabRules(qs)
    return GPUSauterSchwabRules(
        gpu_legendre_rule(qs.sauter_schwab_common_vert),
        gpu_legendre_rule(qs.sauter_schwab_common_edge),
        gpu_legendre_rule(qs.sauter_schwab_common_face))
end

function _active_element_ids(space, ids)
    element_ids = Int[]
    for id in ids
        for shape in space.fns[id]
            push!(element_ids, shape.cellid)
        end
    end
    return unique!(sort!(element_ids))
end

"""
    nearblock_element_mask(tfs, values, bfs, nearvalues) -> SparseMatrixCSC{Bool}

Build a boolean element-pair mask from the per-block dof index lists `values`
(test) and `nearvalues` (trial), as produced for the near-field block loop.

Entry `[p, q]` is `true` iff some block couples a test dof supported on test
element `p` with a trial dof supported on trial element `q`, i.e. the element
pair `(p, q)` has to be integrated on the GPU. Rows index test elements
(`numcells(geometry(tfs))`), columns trial elements (`numcells(geometry(bfs))`).
Pairs shared between blocks are merged into a single `true`.

The nonzeros of the returned mask are the element-pair work list for a batched
GPU assembly: `findnz(mask)` yields the `(p, q)` pairs to integrate once.

Note: this is a CPU utility; [`gpu_sparse_blockassemble`](@ref) builds and
deduplicates its element-pair list directly on the GPU and does not need it. It
is kept for experimentation / inspecting the work list.
"""
function nearblock_element_mask(tfs::Space, values, bfs::Space, nearvalues)
    @assert length(values) == length(nearvalues) "values and nearvalues must align"

    rows = Int[]
    cols = Int[]
    for i in eachindex(values)
        test_els = _active_element_ids(tfs, values[i])
        trial_els = _active_element_ids(bfs, nearvalues[i])
        for q in trial_els
            for p in test_els
                push!(rows, p)
                push!(cols, q)
            end
        end
    end

    n_test_el = numcells(geometry(tfs))
    n_trial_el = numcells(geometry(bfs))
    return sparse(rows, cols, trues(length(rows)), n_test_el, n_trial_el, |)
end

function _gpu_block_request(tfs, testids, bfs, trialids)
    testids_vec = collect(Int, testids)
    trialids_vec = collect(Int, trialids)

    active_test_el_ids = _active_element_ids(tfs, testids_vec)
    active_trial_el_ids = _active_element_ids(bfs, trialids_vec)

    return GPUBlockRequest(
        testids_vec,
        trialids_vec,
        active_test_el_ids,
        active_trial_el_ids,
    )
end

function _block_assembly_gpu(data::GPUSpaceAssemblyData, ids, active_element_ids)
    space = data.space
    T = eltype(data.assembly_d)
    element_id_to_block = Dict{Int,Int}()
    for (i, element_id) in enumerate(active_element_ids)
        element_id_to_block[element_id] = i
    end

    rows = Int[]
    cols = Int[]
    vals = T[]
    for (block_dof, dof) in enumerate(ids)
        for shape in space.fns[dof]
            element_block_id = element_id_to_block[shape.cellid]
            push!(rows, block_dof)
            push!(cols, data.numshapes * (element_block_id - 1) + shape.refid)
            push!(vals, T(shape.coeff))
        end
    end

    return CuSparseMatrixCSC(sparse(
        rows,
        cols,
        vals,
        length(ids),
        data.numshapes * length(active_element_ids)))
end

function _active_data(data::GPUSpaceAssemblyData, ids, active_element_ids)
    data_element_ids = [data.active_cell_to_data_id[id] for id in active_element_ids]
    element_ids_d = CuArray(data_element_ids)
    assembly_d = _block_assembly_gpu(data, ids, active_element_ids)
    return data.elements_d, element_ids_d, assembly_d, (data.quadrule_d, data.shapes_d)
end

function _workspace_buffer!(workspace::GPUBlockWorkspace, field::Symbol, ::Type{T}, dims...) where T
    buffer = getfield(workspace, field)
    if buffer === nothing || eltype(buffer) !== T || size(buffer) != dims
        buffer = CUDA.zeros(T, dims...)
        setfield!(workspace, field, buffer)
    end
    return buffer
end

function _prepare_workspace!(workspace::GPUBlockWorkspace,
    operator::IntegralOperator,
    numshapes_test, test_el_ids_d, test_ad_d,
    numshapes_trial, trial_el_ids_d, trial_ad_d)

    numpairs = length(test_el_ids_d) * length(trial_el_ids_d)
    T = promote_type(scalartype(operator), eltype(test_ad_d), eltype(trial_ad_d))
    trial_proj_type = promote_type(T, eltype(trial_ad_d))

    quadstrat_d = _workspace_buffer!(workspace, :quadstrat_d, Int, numpairs, 4)
    singularity_map_d = _workspace_buffer!(workspace, :singularity_map_d, Bool, numpairs, 4)
    zlocal_d = _workspace_buffer!(
        workspace, :zlocal_d, T,
        numshapes_test * length(test_el_ids_d),
        length(trial_el_ids_d) * numshapes_trial)
    matrix_d = _workspace_buffer!(
        workspace, :matrix_d, T,
        size(test_ad_d, 1),
        size(trial_ad_d, 1))
    trial_proj_d = _workspace_buffer!(
        workspace, :trial_proj_d, trial_proj_type,
        size(trial_ad_d, 1),
        size(zlocal_d, 1))

    return matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d
end

function _prepare_far_workspace!(workspace::GPUBlockWorkspace,
    operator::IntegralOperator,
    numshapes_test, test_el_ids_d, test_ad_d,
    numshapes_trial, trial_el_ids_d, trial_ad_d)

    T = promote_type(scalartype(operator), eltype(test_ad_d), eltype(trial_ad_d))
    trial_proj_type = promote_type(T, eltype(trial_ad_d))

    zlocal_d = _workspace_buffer!(
        workspace, :zlocal_d, T,
        numshapes_test * length(test_el_ids_d),
        length(trial_el_ids_d) * numshapes_trial)
    matrix_d = _workspace_buffer!(
        workspace, :matrix_d, T,
        size(test_ad_d, 1),
        size(trial_ad_d, 1))
    trial_proj_d = _workspace_buffer!(
        workspace, :trial_proj_d, trial_proj_type,
        size(trial_ad_d, 1),
        size(zlocal_d, 1))

    return matrix_d, zlocal_d, trial_proj_d
end

"""
    gpu_blockassembler(biop, tfs, bfs; quadstrat, device, verbose) -> GPUAssembleblockbodyFunctor

GPU counterpart of `BEAST.blockassembler`: build a reusable per-block assembler
for `biop` between test space `tfs` and trial space `bfs` on `device`. Call this
once; the returned [`GPUAssembleblockbodyFunctor`](@ref) is then invoked as
`f(block_d, testids, trialids)` to fill device sub-blocks (see its call method).

Per-space setup (element upload, dof→shape assembly matrices, precomputed shape
values) happens here and is reused across all calls and by the batched
[`gpu_sparse_blockassemble`](@ref). This is the entry point used both for the
block-wise scenario and as the building block of the sparse near-interaction
assembler of fast methods.

`quadstrat` must be `DoubleNumSauterQstrat` (Sauter-near capable) or, for far
blocks only, `DoubleNumQStrat`; conforming test/trial meshes only.
"""
function gpu_blockassembler(biop::IntegralOperator, tfs::Space, bfs::Space;
    quadstrat=defaultquadstrat,
    device=CUDA.deviceid(CUDA.device()),
    verbose=false)

    qs = _resolved_quadstrat(quadstrat, biop, tfs, bfs)
    conforming_qs = _conforming_quadstrat(qs)
    _check_supported_gpu_quadstrat(qs, conforming_qs)
    outer_rule, inner_rule, singularrules = _gpu_block_quad_config(conforming_qs)
    T = scalartype(biop, tfs, bfs)
    CUDA.device!(device)
    testdata = GPUSpaceAssemblyData(biop, tfs, outer_rule, T; verbose)
    trialdata = GPUSpaceAssemblyData(biop, bfs, inner_rule, T; verbose)

    return GPUAssembleblockbodyFunctor(
        biop, tfs, bfs, testdata, trialdata,
        singularrules, GPUBlockWorkspace(), Int(device), verbose)
end

"""
    (f::GPUAssembleblockbodyFunctor)(block_d, testids, trialids) -> block_d

Assemble the `testids × trialids` sub-block of the operator into the preallocated
device matrix `block_d` (a `CuMatrix` or a `@view` of one) and return it. This is
the call a fast method's block loop makes per (near) interaction, on the functor
returned by [`gpu_blockassembler`](@ref).

Far blocks (functor built with `DoubleNumQStrat`, no singular rules) take the
double-numerical path; otherwise the Sauter–Schwab near path runs. Empty
`testids`/`trialids` zero the block and return.
"""
function (f::GPUAssembleblockbodyFunctor)(block_d::AbstractMatrix, testids, trialids)
    CUDA.device!(f.device)

    @assert size(block_d, 1) == length(testids)
    @assert size(block_d, 2) == length(trialids)

    if isempty(testids) || isempty(trialids)
        fill!(block_d, zero(eltype(block_d)))
        return block_d
    end

    request = _gpu_block_request(f.tfs, testids, f.bfs, trialids)
    f.verbose && println(
        "GPU block request: ",
        length(request.testids), " x ", length(request.trialids), " dofs, ",
        length(request.active_test_el_ids), " x ",
        length(request.active_trial_el_ids), " active elements")

    test_el_d, test_el_ids_d, test_ad_d, test_qd = _active_data(
        f.testdata, request.testids, request.active_test_el_ids)
    trial_el_d, trial_el_ids_d, trial_ad_d, trial_qd = _active_data(
        f.trialdata, request.trialids, request.active_trial_el_ids)

    if f.singularrules === nothing
        matrix_d, zlocal_d, trial_proj_d =
            _prepare_far_workspace!(
                f.workspace,
                f.biop,
                f.testdata.numshapes, test_el_ids_d, test_ad_d,
                f.trialdata.numshapes, trial_el_ids_d, trial_ad_d)

        block = assemblechunk_body_gpu_device_far_indexed!(
            matrix_d, zlocal_d, trial_proj_d,
            f.biop,
            refspace(f.tfs), test_el_d, test_el_ids_d, test_ad_d,
            refspace(f.bfs), trial_el_d, trial_el_ids_d, trial_ad_d,
            (test_qd, trial_qd))

        copyto!(block_d, block)
        return block_d
    end

    qd_d = (
        test_qd,
        trial_qd,
        f.singularrules.common_vert,
        f.singularrules.common_edge,
        f.singularrules.common_face,
    )

    matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d =
        _prepare_workspace!(
            f.workspace,
            f.biop,
            f.testdata.numshapes, test_el_ids_d, test_ad_d,
            f.trialdata.numshapes, trial_el_ids_d, trial_ad_d)

    block = assemblechunk_body_gpu_device_indexed!(
        matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d,
        f.biop,
        refspace(f.tfs), test_el_d, test_el_ids_d, test_ad_d,
        refspace(f.bfs), trial_el_d, trial_el_ids_d, trial_ad_d,
        qd_d)

    copyto!(block_d, block)
    return block_d
end
