#============================================================#
# Sparse block assembler: a batched form of the block assembler for the near
# interactions of a fast method (H-matrix / ACA).
#
# Given the per-block dof lists of the near interactions (`values` / `nearvalues`
# from e.g. AdaptiveCrossApproximation.nearinteractions), all requested blocks
# are assembled in one shot and returned as a single sparse matrix:
#   1. build the element-pair work list from the blocks (union over blocks),
#   2. integrate every pair once into a sparse element-shape matrix `z`,
#   3. project to dofs with the (already on-GPU) assembly matrices:
#          Z = A_test * z * A_trialᵀ.
# The per-space assembly matrices are reused from setup, so there is no
# per-block host sparse build / upload. This is unrelated to the radiated near
# field; it only assembles the dense near-interaction blocks.
#============================================================#

# Integrate one element pair per thread into the COO triplets (Iz, Jz, Vz) of
# the sparse element-shape matrix. Singularity is detected in-kernel and the
# matching rule is applied; each branch carries a concrete strat type so the
# kernel stays type stable (no dynamic dispatch on the GPU).
@inline function _sba_doublenum_z(igd, el_test, el_trial, shp_t, shp_s, test_qr, trial_qr, test_rs, trial_rs, ::Type{T}) where T
    NT = numfunctions(test_rs, CompScienceMeshes.domain(el_test))
    NS = numfunctions(trial_rs, CompScienceMeshes.domain(el_trial))
    z = zeros(StaticArrays.SMatrix{NT,NS,T})
    nt = length(test_qr)
    ns = length(trial_qr)
    for l in 1:nt
        px = test_qr[l][1]
        x = neighborhood(el_test, px)
        wx = test_qr[l][2] * jacobian(x)
        for m in 1:ns
            py = trial_qr[m][1]
            y = neighborhood(el_trial, py)
            wy = trial_qr[m][2] * jacobian(y)
            @inbounds z += wx * wy * igd(x, y, shp_t[l], shp_s[m])
        end
    end
    return z
end

@inline function _sba_sauter_z(igd, el_test, el_trial, strat, test_rs, trial_rs, ::Type{T}) where T
    NT = numfunctions(test_rs, CompScienceMeshes.domain(el_test))
    NS = numfunctions(trial_rs, CompScienceMeshes.domain(el_trial))
    I, J, _, _ = gpu_sauterschwab_reorder(
        CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)
    igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)
    z = zeros(StaticArrays.SMatrix{NT,NS,T})
    qps = strat.qps
    for (η1, w1) in qps
        for (η2, w2) in qps
            for (η3, w3) in qps
                for (ξ, w4) in qps
                    z += w1 * w2 * w3 * w4 * strat(igdp, η1, η2, η3, ξ)
                end
            end
        end
    end
    return z
end

function gpu_sparse_block_integrate!(Iz, Jz, Vz, biop, npairs, pair_p, pair_q,
    test_els::CuDeviceVector{S}, trial_els, test_shapes, trial_shapes,
    test_rs, trial_rs, test_qr, trial_qr, cvs, ces, cfs) where S

    gidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if gidx <= npairs
        p = pair_p[gidx]
        q = pair_q[gidx]
        el_test = test_els[p]
        el_trial = trial_els[q]

        NT = numfunctions(test_rs, CompScienceMeshes.domain(el_test))
        NS = numfunctions(trial_rs, CompScienceMeshes.domain(el_trial))
        T = eltype(Vz)

        tol = 1e3 * eps(coordtype(S))
        hits = 0
        for vt in CompScienceMeshes.vertices(el_test)
            for vb in CompScienceMeshes.vertices(el_trial)
                hits += (norm(vt - vb) < tol)
            end
        end

        igd = BEAST.Integrand(biop, test_rs, trial_rs, el_test, el_trial)

        if hits == 0
            shp_t = view(test_shapes, p, :)
            shp_s = view(trial_shapes, q, :)
            z = _sba_doublenum_z(igd, el_test, el_trial, shp_t, shp_s, test_qr, trial_qr, test_rs, trial_rs, T)
        elseif hits == 1
            z = _sba_sauter_z(igd, el_test, el_trial, cvs, test_rs, trial_rs, T)
        elseif hits == 2
            z = _sba_sauter_z(igd, el_test, el_trial, ces, test_rs, trial_rs, T)
        else
            z = _sba_sauter_z(igd, el_test, el_trial, cfs, test_rs, trial_rs, T)
        end

        base = (gidx - 1) * NT * NS
        for b in 1:NS
            for a in 1:NT
                e = base + (b - 1) * NT + a
                @inbounds Iz[e] = Int32(NT * (p - 1) + a)
                @inbounds Jz[e] = Int32(NS * (q - 1) + b)
                @inbounds Vz[e] = z[a, b]
            end
        end
    end

    return nothing
end

# Dense cell-id -> data-id map (0 for inactive cells), for O(1) lookups.
function _cell_to_data(data::GPUSpaceAssemblyData)
    c2d = zeros(Int, numcells(geometry(data.space)))
    for (cell, id) in data.active_cell_to_data_id
        c2d[cell] = id
    end
    return c2d
end

# Sorted, unique element data-ids supporting the dofs `ids` (reuses `buf`).
function _collect_element_data_ids!(buf, space, ids, c2d)
    empty!(buf)
    for id in ids
        for shape in space.fns[id]
            push!(buf, c2d[shape.cellid])
        end
    end
    sort!(buf)
    return unique!(buf)
end

# One thread per (duplicated) element pair: locate its block by binary search in
# the pair-offset prefix sum, then emit the linear key (p-1) + (q-1)*n_test_el.
function _sba_keygen!(keys, total, pair_off, test_off, trial_off, test_flat, trial_flat, nblocks, n_test_el)
    g = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if g <= total
        lo = 1
        hi = nblocks
        while lo < hi
            mid = (lo + hi + 1) >>> 1
            if pair_off[mid] < g
                lo = mid
            else
                hi = mid - 1
            end
        end
        b = lo
        localidx = g - pair_off[b] - 1
        nt = test_off[b+1] - test_off[b]
        p_local = localidx % nt
        q_local = localidx ÷ nt
        @inbounds p = test_flat[test_off[b]+p_local+1]
        @inbounds q = trial_flat[trial_off[b]+q_local+1]
        @inbounds keys[g] = (p - 1) + (q - 1) * n_test_el
    end
    return nothing
end

# Scatter the unique (run-start) keys into a compact array using the prefix sum.
function _sba_compact!(ukeys, keys, flags, pos, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n && flags[i] == Int32(1)
        @inbounds ukeys[pos[i]] = keys[i]
    end
    return nothing
end

# Unique element-pair work list (data-id space) over all requested blocks.
# The per-block element lists are tiny (O(dofs)) and built on the CPU; the
# expensive enumeration + dedup of element pairs happens on the GPU.
# Returns (pair_p, pair_q) as device arrays of data ids.
function _sba_pairs(f::GPUAssembleblockbodyFunctor, values, nearvalues)
    test_c2d = _cell_to_data(f.testdata)
    trial_c2d = _cell_to_data(f.trialdata)
    n_test_el = length(f.testdata.elements_d)

    nblocks = length(values)
    test_flat = Int[]
    trial_flat = Int[]
    test_off = zeros(Int, nblocks + 1)
    trial_off = zeros(Int, nblocks + 1)
    pair_off = zeros(Int, nblocks + 1)
    buf = Int[]
    for i in 1:nblocks
        test_els = _collect_element_data_ids!(buf, f.tfs, values[i], test_c2d)
        nt = length(test_els)
        append!(test_flat, test_els)
        trial_els = _collect_element_data_ids!(buf, f.bfs, nearvalues[i], trial_c2d)
        ns = length(trial_els)
        append!(trial_flat, trial_els)
        test_off[i+1] = test_off[i] + nt
        trial_off[i+1] = trial_off[i] + ns
        pair_off[i+1] = pair_off[i] + nt * ns
    end
    total = pair_off[end]

    pair_off_d = CuArray(pair_off)
    keys = CUDA.zeros(Int, total)
    launch_gpu_kernel!(_sba_keygen!, keys, total, pair_off_d,
        CuArray(test_off), CuArray(trial_off), CuArray(test_flat), CuArray(trial_flat),
        nblocks, n_test_el; gpu_blocksize=(256), problem_size=(total))

    sort!(keys)

    flags = CUDA.fill(Int32(1), total)
    @views flags[2:end] .= Int32.(keys[2:end] .!= keys[1:end-1])
    pos = accumulate(+, flags)
    nuniq = CUDA.@allowscalar pos[end]
    ukeys = CUDA.zeros(Int, nuniq)
    launch_gpu_kernel!(_sba_compact!, ukeys, keys, flags, pos, total;
        gpu_blocksize=(256), problem_size=(total))

    pair_p = ukeys .% n_test_el .+ 1
    pair_q = ukeys .÷ n_test_el .+ 1
    return pair_p, pair_q
end

# Integrate all element pairs into the COO triplets of the element-shape
# matrix z (size NT*n_test_el x NS*n_trial_el).
function _sba_zcoo(f::GPUAssembleblockbodyFunctor, p_data, q_data)
    tdata = f.testdata
    sdata = f.trialdata
    NT = tdata.numshapes
    NS = sdata.numshapes
    n_test_el = length(tdata.elements_d)
    n_trial_el = length(sdata.elements_d)

    npairs = length(p_data)
    T = eltype(tdata.assembly_d)

    pair_p_d = p_data isa CuArray ? p_data : CuArray(p_data)
    pair_q_d = q_data isa CuArray ? q_data : CuArray(q_data)

    nentries = npairs * NT * NS
    Iz = CUDA.zeros(Int32, nentries)
    Jz = CUDA.zeros(Int32, nentries)
    Vz = CUDA.zeros(T, nentries)

    test_qr, test_shapes = tdata.quadrule_d, tdata.shapes_d
    trial_qr, trial_shapes = sdata.quadrule_d, sdata.shapes_d

    cvs = CommonVertex(f.singularrules.common_vert)
    ces = CommonEdge(f.singularrules.common_edge)
    cfs = CommonFace(f.singularrules.common_face)

    launch_gpu_kernel!(gpu_sparse_block_integrate!, Iz, Jz, Vz, f.biop, npairs, pair_p_d, pair_q_d,
        tdata.elements_d, sdata.elements_d, test_shapes, trial_shapes,
        refspace(f.tfs), refspace(f.bfs), test_qr, trial_qr, cvs, ces, cfs;
        gpu_blocksize=(128), problem_size=(npairs))

    return CuSparseMatrixCOO(Iz, Jz, Vz, (NT * n_test_el, NS * n_trial_el))
end

# Build the sparse element-shape matrix z for all element pairs.
function _sba_zsparse(f::GPUAssembleblockbodyFunctor, p_data, q_data)
    return CuSparseMatrixCSR(_sba_zcoo(f, p_data, q_data))
end

# Element pairs per chunk that keep the intermediate COO near `budget` bytes.
# Chunking this way is both leaner on memory and faster than one giant sort.
function _default_chunkpairs(f::GPUAssembleblockbodyFunctor; budget=1 << 30)
    bytes_per_pair = f.testdata.numshapes * f.trialdata.numshapes *
                     (8 + sizeof(eltype(f.testdata.assembly_d)))
    return max(1, budget ÷ bytes_per_pair)
end

# Bounded-memory sparse-sparse product (CUSPARSE ALG1/DEFAULT runs out of
# resources for large dimensions; ALG2 chunks the intermediate products).
_spgemm(A, B) = CUDA.CUSPARSE.gemm('N', 'N', one(eltype(A)), A, B, 'O',
    CUDA.CUSPARSE.CUSPARSE_SPGEMM_ALG2)

# Integrate + project a (device-resident) element-pair list into Z,
# streaming the pairs in chunks of at most `maxchunkpairs` to bound memory.
# Projection is Z = A_test * z * A_trialᵀ; A_trialᵀ in CSR is the trial
# assembly CSC reinterpreted, i.e. a transpose for free.
function _sba_assemble_pairs(f::GPUAssembleblockbodyFunctor, p_data, q_data;
    maxchunkpairs=_default_chunkpairs(f))
    A_test = CuSparseMatrixCSR(f.testdata.assembly_d)
    adt = f.trialdata.assembly_d
    A_trialT = CuSparseMatrixCSR(adt.colPtr, adt.rowVal, adt.nzVal, (size(adt, 2), size(adt, 1)))

    npairs = length(p_data)
    Z = nothing
    for lo in 1:maxchunkpairs:npairs
        hi = min(lo + maxchunkpairs - 1, npairs)
        z_chunk = _sba_zsparse(f, p_data[lo:hi], q_data[lo:hi])
        partial = _spgemm(A_test, _spgemm(z_chunk, A_trialT))
        Z = Z === nothing ? partial : Z + partial
    end
    return Z
end

"""
    gpu_sparse_blockassemble(f, values, nearvalues; maxchunkpairs) -> CuSparseMatrixCSR

Assemble all near-interaction blocks described by the per-block dof lists
`values` (test) / `nearvalues` (trial) — e.g. the output of
`AdaptiveCrossApproximation.nearinteractions` — in a single batched GPU pass,
returning them as one sparse matrix `Z` (rows = test dofs, cols = trial dofs).
This is the sparse, batched counterpart of the per-block [`gpu_blockassembler`](@ref);
it has nothing to do with the radiated near field.

`f` is the functor returned by [`gpu_blockassembler`](@ref); its precomputed,
GPU-resident per-space assembly data is reused, so no per-block host work is
needed. Individual blocks are slices `Z[values[i], nearvalues[i]]`.

The element-pair work list is streamed in chunks of at most `maxchunkpairs`
pairs (default: as many as keep the intermediate ~1 GiB). Each pair belongs to
exactly one chunk, so `Z` is the exact sum of the per-chunk projections.
Chunking bounds the peak size of the (otherwise dominant) intermediate
element-shape matrix and is also faster than one giant sort, so it is on by
default; raise the budget for fewer/larger chunks or lower it for tighter GPUs.

See the [`gpu_sparse_blockassemble(functors, ...)`](@ref) method for the
multi-device variant.
"""
function gpu_sparse_blockassemble(f::GPUAssembleblockbodyFunctor, values, nearvalues;
    maxchunkpairs=_default_chunkpairs(f))
    CUDA.device!(f.device)
    f.singularrules === nothing && throw(ArgumentError(
        "gpu_sparse_blockassemble needs a Sauter-capable quad strategy (DoubleNumSauterQstrat)."))

    p_data, q_data = _sba_pairs(f, values, nearvalues)
    return _sba_assemble_pairs(f, p_data, q_data; maxchunkpairs=maxchunkpairs)
end

"""
    gpu_sparse_blockassemble_functors(biop, tfs, bfs; quadstrat, devices) -> Vector

Build one [`gpu_blockassembler`](@ref) functor per device id in `devices` (the
per-space assembly data is replicated on each GPU). Pass the result to the
multi-device [`gpu_sparse_blockassemble`](@ref).
"""
function gpu_sparse_blockassemble_functors(biop::IntegralOperator, tfs::Space, bfs::Space;
    quadstrat=BEAST.defaultquadstrat, devices)
    functors = Vector{Any}(undef, length(devices))
    @sync for (i, d) in enumerate(devices)
        Threads.@spawn begin
            functors[i] = gpu_blockassembler(biop, tfs, bfs; quadstrat=quadstrat, device=d)
        end
    end
    return [functors...]
end

"""
    gpu_sparse_blockassemble(functors, values, nearvalues) -> Vector{CuSparseMatrixCSR}

Multi-device sparse block assembly. The global element-pair work list is built
once and deduplicated, then split into disjoint slices — one per functor/device.
Each device integrates and projects its slice independently and in parallel,
returning a partial `Z` on that device. Because the slices are disjoint, the
full result is the exact sum of the partials (apply them additively, e.g. in a
matvec); nothing is ever gathered into a single global matrix.

`functors` is one functor per device (see
[`gpu_sparse_blockassemble_functors`](@ref)); Julia must be started with enough
threads to overlap the devices.
"""
function gpu_sparse_blockassemble(functors::AbstractVector{<:GPUAssembleblockbodyFunctor},
    values, nearvalues; maxchunkpairs=_default_chunkpairs(functors[1]))
    G = length(functors)
    f1 = functors[1]
    CUDA.device!(f1.device)
    f1.singularrules === nothing && throw(ArgumentError(
        "gpu_sparse_blockassemble needs a Sauter-capable quad strategy (DoubleNumSauterQstrat)."))

    p_d, q_d = _sba_pairs(f1, values, nearvalues)
    p_host = Array(p_d)
    q_host = Array(q_d)
    CUDA.unsafe_free!(p_d)
    CUDA.unsafe_free!(q_d)
    npairs = length(p_host)

    partials = Vector{Any}(undef, G)
    @sync for g in 1:G
        Threads.@spawn begin
            f = functors[g]
            CUDA.device!(f.device)
            lo = (g - 1) * npairs ÷ G + 1
            hi = g * npairs ÷ G
            p_slice = CuArray(p_host[lo:hi])
            q_slice = CuArray(q_host[lo:hi])
            partials[g] = _sba_assemble_pairs(f, p_slice, q_slice; maxchunkpairs=maxchunkpairs)
        end
    end
    return [partials...]
end
