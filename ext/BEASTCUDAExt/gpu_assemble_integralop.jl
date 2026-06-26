#============================================================#
# AssemblyData
#============================================================#

function l2g_maps!(ad, nfunctions)
    glb_dofs = zeros(Int, nfunctions)

    for index in CartesianIndices(ad.data)
        dof = ad.data[index][1]
        if dof > 0
            glb_dofs[dof] = 1
        end
    end

    tally = accumulate(+, glb_dofs)
    l2g_map = zeros(Int, tally[end])
    localmap = Dict{Int,Int}()
    for (dof, flag) in enumerate(glb_dofs)
        if flag > 0
            local_dof = tally[dof]
            l2g_map[local_dof] = dof
            localmap[dof] = local_dof
        end
    end

    for index in CartesianIndices(ad.data)
        dof, coeff = ad.data[index]
        if haskey(localmap, dof)
            ad.data[index] = (localmap[dof], coeff)
        end
    end

    return l2g_map
end

# Transform assembly data into a format centered around the Dof and transfer it to the GPU.
function load_assemblydata_gpu(X, ::Type{T}) where T
    el, ad, cls = BEAST.assemblydata(X)

    num_shapes = numfunctions(refspace(X), domain(el[1]))

    l2g_map = l2g_maps!(ad, numfunctions(X))

    rows = Int[]
    cols = Int[]
    vals = T[]
    ax = axes(ad.data)
    for i in ax[1]
        for j in ax[2]
            for k in ax[3]
                dof = ad.data[i, j, k][1]
                if dof > 0
                    push!(rows, dof)
                    push!(cols, num_shapes * (k - 1) + j)
                    push!(vals, T(ad.data[i, j, k][2]))
                end
            end
        end
    end

    dof_ad = sparse(rows, cols, vals, length(l2g_map), num_shapes * length(el))

    ad_sparse_d = CuSparseMatrixCSC(dof_ad)
    el_d = CuArray(el)
    return el_d, ad_sparse_d, l2g_map, cls
end

function gpu_triangle_rule(quadrule)
    qrule = CompScienceMeshes.trgauss(quadrule)
    q = Array{Tuple{SVector{2,Float64},Float64}}(undef, length(qrule[2]))
    for (i, a) in enumerate(zip(eachcol(qrule[1]), qrule[2]))
        q[i] = a
    end
    return CuArray(q)
end

function gpu_legendre_rule(order)
    rule = CompScienceMeshes.legendre(order, 0.0, 1.0)
    q = Array{Tuple{Float64,Float64}}(undef, length(rule[2]))
    for (i, a) in enumerate(zip(rule[1], rule[2]))
        q[i] = a
    end
    return CuArray(q)
end



#============================================================#
# Singularity detection
#============================================================#

#Flag singularities based on overlap of test and trial element.
function gpu_singularityflag!(singularity_d, testel_d::CuDeviceVector{S}, trialel_d::CuDeviceVector{S}) where S
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    cols = length(trialel_d)
    rows = length(testel_d)

    if glb_idx <= rows * cols

        T = coordtype(S)

        tol = 1e3 * eps(T)

        i = mod(glb_idx - 1, rows) + 1
        j = div(glb_idx - 1, rows) + 1

        hits = 1
        for t in vertices(testel_d[i])
            for b in vertices(trialel_d[j])
                d = norm(t - b)
                hits += (d < tol)
            end
        end
        @inbounds singularity_d[glb_idx, hits] = true
    end
    return nothing
end

function gpu_singularityflag_indexed!(singularity_d, testel_d::CuDeviceVector{S}, test_ids_d, trialel_d::CuDeviceVector{S}, trial_ids_d) where S
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    rows = length(test_ids_d)
    cols = length(trial_ids_d)

    if glb_idx <= rows * cols
        T = coordtype(S)
        tol = 1e3 * eps(T)

        i = mod(glb_idx - 1, rows) + 1
        j = div(glb_idx - 1, rows) + 1

        hits = 1
        for t in vertices(testel_d[test_ids_d[i]])
            for b in vertices(trialel_d[trial_ids_d[j]])
                d = norm(t - b)
                hits += (d < tol)
            end
        end
        @inbounds singularity_d[glb_idx, hits] = true
    end
    return nothing
end

# Assign quadrature strategy based on singularity flags.
function gpu_quadstrat!(quadstrat_d, singularity_map_d, map_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sing_idx = threadIdx().y

    N = size(singularity_map_d, 1)

    if glb_idx <= N && sing_idx <= 4
        if singularity_map_d[glb_idx, sing_idx] == true
            @inbounds quadstrat_d[map_d[glb_idx, sing_idx], sing_idx] = glb_idx
        end
    end

    return nothing
end

# Assign strat to each element pair
function singularitydetection!(quadstrat_d, numpairs, test_el_d, trial_el_d;
    singularity_map_d=nothing)


    if singularity_map_d === nothing
        singularity_map_d = CUDA.fill(false, length(test_el_d) * length(trial_el_d), 4)
    else
        fill!(singularity_map_d, false)
    end

    launch_gpu_kernel!(gpu_singularityflag!, singularity_map_d, test_el_d, trial_el_d; gpu_blocksize=(256), problem_size=(length(test_el_d) * length(trial_el_d)))


    tally_strats_d = accumulate(+, singularity_map_d, dims=1)

    numpairs .= Array(tally_strats_d[end, :])

    launch_gpu_kernel!(gpu_quadstrat!, quadstrat_d, singularity_map_d, tally_strats_d; gpu_blocksize=(256, 4), problem_size=size(singularity_map_d))

    return
end

function singularitydetection_indexed!(quadstrat_d, numpairs, test_el_d, test_ids_d, trial_el_d, trial_ids_d;
    singularity_map_d=nothing)

    if singularity_map_d === nothing
        singularity_map_d = CUDA.fill(false, length(test_ids_d) * length(trial_ids_d), 4)
    else
        fill!(singularity_map_d, false)
    end

    launch_gpu_kernel!(gpu_singularityflag_indexed!, singularity_map_d, test_el_d, test_ids_d, trial_el_d, trial_ids_d;
        gpu_blocksize=(256), problem_size=(length(test_ids_d) * length(trial_ids_d)))

    tally_strats_d = accumulate(+, singularity_map_d, dims=1)
    numpairs .= Array(tally_strats_d[end, :])

    launch_gpu_kernel!(gpu_quadstrat!, quadstrat_d, singularity_map_d, tally_strats_d;
        gpu_blocksize=(256, 4), problem_size=size(singularity_map_d))

    return
end


#============================================================#
# Shape function evaluation
#============================================================#

function gpu_shapefunction_eval!(shapefunction, el_d, refspace, quadrule)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    qp_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    N = length(el_d)

    if glb_idx <= N && qp_idx <= length(quadrule)

        el = el_d[glb_idx]

        px = quadrule[qp_idx][1]
        mp = neighborhood(el, px)
        val = refspace(mp)
        shapefunction[glb_idx, qp_idx] = val
    end

    return nothing
end

#============================================================#
# Double Num Quadrature
#============================================================#

function gpu_momintegral_doublenum_pair!(zlocal, biop, pair, test_els, trial_els, test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    rows = length(test_els)

    npts_test = length(test_qrule_d)
    npts_trial = length(trial_qrule_d)

    i = mod(pair - 1, rows) + 1
    j = div(pair - 1, rows) + 1

    el_test = test_els[i]
    el_trial = trial_els[j]
    shape_test = view(test_shapes, i, :)
    shape_trial = view(trial_shapes, j, :)
    test_domain = CompScienceMeshes.domain(el_test)
    trial_domain = CompScienceMeshes.domain(el_trial)
    numshapes_test = numfunctions(test_refspace, test_domain)
    numshapes_trial = numfunctions(trial_refspace, trial_domain)

    igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)

    T = eltype(zlocal)
    z = zeros(StaticArrays.SMatrix{numshapes_test,numshapes_trial,T})
    for l in 1:npts_test
        px = test_qrule_d[l][1]
        x = neighborhood(el_test, px)
        wx = test_qrule_d[l][2] * jacobian(x)
        for m in 1:npts_trial
            py = trial_qrule_d[m][1]
            y = neighborhood(el_trial, py)
            wy = trial_qrule_d[m][2] * jacobian(y)

            @inbounds z += wx * wy * igd(x, y, shape_test[l], shape_trial[m])
        end
    end
    view(zlocal, numshapes_test*(i-1)+1:numshapes_test*i, numshapes_trial*(j-1)+1:numshapes_trial*j) .= z

    return nothing
end

function gpu_momintegral_doublenum_pair_indexed!(zlocal, biop, pair, test_els, test_ids_d, trial_els, trial_ids_d, test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    rows = length(test_ids_d)

    active_i = mod(pair - 1, rows) + 1
    active_j = div(pair - 1, rows) + 1
    i = test_ids_d[active_i]
    j = trial_ids_d[active_j]

    npts_test = length(test_qrule_d)
    npts_trial = length(trial_qrule_d)

    el_test = test_els[i]
    el_trial = trial_els[j]
    shape_test = view(test_shapes, i, :)
    shape_trial = view(trial_shapes, j, :)
    test_domain = CompScienceMeshes.domain(el_test)
    trial_domain = CompScienceMeshes.domain(el_trial)
    numshapes_test = numfunctions(test_refspace, test_domain)
    numshapes_trial = numfunctions(trial_refspace, trial_domain)

    igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)

    T = eltype(zlocal)
    z = zeros(StaticArrays.SMatrix{numshapes_test,numshapes_trial,T})
    for l in 1:npts_test
        px = test_qrule_d[l][1]
        x = neighborhood(el_test, px)
        wx = test_qrule_d[l][2] * jacobian(x)
        for m in 1:npts_trial
            py = trial_qrule_d[m][1]
            y = neighborhood(el_trial, py)
            wy = trial_qrule_d[m][2] * jacobian(y)
            @inbounds z += wx * wy * igd(x, y, shape_test[l], shape_trial[m])
        end
    end
    view(zlocal,
        numshapes_test*(active_i-1)+1:numshapes_test*active_i,
        numshapes_trial*(active_j-1)+1:numshapes_trial*active_j) .= z

    return nothing
end

function gpu_momintegral_doublenum!(zlocal, biop, npairs, elpairs, test_els, trial_els, test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if glb_idx <= npairs
        gpu_momintegral_doublenum_pair!(
            zlocal, biop, elpairs[glb_idx], test_els, trial_els,
            test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    end

    return nothing
end

function gpu_momintegral_doublenum_indexed!(zlocal, biop, npairs, elpairs, test_els, test_ids_d, trial_els, trial_ids_d, test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if glb_idx <= npairs
        gpu_momintegral_doublenum_pair_indexed!(
            zlocal, biop, elpairs[glb_idx], test_els, test_ids_d, trial_els, trial_ids_d,
            test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    end

    return nothing
end

function gpu_momintegral_doublenum_allpairs_indexed!(zlocal, biop, npairs, test_els, test_ids_d, trial_els, trial_ids_d, test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if glb_idx <= npairs
        gpu_momintegral_doublenum_pair_indexed!(
            zlocal, biop, glb_idx, test_els, test_ids_d, trial_els, trial_ids_d,
            test_shapes, trial_shapes, test_refspace, trial_refspace, test_qrule_d, trial_qrule_d)
    end

    return nothing
end

#============================================================#
# SauterSchwab Quadrature
#============================================================#

# Calculate set difference A \ B assuming A contains all elements of B.
function gpu_setdifference(A::SVector{N1,T}, B::SVector{N2,T}) where {N1,N2,T}

    out = zeros(StaticArrays.MVector{N1 - N2,Int64})
    l = 1
    for i in eachindex(A)
        found = false
        for j in eachindex(B)
            if A[i] == B[j]
                found = true
            end
        end
        if !found
            @inbounds out[l] = A[i]
            l += 1
        end
    end

    return out
end

function gpu_sauterschwab_reorder(t, s, strat::CommonVertex)
    T = eltype(t[1])
    tol = 1e4 * eps(T)

    # Find the permutation P of t and s that make
    # Pt = [P, A1, A2]
    # Ps = [P, B1, B2]
    I1 = 0
    J1 = 0
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if norm(w - v) < tol
                I1 = i
                J1 = j
                e += 1
                break
            end
        end
        e == 2 && break
    end

    A = SVector{3,Int64}(1, 2, 3)
    a = gpu_setdifference(A, SVector{1,Int64}(I1))
    b = gpu_setdifference(A, SVector{1,Int64}(J1))

    I = SVector{3,Int64}(I1, a...)
    J = SVector{3,Int64}(J1, b...)

    return I, J, nothing, nothing
end


function gpu_sauterschwab_reorder(t, s, strat::CommonEdge)
    T = eltype(t[1])
    tol = 1e3 * eps(T)


    I1, I2 = 0, 0
    J1, J2 = 0, 0
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if norm(w - v) < tol
                if e == 1
                    I1 = i
                    J1 = j
                    e += 1
                elseif e == 2
                    I2 = i
                    J2 = j
                    e += 1
                end
                break
            end
        end
    end

    I = SVector{3,Int64}(I2, 6 - I2 - I1, I1)
    J = SVector{3,Int64}(J2, 6 - J2 - J1, J1)

    return I, J, nothing, nothing
end


function gpu_sauterschwab_reorder(t, s, strat::CommonFace)
    T = eltype(t[1])
    tol = 1e3 * eps(T)

    I = SVector{3,Int64}(1, 2, 3)
    J = zeros(MVector{3,Int64})
    e = 1
    for i in 1:3
        v = t[i]
        for j in 1:3
            w = s[j]
            if norm(w - v) < tol
                J[i] = j
                e += 1
            end
        end
    end

    J = SVector{3,Int64}(J...)

    return I, J, nothing, nothing
end


function gpu_momintegral_sauterschwab!(zlocal, biop, npairs, elpairs, test_els, trial_els, test_refspace, trial_refspace, strat)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x


    if glb_idx <= npairs

        rows = length(test_els)

        i = mod(elpairs[glb_idx] - 1, rows) + 1
        j = div(elpairs[glb_idx] - 1, rows) + 1

        el_test = test_els[i]
        el_trial = trial_els[j]
        test_domain = CompScienceMeshes.domain(el_test)
        trial_domain = CompScienceMeshes.domain(el_trial)
        numshapes_test = numfunctions(test_refspace, test_domain)
        numshapes_trial = numfunctions(trial_refspace, trial_domain)

        I, J, _, _ = gpu_sauterschwab_reorder(CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)

        igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)
        igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)

        T = eltype(zlocal)
        z = zeros(StaticArrays.SMatrix{numshapes_test,numshapes_trial,T})

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
        view(zlocal, numshapes_test*(i-1)+1:numshapes_test*i, numshapes_trial*(j-1)+1:numshapes_trial*j) .= z
    end

    return nothing
end

function gpu_momintegral_sauterschwab_indexed!(zlocal, biop, npairs, elpairs, test_els, test_ids_d, trial_els, trial_ids_d, test_refspace, trial_refspace, strat)
    glb_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if glb_idx <= npairs
        rows = length(test_ids_d)
        active_i = mod(elpairs[glb_idx] - 1, rows) + 1
        active_j = div(elpairs[glb_idx] - 1, rows) + 1

        i = test_ids_d[active_i]
        j = trial_ids_d[active_j]

        el_test = test_els[i]
        el_trial = trial_els[j]
        test_domain = CompScienceMeshes.domain(el_test)
        trial_domain = CompScienceMeshes.domain(el_trial)
        numshapes_test = numfunctions(test_refspace, test_domain)
        numshapes_trial = numfunctions(trial_refspace, trial_domain)

        I, J, _, _ = gpu_sauterschwab_reorder(CompScienceMeshes.vertices(el_test), CompScienceMeshes.vertices(el_trial), strat)

        igd = BEAST.Integrand(biop, test_refspace, trial_refspace, el_test, el_trial)
        igdp = BEAST.pulledback_integrand(igd, I, el_test, J, el_trial)

        T = eltype(zlocal)
        z = zeros(StaticArrays.SMatrix{numshapes_test,numshapes_trial,T})

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
        view(zlocal,
            numshapes_test*(active_i-1)+1:numshapes_test*active_i,
            numshapes_trial*(active_j-1)+1:numshapes_trial*active_j) .= z
    end

    return nothing
end


#============================================================#
# Assemble matrix
#============================================================#


function build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d, trial_proj_d=nothing)

    if trial_proj_d === nothing
        trial_proj_d = CUDA.fill(zero(promote_type(eltype(zlocal_d), eltype(trial_ad_d))), size(trial_ad_d, 1), size(zlocal_d, 1))
    else
        fill!(trial_proj_d, zero(eltype(trial_proj_d)))
    end
    alpha1 = one(eltype(trial_proj_d))
    beta1 = zero(eltype(trial_proj_d))
    CUSPARSE.mm!('N', 'T', alpha1, trial_ad_d, zlocal_d, beta1, trial_proj_d, 'O')
    alpha2 = one(eltype(matrix_d))
    beta2 = zero(eltype(matrix_d))
    CUSPARSE.mm!('N', 'T', alpha2, test_ad_d, trial_proj_d, beta2, matrix_d, 'O')

    return nothing
end


#=======================================#
# BEAST assemble function for GPU threading
#=======================================#

"""
    assemble(operator, test_functions, trial_functions; threading=:gpu, quadstrat, tilingstrat, verbose)

GPU backend for the full (dense) operator matrix. Reached from the BEAST public
API `assemble(op, X, Y; threading=:gpu)`, which dispatches here via
`assemble!(…, Threading{:gpu})`.

Both element grids are uploaded once and optionally split into tiles
(`tilingstrat`, see [`TilingStrategy`](@ref)) that are assembled concurrently on
several CUDA streams — this bounds peak GPU memory for large problems. Each
element pair is classified (regular / common vertex|edge|face) and integrated
with double-numerical or Sauter–Schwab quadrature, then projected to dofs and
stored.

`quadstrat` must be a `DoubleNumSauterQstrat` (regular far + Sauter near); the
BEAST default `DoubleNumWiltonSauterQStrat` is not supported (no GPU Wilton
path). See the extension README for the full support matrix and limitations.
"""
function assemble!(operator::Operator, test_functions::Space, trial_functions::Space,
    store, threading::Type{Threading{:gpu}};
    quadstrat=defaultquadstrat, tilingstrat=TilingStrategy(EqualTiling(1), EqualTiling(1)),
    verbose=false, kwargs...)

    quadstrat = quadstrat(operator, test_functions, trial_functions)
    T = scalartype(operator, test_functions, trial_functions)

    verbose && println("GPU assemble called.")

    test_geo = geometry(test_functions)
    trial_geo = geometry(trial_functions)

    test_splits = split(numcells(test_geo), tilingstrat[1])
    trial_splits = split(numcells(trial_geo), tilingstrat[2])

    if verbose
        println("GPU tiling: $tilingstrat")
        println("test splits: ", first.(test_splits))
        println("trial splits: ", first.(trial_splits))
    end


    test_l2g = Vector{Vector{Int}}(undef, length(test_splits))
    test_ad_qd = Vector{Tuple{Tuple{CuArray,CuSparseMatrixCSC},Tuple{CuArray,CuArray},Vector{Int}}}(undef, length(test_splits))
    trial_l2g = Vector{Vector{Int}}(undef, length(trial_splits))
    trial_ad_qd = Vector{Tuple{Tuple{CuArray,CuSparseMatrixCSC},Tuple{CuArray,CuArray},Vector{Int}}}(undef, length(trial_splits))

    @sync begin
        for i in eachindex(test_splits)
            Threads.@spawn begin
                test_subgeo = CompScienceMeshes.SubMesh(test_geo, test_splits[i])
                test_functions_p = restrict(test_functions, test_subgeo)

                test_l2g[i], test_ad_qd[i] = assemble_primer_gpu(
                    operator, test_functions_p, quadstrat.outer_rule, T; verbose=verbose)
            end
        end


        for i in eachindex(trial_splits)
            Threads.@spawn begin
                trial_subgeo = CompScienceMeshes.SubMesh(trial_geo, trial_splits[i])
                trial_functions_p = restrict(trial_functions, trial_subgeo)

                trial_l2g[i], trial_ad_qd[i] = assemble_primer_gpu(
                    operator, trial_functions_p, quadstrat.inner_rule, T; verbose=verbose)
            end
        end
    end

    cvrule_d = gpu_legendre_rule(quadstrat.sauter_schwab_common_vert)
    cerule_d = gpu_legendre_rule(quadstrat.sauter_schwab_common_edge)
    cfrule_d = gpu_legendre_rule(quadstrat.sauter_schwab_common_face)


    NUM_CUDA_STREAMS = 10
    N = length(test_splits)
    M = length(trial_splits)

    Zbuffer = Channel{Tuple{Matrix{T},Vector{Int},Vector{Int}}}(2 * NUM_CUDA_STREAMS)

    for l in 1:NUM_CUDA_STREAMS
        errormonitor(Threads.@spawn begin
            for m in 1:NUM_CUDA_STREAMS:N*M

                blk = m + (l - 1)
                if blk > N * M
                    continue
                end
                i = mod1(blk, N)
                j = div(blk - 1, N) + 1

                (test_el_d, test_ad_d), test_qd, _ = test_ad_qd[i]

                (trial_el_d, trial_ad_d), trial_qd, _ = trial_ad_qd[j]

                qd_d = (test_qd, trial_qd, cvrule_d, cerule_d, cfrule_d)

                matrix = assemblechunk_body_gpu!(operator,
                    refspace(test_functions), test_el_d, test_ad_d,
                    refspace(trial_functions), trial_el_d, trial_ad_d,
                    qd_d)
                put!(Zbuffer, (matrix, test_l2g[i], trial_l2g[j]))

            end
        end)
    end

    k = N * M
    pbar = BEAST.progressbar(k, true)
    maxk = k
    while k > 0

        (Z, test_l2g_map, trial_l2g_map) = take!(Zbuffer)
        store1(v, m, n) = store(v, test_l2g_map[m], trial_l2g_map[n])
        for j in axes(Z, 2)
            for i in axes(Z, 1)
                store1(Z[i, j], i, j)
            end
        end
        k -= 1

        update!(pbar, maxk - k)
    end
    finish!(pbar)
end

"""
    assemble_primer_gpu(operator, functions, quadrule, ::Type{T}; verbose)
        -> (l2g_map, ((elements_d, assembly_d), (quadrule_d, shapes_d), activecells))

Per-space GPU setup shared by both GPU paths. Uploads the element grid and the
dof→(element, shape) assembly matrix, builds the device quadrature rule, and
precomputes the reference-space shape-function values ([`shapetype`](@ref)) at
every quadrature point. Called once per (sub)space by the full [`assemble!`](@ref)
and, via `GPUSpaceAssemblyData`, by [`gpu_blockassembler`](@ref).
"""
function assemble_primer_gpu(operator::Operator, functions::Space, quadrule, ::Type{T}; verbose=false) where T

    space = refspace(functions)
    verbose && println("GPU refspace: ", space)

    quadrule_d = gpu_triangle_rule(quadrule)

    el_d, ad_d, l2g_map, activecells = load_assemblydata_gpu(functions, T)

    chart = CUDA.@allowscalar domain(el_d[1])
    numshapes = numfunctions(space, chart)

    shapefunction_type = shapetype(space)
    shapes_d = CuArray{SVector{numshapes,shapefunction_type}}(undef, length(el_d), length(quadrule_d))

    launch_gpu_kernel!(gpu_shapefunction_eval!, shapes_d, el_d, space, quadrule_d;
        gpu_blocksize=(64, 4), problem_size=(length(el_d), length(quadrule_d)))


    return l2g_map, ((el_d, ad_d), (quadrule_d, shapes_d), activecells)
end



function assemblechunk_body_gpu_device(operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ad_d::CuSparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ad_d::CuSparseMatrixCSC,
    qd_d)

    test_domain = CUDA.@allowscalar domain(test_el_d[1])
    trial_domain = CUDA.@allowscalar domain(trial_el_d[1])

    numshapes_test = numfunctions(test_space, test_domain)
    numshapes_trial = numfunctions(trial_space, trial_domain)

    T = promote_type(scalartype(operator), eltype(test_ad_d), eltype(trial_ad_d))
    quadstrat_d = CUDA.fill(0, length(test_el_d) * length(trial_el_d), 4)
    singularity_map_d = CUDA.fill(false, length(test_el_d) * length(trial_el_d), 4)
    zlocal_d = CUDA.fill(zero(T), numshapes_test * length(test_el_d), length(trial_el_d) * numshapes_trial)
    matrix_d = CUDA.fill(zero(T), size(test_ad_d, 1), size(trial_ad_d, 1))
    trial_proj_d = CUDA.fill(zero(promote_type(T, eltype(trial_ad_d))), size(trial_ad_d, 1), size(zlocal_d, 1))

    return assemblechunk_body_gpu_device!(
        matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d,
        operator,
        test_space, test_el_d, test_ad_d,
        trial_space, trial_el_d, trial_ad_d,
        qd_d)
end

function assemblechunk_body_gpu_device!(matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d,
    operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ad_d::CuSparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ad_d::CuSparseMatrixCSC,
    qd_d)

    (test_quadrule_d, test_shapes_d), (trial_quadrule_d, trial_shapes_d), cvrule_d, cerule_d, cfrule_d = qd_d

    fill!(quadstrat_d, 0)
    fill!(zlocal_d, zero(eltype(zlocal_d)))
    fill!(matrix_d, zero(eltype(matrix_d)))

    numpairs = zeros(Int, 4)
    singularitydetection!(quadstrat_d, numpairs, test_el_d, trial_el_d;
        singularity_map_d=singularity_map_d)

    launch_gpu_kernel!(gpu_momintegral_doublenum!, zlocal_d, operator, numpairs[1], view(quadstrat_d, :, 1),
        test_el_d, trial_el_d, test_shapes_d, trial_shapes_d, test_space, trial_space, test_quadrule_d, trial_quadrule_d;
        gpu_blocksize=(256), problem_size=(numpairs[1]))

    strategy = CommonVertex(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[2], view(quadstrat_d, :, 2), test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[2]))

    strategy = CommonEdge(cerule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[3], view(quadstrat_d, :, 3), test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[3]))

    strategy = CommonFace(cfrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab!, zlocal_d, operator, numpairs[4], view(quadstrat_d, :, 4), test_el_d, trial_el_d,
        test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[4]))

    build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d, trial_proj_d)

    return matrix_d
end

function assemblechunk_body_gpu_device_indexed!(matrix_d, zlocal_d, quadstrat_d, singularity_map_d, trial_proj_d,
    operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ids_d, test_ad_d::CuSparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ids_d, trial_ad_d::CuSparseMatrixCSC,
    qd_d)

    (test_quadrule_d, test_shapes_d), (trial_quadrule_d, trial_shapes_d), cvrule_d, cerule_d, cfrule_d = qd_d

    fill!(quadstrat_d, 0)
    fill!(zlocal_d, zero(eltype(zlocal_d)))
    fill!(matrix_d, zero(eltype(matrix_d)))

    numpairs = zeros(Int, 4)
    singularitydetection_indexed!(quadstrat_d, numpairs, test_el_d, test_ids_d, trial_el_d, trial_ids_d;
        singularity_map_d=singularity_map_d)

    launch_gpu_kernel!(gpu_momintegral_doublenum_indexed!, zlocal_d, operator, numpairs[1], view(quadstrat_d, :, 1),
        test_el_d, test_ids_d, trial_el_d, trial_ids_d, test_shapes_d, trial_shapes_d, test_space, trial_space, test_quadrule_d, trial_quadrule_d;
        gpu_blocksize=(256), problem_size=(numpairs[1]))

    strategy = CommonVertex(cvrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab_indexed!, zlocal_d, operator, numpairs[2], view(quadstrat_d, :, 2),
        test_el_d, test_ids_d, trial_el_d, trial_ids_d, test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[2]))

    strategy = CommonEdge(cerule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab_indexed!, zlocal_d, operator, numpairs[3], view(quadstrat_d, :, 3),
        test_el_d, test_ids_d, trial_el_d, trial_ids_d, test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[3]))

    strategy = CommonFace(cfrule_d)
    launch_gpu_kernel!(gpu_momintegral_sauterschwab_indexed!, zlocal_d, operator, numpairs[4], view(quadstrat_d, :, 4),
        test_el_d, test_ids_d, trial_el_d, trial_ids_d, test_space, trial_space, strategy;
        gpu_blocksize=(256), problem_size=(numpairs[4]))

    build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d, trial_proj_d)

    return matrix_d
end

function assemblechunk_body_gpu_device_far_indexed!(matrix_d, zlocal_d, trial_proj_d,
    operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ids_d, test_ad_d::CuSparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ids_d, trial_ad_d::CuSparseMatrixCSC,
    qd_d)

    (test_quadrule_d, test_shapes_d), (trial_quadrule_d, trial_shapes_d) = qd_d

    fill!(zlocal_d, zero(eltype(zlocal_d)))
    fill!(matrix_d, zero(eltype(matrix_d)))

    npairs = length(test_ids_d) * length(trial_ids_d)
    launch_gpu_kernel!(gpu_momintegral_doublenum_allpairs_indexed!, zlocal_d, operator, npairs,
        test_el_d, test_ids_d, trial_el_d, trial_ids_d, test_shapes_d, trial_shapes_d,
        test_space, trial_space, test_quadrule_d, trial_quadrule_d;
        gpu_blocksize=(256), problem_size=(npairs))

    build_matrix!(matrix_d, zlocal_d, test_ad_d, trial_ad_d, trial_proj_d)

    return matrix_d
end

function assemblechunk_body_gpu!(operator::IntegralOperator,
    test_space, test_el_d::CuArray, test_ad_d::CuSparseMatrixCSC,
    trial_space, trial_el_d::CuArray, trial_ad_d::CuSparseMatrixCSC,
    qd_d)

    return Array(assemblechunk_body_gpu_device(
        operator,
        test_space, test_el_d, test_ad_d,
        trial_space, trial_el_d, trial_ad_d,
        qd_d))
end
