
"""
    Tiling

How to split an element grid into tiles for GPU assembly. Concrete types:
[`EqualTiling`](@ref) (`N` tiles), [`WorksizeTiling`](@ref) (fixed tile size),
[`IndexTiling`](@ref) (explicit indices). A test/trial pair of these is wrapped
in a [`TilingStrategy`](@ref) and passed to `assemble(…; threading=:gpu,
tilingstrat=…)`. `split(workloadsize, ::Tiling)` returns the tile index vectors.
"""
abstract type Tiling end

"`IndexTiling(indices)`: one tile with the given explicit element indices. See [`Tiling`](@ref)."
struct IndexTiling <: Tiling
    indices::Vector{Int}
end

"`EqualTiling(N)`: split the grid into `N` (near-)equal tiles. See [`Tiling`](@ref)."
struct EqualTiling <: Tiling
    N::Int
end

"`WorksizeTiling(n)`: split the grid into tiles of at most `n` elements. See [`Tiling`](@ref)."
struct WorksizeTiling <: Tiling
    workpackagesize::Int
end


function split(workloadsize::Int, tiling::IndexTiling)
    return tiling.indices
end

function split(workloadsize::Int, tiling::EqualTiling)
    len = workloadsize
    N = tiling.N
    splits = [collect(s:min(s+ceil(Int,len/N)-1, len)) for s in 1:ceil(Int,len/N):len]
    return splits
end

function split(workloadsize::Int, tiling::WorksizeTiling)
    len = workloadsize
    N = tiling.workpackagesize
    splits = [collect(s:min(s+N-1, len)) for s in 1:N:len]
    return splits
end



"""
    TilingStrategy(test_tiling, trial_tiling)

Pair of [`Tiling`](@ref)s consumed by `assemble(…; threading=:gpu,
tilingstrat=…)`. The test and trial element grids are split with
`test_tiling` / `trial_tiling` and the resulting tiles are assembled
concurrently, bounding peak GPU memory for large problems. The default,
`TilingStrategy(EqualTiling(1), EqualTiling(1))`, is a single tile (no tiling).
"""
struct TilingStrategy{N}
    tiling::NTuple{N,Tiling}
end

Base.getindex(tstrat::TilingStrategy{N}, i::Int) where N = tstrat.tiling[i]

TilingStrategy(args...) = TilingStrategy{length(args)}(NTuple(args)) 
