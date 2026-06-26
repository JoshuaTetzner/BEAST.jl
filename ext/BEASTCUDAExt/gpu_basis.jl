
"""
    shapetype(refspace) -> Type

The `NamedTuple` type produced by evaluating `refspace` at a quadrature point on
the GPU (the `value` plus its `curl` / `divergence` / `gradient` / `derivative`,
depending on the space). The GPU shape-function precomputation in
[`assemble_primer_gpu`](@ref) preallocates `CuArray`s of this isbits type, so
**every reference space used on the GPU needs a method here** — add one to
support a new space.
"""
shapetype(::RTRefSpace{T}) where T =  @NamedTuple{value::SVector{3,T},divergence::T}

shapetype(::LagrangeRefSpace{T,D,3}) where {T,D} =  @NamedTuple{value::T,curl::SVector{3,T}}

shapetype(::LagrangeRefSpace{T,D,2}) where {T,D} =  @NamedTuple{value::T,derivative::T}

shapetype(::LagrangeRefSpace{T,D,4}) where {T,D} =  @NamedTuple{value::T,gradient::SVector{3,T}}

shapetype(::GWPDivRefSpace{T,D}) where {T,D} =  @NamedTuple{value::SVector{3,T},divergence::T}
