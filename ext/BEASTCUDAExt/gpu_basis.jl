
shapetype(::RTRefSpace{T}) where T =  @NamedTuple{value::SVector{3,T},divergence::T}

shapetype(::LagrangeRefSpace{T,D,3}) where {T,D} =  @NamedTuple{value::T,curl::SVector{3,T}}

shapetype(::LagrangeRefSpace{T,D,2}) where {T,D} =  @NamedTuple{value::T,derivative::T}

shapetype(::LagrangeRefSpace{T,D,4}) where {T,D} =  @NamedTuple{value::T,gradient::SVector{3,T}}

shapetype(::GWPDivRefSpace{T,D}) where {T,D} =  @NamedTuple{value::SVector{3,T},divergence::T}
