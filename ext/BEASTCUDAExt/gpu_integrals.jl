function _integrands_gen(::Type{U}, ::Type{V}) where {U<:NamedTuple, V<:NamedTuple}
    return :(f(a, b))
end

@generated function _integrands(f, a::NamedTuple{T}, b::NamedTuple{S}) where {T,S}
    return _integrands_gen(a, b)
end
