using Test
using BEAST
using CompScienceMeshes
using LinearAlgebra
using StaticArrays

@testset "EFIE integrand preserves input precision" begin
    for geometrytype in (Float32, Float64), operatortype in (Float32, Float64),
        widen_coefficients in (false, true)
        mesh = convert(geometrytype, meshicosphere(2, 1.0))
        space = raviartthomas(mesh)
        elements, _, _ = BEAST.assemblydata(space; onlyactives=false)
        testelement, trialelement = first(elements), last(elements)
        point = SVector{2,geometrytype}(0.2, 0.3)
        x = neighborhood(testelement, point)
        y = neighborhood(trialelement, point)
        localspace = refspace(space)
        f, g = localspace(x), localspace(y)
        op = Maxwell3D.singlelayer(; wavenumber=operatortype(2π/4))
        if widen_coefficients
            op = BEAST.MWSingleLayer3D(op.gamma, ComplexF64(op.α), ComplexF64(op.β))
        end
        integrand = BEAST.Integrand(op, localspace, localspace,
            testelement, trialelement)
        value = @inferred integrand(x, y, f, g)
        T = real(promote_type(geometrytype, BEAST.scalartype(op)))
        @test eltype(value) == Complex{T}
        @test all(isfinite, value)

        # Compare with the previous expression, which promoted through Float64 i4pi.
        R = norm(cartesian(x) - cartesian(y))
        green = exp(-op.gamma*R)*(BEAST.i4pi*(1/R))
        alphaG, betaG = op.α*green, op.β*green
        expected = BEAST._integrands(f, g) do fi, gj
            alphaG*dot(fi.value, gj.value) +
                betaG*dot(fi.divergence, gj.divergence)
        end
        @test value ≈ expected rtol=5eps(T)
    end
end
