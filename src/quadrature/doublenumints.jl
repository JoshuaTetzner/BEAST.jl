struct DoubleQuadRule{P,Q}
  outer_quad_points::P
  inner_quad_points::Q
end

@inline _workintegrand(igd, x, y, f, g, ::Type) = igd(x,y,f,g)
@inline _workweighttype(igd, ::Type, ::Type{W}) where {W} = W


"""
integrate!(biop, tshs, bshs, tcell, bcell, interactions, strat)

Function for the computation of moment integrals using simple double quadrature.
"""
function integrate!(biop,
    tshs, bshs, tcell, bcell, z, strat::DoubleQuadRule)

    igd = Integrand(biop, tshs, bshs, tcell, bcell)

    womps = strat.outer_quad_points
    wimps = strat.inner_quad_points
    
    for womp in womps
        tgeo = womp.point
        tvals = womp.value
        M = length(tvals)
        jx = womp.weight
        
        for wimp in wimps
            bgeo = wimp.point
            bvals = wimp.value
            N = length(bvals)
            jy = wimp.weight

            W = _workweighttype(igd,eltype(z),promote_type(typeof(jx),typeof(jy)))
            j = W(jx) * W(jy)

            z1 = j * _workintegrand(igd,tgeo,bgeo,tvals,bvals,eltype(z))
            for n in 1:N
                for m in 1:M
                    z[m,n] += z1[m,n]
            end end
        end
    end

    return z
end


_TransposedStrat(a::DoubleQuadRule) = a
