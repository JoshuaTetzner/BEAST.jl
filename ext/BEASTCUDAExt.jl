module BEASTCUDAExt

using CUDA
using CUDA.Adapt
using CUDA.CUSPARSE

using BEAST
import BEAST: assemble!, Threading, Operator, Space, IntegralOperator
import BEAST: _integrands, _integrands_gen
import BEAST: LagrangeRefSpace, RTRefSpace, GWPDivRefSpace, GWPCurlRefSpace
using BEAST.CompScienceMeshes
using BEAST.SauterSchwabQuadrature
using BEAST.StaticArrays
using BEAST.SparseArrays
using BEAST.LinearAlgebra
using BEAST.ProgressMeter

Adapt.@adapt_structure CommonVertex
Adapt.@adapt_structure CommonEdge
Adapt.@adapt_structure CommonFace

function Adapt.adapt_structure(to, obj::GWPDivRefSpace{T,Degree}) where {T,Degree}
    GWPDivRefSpace{T,Degree}()
end

function Adapt.adapt_structure(to, obj::GWPCurlRefSpace{T,Degree}) where {T,Degree}
    GWPCurlRefSpace{T,Degree}()
end

include("BEASTCUDAExt/tiling.jl")

include("BEASTCUDAExt/gpu_utils.jl")
include("BEASTCUDAExt/gpu_basis.jl")
include("BEASTCUDAExt/gpu_integrals.jl")
include("BEASTCUDAExt/gpu_assemble_integralop.jl")
include("BEASTCUDAExt/gpu_blockassembler.jl")

end
