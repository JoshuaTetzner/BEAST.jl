# BEASTCUDAExt — GPU assembly for BEAST

CUDA backend for assembling boundary-element operator matrices. It is a package
extension (weakdep on `CUDA`): load `CUDA` in a session where it is available and
the methods below light up. Names that are not exported are reached through
`Base.get_extension(BEAST, :BEASTCUDAExt)`.

```julia
using CUDA, BEAST
ext = Base.get_extension(BEAST, :BEASTCUDAExt)
```

## What it provides

Three assembly paths, all validated against the CPU assembly as ground truth
(see `test/gpu/`):

1. **Full dense matrix** — `assemble(op, X, Y; threading=:gpu)`.
   Element grids are uploaded once, optionally split into tiles
   (`tilingstrat`, see [`TilingStrategy`]) assembled concurrently on several CUDA
   streams to bound peak memory. Entry point: `assemble!(…, Threading{:gpu})`.

2. **Per-block assembler** — `ext.gpu_blockassembler(op, X, Y; quadstrat, device)`
   returns a callable `f(block_d, testids, trialids)` that fills a preallocated
   device matrix (or a `@view`) with one dof sub-block. This is the GPU
   counterpart of `BEAST.blockassembler` and the building block for fast methods
   when the **block structure** must be preserved.

3. **Sparse block assembler** — `ext.gpu_sparse_blockassemble(f, values, nearvalues)`
   batches all the **near interactions** of a fast method (H-matrix / ACA — the
   dense near blocks from e.g. `AdaptiveCrossApproximation.nearinteractions`)
   into a single sparse matrix in one GPU pass. `f` is a `gpu_blockassembler`
   functor; the element-pair work list is built and deduplicated on the GPU and
   streamed in memory-bounded chunks. **Unrelated to the radiated near field.**
   A multi-device variant (`gpu_sparse_blockassemble_functors` +
   `gpu_sparse_blockassemble(functors, …)`) splits the deduplicated pair list
   disjointly across GPUs and returns per-device partials to be summed.

## File layout

| File | Contents |
|------|----------|
| `BEASTCUDAExt.jl` | module: imports, `Adapt` rules, includes |
| `tiling.jl` | `Tiling` types + `TilingStrategy` for the full assembler |
| `gpu_utils.jl` | `launch_gpu_kernel!` (grid/block sizing) |
| `gpu_basis.jl` | `shapetype` — per-refspace shape NamedTuple (extension point) |
| `gpu_integrals.jl` | `_integrands` GPU specialization |
| `gpu_assemble_integralop.jl` | full assembly: setup, singularity detection, double-num + Sauter–Schwab kernels, `assemble!` |
| `gpu_blockassembler.jl` | `gpu_blockassembler` + `GPUAssembleblockbodyFunctor` |
| `gpu_sparse_blockassembler.jl` | batched sparse block assembler for near interactions |

## Support matrix (validated)

| | Lagrange | RaviartThomas | BuffaChristiansen |
|--|--|--|--|
| Helmholtz3D single layer | ✓ | ✓ | ✓ |
| Maxwell3D single layer | ✓ | ✓ | ✓ |

Across all three paths (full / block / sparse-block). Quadrature: explicit
`DoubleNumSauterQstrat` (near+far) and `DoubleNumQStrat` (far blocks only).

## Limitations / what is missing

- **Default quadrature unsupported.** `defaultquadstrat` for Helmholtz3D/Maxwell3D
  is `DoubleNumWiltonSauterQStrat`; there is no GPU Wilton path, and `assemble!`
  reads `quadstrat.outer_rule`/`.inner_rule`, which that strategy does not have.
  A `DoubleNumSauterQstrat` must be passed explicitly.
- **3D triangles only.** Kernels hardcode triangular elements (`trgauss`, 3
  vertices). Helmholtz2D (segment geometry, 1-D Sauter–Schwab) is not supported.
- **Conforming meshes only** for the block assembler (no refinement/nonconforming
  quad strategies).
- **Operators beyond single layer** (double layer, hypersingular) are untested on
  the GPU `Integrand` path; some may need their kernels made GPU-isbits.
- **New reference spaces** need a [`shapetype`] method.

## Testing

GPU tests live in `test/gpu/` as `@testitem`s tagged `:gpu`, opt-in via
`BEAST_TEST_GPU=1` (CUDA must be available; it is not in `test/Project.toml`).
Pick the device with `BEAST_GPU_DEVICE`. See `test/gpu/README.md`.

Examples: `examples/gpu_sparse_blockassembly.jl` (single + multi device),
`examples/efie_gpu.jl`.
