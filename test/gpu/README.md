# GPU tests (BEASTCUDAExt)

Test items validating the CUDA extension. Every item here is a `@testitem`
tagged `:gpu`, auto-discovered by `@run_package_tests` in `test/runtests.jl`.

These tests are **opt-in**: they need a CUDA-capable device and `CUDA` available
in the active environment, so they are skipped in the default run (same mechanism
as the `:example` / `:diagnostics` tags).

## Running

```sh
# all tests, including :gpu
BEAST_TEST_GPU=1 julia --project=test test/runtests.jl
```

`CUDA` must be resolvable in the active environment (it is only a weakdep of
BEAST and is not in `test/Project.toml`). Run from an environment where CUDA is
installed, or add it to the test environment.

Environment variables:

- `BEAST_TEST_GPU=1` — include the `:gpu` test items in the run.
- `BEAST_GPU_DEVICE` — CUDA device index to use (default `0`).

Ground truth throughout is the CPU assembly of the same operator / space /
quadrature (validated by the rest of the BEAST test suite).

## Files

- `test_gpu_environment.jl` — CUDA device / functional smoke.
- `test_gpu_assemble.jl` — full-matrix GPU assembly vs CPU (Lagrange / RT / BC), incl. tiling.
- `test_gpu_blockassembler.jl` — per-block GPU functor vs CPU slice (near, far, subset, view).
- `test_gpu_sparse_blockassembler.jl` — batched sparse block assembler for near
  interactions of fast methods (single + multi device). Not the radiated near field.
