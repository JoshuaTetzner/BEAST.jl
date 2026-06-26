# GPU test suite for the BEASTCUDAExt extension.
#
# All test items here are tagged `:gpu` and are EXCLUDED from the default test
# run (see test/runtests.jl). They require a CUDA-capable device and `CUDA` in
# the active environment. To run them, set BEAST_TEST_GPU=1 (and make sure CUDA
# is available), e.g.
#
#     BEAST_TEST_GPU=1 julia --project=test test/runtests.jl
#
# Pick the device with BEAST_GPU_DEVICE (default 0).

@testitem "GPU environment smoke" tags = [:gpu] begin
    using CUDA
    using Test

    @test CUDA.functional()
    @test length(CUDA.devices()) >= 1

    println("CUDA.functional() = ", CUDA.functional())
    println("CUDA runtime      = ", CUDA.runtime_version())
    println("CUDA driver       = ", CUDA.driver_version())
    println("CUDA devices      = ", length(CUDA.devices()))

    for dev in CUDA.devices()
        CUDA.device!(dev)
        println("  device ", CUDA.deviceid(dev), ": ", CUDA.name(dev),
            ", capability=", CUDA.capability(dev),
            ", totalmem_GiB=", round(CUDA.totalmem(dev) / 2.0^30; digits=2))
    end

    device_id = begin
        value = get(ENV, "BEAST_GPU_DEVICE", "")
        isempty(value) ? 0 : parse(Int, value)
    end
    CUDA.device!(device_id)
    x = CUDA.fill(Float32(device_id + 1), 1_000_000)
    y = similar(x)
    CUDA.@sync y .= 3.0f0 .* x .- 1.0f0
    expected = Float32(3 * (device_id + 1) - 1) * length(y)
    @test sum(y) ≈ expected rtol = 1.0f-5
    CUDA.unsafe_free!(x)
    CUDA.unsafe_free!(y)
end
