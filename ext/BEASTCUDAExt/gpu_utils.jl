function launch_gpu_kernel!(gpu_kernel, args...; gpu_blocksize=(32, 32), problem_size)
    if problem_size == 0
        return
    end
    @assert all(gpu_blocksize .> 0) "GPU block size must be positive integers."

    threadsPerBlock = prod(gpu_blocksize)
    @assert threadsPerBlock <= 1024 "GPU block size exceeds maximum threads per block."

    blocks = ceil.(Int, problem_size ./ gpu_blocksize)
    @cuda blocks=blocks threads=gpu_blocksize gpu_kernel(args...)

    return
end
