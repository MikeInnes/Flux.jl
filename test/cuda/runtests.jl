using Flux, Test, CUDA
using Zygote
using Zygote: pullback

@info "Testing GPU Support"
CUDA.allowscalar(false)

function gpu_gradtest(f, args...)
  args_gpu = gpu.(args)

  l_cpu, back_cpu = pullback((x...) -> f(x...), args...)
  g_cpu = back_cpu(1f0)[1]

  l_gpu, back_gpu = pullback((x...) -> f(x...), args_gpu...)
  g_gpu = back_gpu(1f0)[1]

  @test l_cpu ≈ l_gpu   rtol=1e-4 atol=1e-4
  @test g_gpu isa CuArray
  @test g_cpu ≈ collect(g_gpu)   rtol=1e-4 atol=1e-4
end

@testset "Moving Zeros to GPU" begin
  z = Flux.Zeros()
  z2 = Flux.Zeros(3,3)
  @test z === gpu(z)
  @test z2 === gpu(z2)
end

include("test_utils.jl")
include("cuda.jl")
include("losses.jl")
include("layers.jl")

if CUDA.has_cudnn()
  @info "Testing Flux/CUDNN"
  include("cudnn.jl")
  include("curnn.jl")
else
  @warn "CUDNN unavailable, not testing GPU DNN support"
end
