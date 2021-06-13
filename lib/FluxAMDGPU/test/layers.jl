# Test layers and data/model movements on and off the GPU
# Add tests for layers and their gradients on the GPU
# Most of the forward passes should be fine being applied
# to bitstype objects, but this gives higher coverage for our use-cases
# Check that getting the gradients does not throw

# generic movement tests
@testset "Basic GPU Movement" begin
  @test gradient(x -> sum(gpu(x)), rand(3,3)) isa Tuple
  @test gradient(x -> sum(cpu(x)), gpu(rand(3,3))) isa Tuple
end

# TODO: These layers get into scalar indexing
# `AlphaDropout` throws a compilation error on GPUs,
# whereas, the rest are scalar indexing issues.
const BROKEN_LAYERS = Union{DepthwiseConv,
                            AlphaDropout}

function gpu_gradtest(name::String, layers::Vector, x_cpu, args...;
            setmode=false, test_cpu=true, rtol=1e-5, atol=1e-5)
  @testset "$name GPU grad tests" begin
    for layer in layers
      @testset "$layer GPU grad test" begin
        l_cpu = layer(args...)
        if l_cpu isa BROKEN_LAYERS
          l_gpu, x_gpu = l_cpu |> gpu, x_cpu |> gpu
          @test_broken gradient(() -> sum(l_gpu(x_gpu)), Flux.params(l_gpu)) isa Flux.Zygote.Grads
        else
          gpu_autodiff_test(l_cpu, x_cpu,
              test_equal=test_cpu, rtol=rtol, atol=atol)
          if setmode
            testmode!(l_cpu)
            gpu_autodiff_test(l_cpu, x_cpu,
              test_equal=test_cpu, rtol=rtol, atol=atol)
          end
        end
      end
    end
  end
end


# Just to give testset in gradtest meaningful labels
ConvNoBias(args...) = Conv(args...; bias=false)
ConvTransposeNoBias(args...) = ConvTranspose(args...; bias=false)
CrossCorNoBias(args...) = CrossCor(args...; bias=false)
DepthwiseConvNoBias(args...) = DepthwiseConv(args...; bias=false)
r = rand(Float32, 28, 28, 1, 1)
conv_layers = [Conv, ConvNoBias, ConvTranspose, ConvTransposeNoBias, CrossCor, CrossCorNoBias, DepthwiseConv, DepthwiseConvNoBias]
gpu_gradtest("Conv", conv_layers, r, (2,2), 1=>3)

pooling_layers = [MaxPool, MeanPool]
gpu_gradtest("Pooling", pooling_layers, r, (2,2))

adaptive_pooling_layers = [AdaptiveMaxPool, AdaptiveMeanPool]
gpu_gradtest("AdaptivePooling", adaptive_pooling_layers, r, (7,7))

dropout_layers = [Dropout, AlphaDropout]
gpu_gradtest("Dropout", dropout_layers, r, 0.5f0; test_cpu=false, setmode=true) # dropout is not deterministic

layer_norm = [i -> LayerNorm(i; affine=false), i -> LayerNorm(i; affine=true)]
gpu_gradtest("LayerNorm 1", layer_norm, rand(Float32, 8, 8, 3, 4), 8)
gpu_gradtest("LayerNorm 2", layer_norm, rand(Float32, 8, 8, 3, 4), (8,8))
gpu_gradtest("LayerNorm 3", layer_norm, rand(Float32, 5, 4), 5)

batch_norm = [BatchNorm]
gpu_gradtest("BatchNorm 3d", batch_norm, rand(Float32, 8, 8, 8, 3, 4), 3, setmode=false) # bug in CUDA.jl with gradient in testmode
gpu_gradtest("BatchNorm 2d", batch_norm, rand(Float32, 8, 8, 3, 4), 3, setmode=false) # bug in CUDA.jl with gradient in testmode
gpu_gradtest("BatchNorm 1d", batch_norm, rand(Float32, 8, 3, 4), 3, setmode=false) # bug in CUDA.jl with gradient in testmode
gpu_gradtest("BatchNorm fullyconn", batch_norm, rand(Float32, 5,4), 5, setmode=false)

instancenorm = [i -> InstanceNorm(i; affine=false), i -> InstanceNorm(i; affine=true)]
gpu_gradtest("InstanceNorm 3d", instancenorm, rand(Float32, 8, 8, 8, 3, 4), 3, setmode=true)
gpu_gradtest("InstanceNorm 2d", instancenorm, rand(Float32, 8, 8, 3, 4), 3, setmode=true)
gpu_gradtest("InstanceNorm 1d", instancenorm, rand(Float32, 8, 3, 4), 3, setmode=true)

groupnorm = [(i, j) -> GroupNorm(i, j; affine=false), (i, j) -> GroupNorm(i, j; affine=true)]
gpu_gradtest("GroupNorm 3d", groupnorm, rand(Float32, 8, 8, 8, 12, 4), 12, 3, setmode=true)
gpu_gradtest("GroupNorm 2d", groupnorm, rand(Float32, 8, 8, 12, 4), 12, 3, setmode=true)
gpu_gradtest("GroupNorm 1d", groupnorm, rand(Float32, 8, 3, 12, 4), 12, 3, setmode=true)

upsample = [x -> Upsample(scale=x)]
gpu_gradtest("Upsample 2d", upsample, rand(Float32, 3, 4, 2, 3), (2,2))
gpu_gradtest("Upsample 1d", upsample, rand(Float32, 3, 4, 2, 3), (2,))

pixelshuffle = [PixelShuffle]
gpu_gradtest("PixelShuffle 2d", pixelshuffle, rand(Float32, 3, 4, 18, 3), 3)
gpu_gradtest("PixelShuffle 1d", pixelshuffle, rand(Float32, 3, 18, 3), 3)


@testset "function layers" begin
  x = rand(Float32, 3,3)
  gpu_autodiff_test(x -> sum(Flux.normalise(x; dims=1)), x)
  gpu_autodiff_test(x -> sum(Flux.normalise(x; dims=2)), x)
  gpu_autodiff_test(x -> sum(Flux.normalise(x)), x)
end

@testset "BatchNorm mix stuff" begin
  m_cpu = BatchNorm(2)
  m_gpu = m_cpu |> gpu
  x_cpu = rand(Float32, 3, 2, 2)
  x_gpu = x_cpu |> gpu

  ## In :auto mode, track statistics only in gradient contest
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test m_cpu.μ ≈ μ_cpu
  gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  @test !(m_cpu.μ ≈ μ_cpu)

  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test m_gpu.μ ≈ μ_gpu
  gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
  @test !(m_gpu.μ ≈ μ_gpu)

  @test Array(m_gpu.μ) ≈ m_cpu.μ

  ## In testmode, never track statistics
  testmode!(m_cpu)
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test m_cpu.μ ≈ μ_cpu
  gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  @test m_cpu.μ ≈ μ_cpu

  testmode!(m_gpu)
  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test m_gpu.μ ≈ μ_gpu
  gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
  @test m_gpu.μ ≈ μ_gpu

  ## In trainmode, always track statistics
  trainmode!(m_cpu)
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test !(m_cpu.μ ≈ μ_cpu)
  μ_cpu = copy(m_cpu.μ)
  gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  @test !(m_cpu.μ ≈ μ_cpu)

  trainmode!(m_gpu)
  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test !(m_gpu.μ ≈ μ_gpu)
  μ_gpu = copy(m_gpu.μ)
  gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
  @test !(m_gpu.μ ≈ μ_gpu)

  ## No errors if input type mistmatch
  x_cpu = rand(Float64, 3, 2, 2)
  x_gpu = x_cpu |> gpu
  m_cpu(x_cpu)
  gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  m_gpu(x_gpu)
  gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
end

@testset "Zeros mapped for $cl" for cl in (Conv, ConvTranspose, CrossCor, DepthwiseConv)
  l = cl((2,2), 1=>3, bias = false) |> gpu
  ip = zeros(Float32, 28,28,1,1) |> gpu
  if l isa BROKEN_LAYERS
    @test_broken sum(l(ip)) ≈ 0.f0
    @test_broken gradient(() -> sum(l(ip)), Flux.params(l)) isa Flux.Zygote.Grads
  else
    @test sum(l(ip)) ≈ 0.f0
    gs = gradient(() -> sum(l(ip)), Flux.params(l))
    @test l.bias ∉ gs.params
  end
end

@testset "Dense with Zeros bias" begin
  l = Dense(ones(Float32, 4,3), Flux.Zeros()) |> gpu
  ip = zeros(Float32, 3, 7) |> gpu

  @test sum(l(ip)) ≈ 0.f0
  gs = gradient(() -> sum(l(ip)), Flux.params(l))
  @test l.b ∉ gs.params
end

@testset "Two-streams Bilinear" begin
  x = zeros(Float32,10,9) |> gpu
  y = zeros(Float32,2,9) |> gpu
  b = Flux.Bilinear(10, 2, 3) |> gpu
  @test size(b(x,y)) == (3,9)
  @test sum(abs2, b(x,y)) ≈ 0f0
  gs_gpu = gradient(() -> sum(abs2.(b(x, y))), params(b))
  b_cpu, x_cpu, y_cpu = b |> cpu, x |> cpu, y |> cpu
  gs_cpu = gradient(() -> sum(abs2.(b_cpu(x_cpu, y_cpu))), params(b_cpu))
  for (pgpu, pcpu) in zip(params(b), params(b_cpu))
    @test gs_cpu[pcpu] ≈ Array(gs_gpu[pgpu])
  end
end

@testset "Parallel" begin
  @testset "zero sum" begin
    input = randn(10, 10, 10, 10) |> gpu
    layer_gpu = Parallel(+, zero, identity) |> gpu
    @test layer_gpu(input) == input
    @test layer_gpu(input) isa ROCArray
  end

  @testset "vararg input" begin
    inputs = (randn(10), randn(5), randn(4)) .|> gpu
    layer = Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2)) |> gpu
    @test size(layer(inputs)) == (2,)
  end

  @testset "gradient" begin
    input_cpu = randn(10, 10, 10, 10)
    input_gpu = input_cpu |> gpu
    layer_cpu = Parallel(+, x -> zero(x), identity)
    layer_gpu = layer_cpu |> gpu
    gs_cpu = gradient(() -> sum(abs2.(layer_cpu(input_cpu))), params(layer_cpu))
    gs_gpu = gradient(() -> sum(abs2.(layer_gpu(input_gpu))), params(layer_gpu))
    for (pgpu, pcpu) in zip(params(layer_cpu), params(layer_gpu))
      @test gs_cpu[pcpu] ≈ gs_gpu[pgpu]
    end
  end
end
