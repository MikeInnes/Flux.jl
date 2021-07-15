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
# The norm layers behave differently on the CPU and
# the GPU too.
const BROKEN_LAYERS = Union{DepthwiseConv,
                            AlphaDropout}

const ACTIVATIONS = [identity, relu, tanh,
                     sigmoid, exp, softplus,
                     elu, selu]

function gpu_gradtest(name::String, layers::Vector, x_cpu = nothing, args...; test_cpu = true)
  isnothing(x_cpu) && error("Missing input to test the layers against.")
  @testset "$name GPU grad tests" begin
    for layer in layers
      @testset "$layer Layer GPU grad test" begin

        # compute output and grad of parameters
        l_cpu = layer(args...)
        ps_cpu = Flux.params(l_cpu)
        y_cpu, back_cpu = pullback(() -> sum(l_cpu(x_cpu)), ps_cpu)
        gs_cpu = back_cpu(1f0)

        x_gpu = gpu(x_cpu)
        l_gpu = l_cpu |> gpu
        ps_gpu = Flux.params(l_gpu)

        if typeof(l_gpu) <: BROKEN_LAYERS
          @test_broken gradient(() -> sum(l_gpu(x_gpu)), ps_gpu) isa Flux.Zygote.Grads
        else
          y_gpu, back_gpu = pullback(() -> sum(l_gpu(x_gpu)), ps_gpu)
          gs_gpu = back_gpu(1f0) # TODO many layers error out when backprop int 1, should fix

          # compute grad of input
          xg_cpu = gradient(x -> sum(l_cpu(x)), x_cpu)[1]
          xg_gpu = gradient(x -> sum(l_gpu(x)), x_gpu)[1]

          # test 
          if test_cpu
            @test y_gpu ≈ y_cpu rtol=1f-3 atol=1f-3
            if isnothing(xg_cpu)
              @test isnothing(xg_gpu)
            else
              @test Array(xg_gpu) ≈ xg_cpu rtol=1f-3 atol=1f-3
            end
          end
          @test gs_gpu isa Flux.Zygote.Grads
          for (p_cpu, p_gpu) in zip(ps_cpu, ps_gpu)
            if isnothing(gs_cpu[p_cpu])
              @test isnothing(gs_gpu[p_gpu])
            else
              @test gs_gpu[p_gpu] isa Flux.CUDA.CuArray
              if test_cpu
                @test Array(gs_gpu[p_gpu]) ≈ gs_cpu[p_cpu] rtol=1f-3 atol=1f-3
              end
            end
          end
        end
      end
    end
  end
end

# Just to give testset in gpu_gradtest meaningful labels
ConvNoBias(args...) = Conv(args...; bias = false)
ConvTransposeNoBias(args...) = ConvTranspose(args...; bias = false)
CrossCorNoBias(args...) = CrossCor(args...; bias = false)
DepthwiseConvNoBias(args...) = DepthwiseConv(args...; bias = false)

for act in ACTIVATIONS
  r = rand(Float32, 28, 28, 1, 1)
  conv_layers = [Conv, ConvNoBias,
                 ConvTranspose, ConvTransposeNoBias,
                 CrossCor, CrossCorNoBias,
                 DepthwiseConv, DepthwiseConvNoBias]
  gpu_gradtest("Convolution with $act", conv_layers, r, (2,2), 1=>3, act, test_cpu = false)
  
  batch_norm = [BatchNorm]
  gpu_gradtest("BatchNorm 1 with $act", batch_norm, rand(Float32, 28,28,3,4), 3, act, test_cpu = false) #TODO fix errors
  gpu_gradtest("BatchNorm 2 with $act", batch_norm, rand(Float32, 5,4), 5, act, test_cpu = false)
  
  instancenorm = [InstanceNorm]
  gpu_gradtest("InstanceNorm with $act", instancenorm, r, 1, act, test_cpu = false)
  
  groupnorm = [GroupNorm]
  gpu_gradtest("GroupNorm with $act", groupnorm, rand(Float32, 28,28,3,1), 3, 1, act, test_cpu = false)
end

r = rand(Float32, 28, 28, 1, 1)

pooling_layers = [MaxPool, MeanPool]
gpu_gradtest("Pooling", pooling_layers, r, (2,2))

adaptive_pooling_layers = [AdaptiveMaxPool, AdaptiveMeanPool]
gpu_gradtest("AdaptivePooling", adaptive_pooling_layers, r, (7,7), test_cpu = false)

dropout_layers = [Dropout, AlphaDropout]
gpu_gradtest("Dropout", dropout_layers, r, 0.5f0; test_cpu = false) # dropout is not deterministic

layer_norm = [LayerNorm]
gpu_gradtest("LayerNorm 1", layer_norm, rand(Float32, 28,28,3,4), 1, test_cpu = false) #TODO fix errors
gpu_gradtest("LayerNorm 2", layer_norm, rand(Float32, 5,4), 5)

upsample = [x -> Upsample(scale=x)]
gpu_gradtest("Upsample 2d", upsample, rand(Float32, 3, 4, 2, 3), (2,2))
gpu_gradtest("Upsample 1d", upsample, rand(Float32, 3, 4, 2, 3), (2,))

pixelshuffle = [PixelShuffle]
gpu_gradtest("PixelShuffle 2d", pixelshuffle, rand(Float32, 3, 4, 18, 3), 3)
gpu_gradtest("PixelShuffle 1d", pixelshuffle, rand(Float32, 3, 18, 3), 3)

embedding = [Flux.Embedding]
gpu_gradtest("Embedding", embedding, [1,3,5], 5 => 2)
gpu_gradtest("Embedding repeated indices", embedding, rand(1:50, 10^6), 50 => 2)
gpu_gradtest("Embedding integer index", embedding, 1, 5 => 2)
gpu_gradtest("Embedding 2d index", embedding, [1 2; 3 4], 5 => 2)
gpu_gradtest("Embedding OneHotVec index", embedding, OneHotVector(1, 5), 5 => 2)
gpu_gradtest("Embedding OneHotMatrix index", embedding,  OneHotMatrix([1,2,3], 5), 5 => 2)
gpu_gradtest("Embedding OneHotMatrix repeated indices", embedding, OneHotMatrix(rand(1:50, 10^6), 50), 50 => 2)

@testset "function layers" begin
  x = rand(Float32, 3,3)
  gpu_autodiff_test(x -> sum(Flux.normalise(x; dims=1)), x)
  gpu_autodiff_test(x -> sum(Flux.normalise(x; dims=2)), x)
  gpu_autodiff_test(x -> sum(Flux.normalise(x)), x)
end

@testset "Zeros mapped for $cl" for cl in (Conv, ConvTranspose, CrossCor, DepthwiseConv)
  l = cl((2,2), 1=>3, bias = false) |> gpu
  ip = zeros(Float32, 28,28,1,1) |> gpu
  if typeof(l) <: BROKEN_LAYERS
    @test_broken sum(l(ip)) ≈ 0.f0
    @test_broken gradient(() -> sum(l(ip)), Flux.params(l)) isa Flux.Zygote.Grads
  else
    @test sum(l(ip)) ≈ 0.f0
    gs = gradient(() -> sum(l(ip)), Flux.params(l))
    @test l.bias ∉ gs.params 
  end
end

@testset "Dense with Zeros bias" begin
  l = Dense(ones(Float32, 4, 3), Flux.Zeros()) |> gpu
  ip = zeros(Float32, 3, 7) |> gpu

  @test sum(l(ip)) ≈ 0.f0
  gs = gradient(() -> sum(l(ip)), Flux.params(l))
  @test l.bias ∉ gs.params
end

@testset "Extended BatchNorm" begin
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
  # x_cpu = rand(Float64, 3, 2, 2)
  # x_gpu = x_cpu |> gpu
  # m_cpu(x_cpu)
  # gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  # m_gpu(x_gpu)
  # gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
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
    @test layer_gpu(input) isa Flux.CUDA.CuArray
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
