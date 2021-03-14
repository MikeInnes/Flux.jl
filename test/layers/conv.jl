using Flux, Test
using Flux: maxpool, meanpool
using Flux: gradient

@testset "Pooling" begin
  x = randn(Float32, 10, 10, 3, 2)
  y = randn(Float32, 20, 20, 3, 2)
  ampx = AdaptiveMaxPool((5,5))
  @test ampx(x) == maxpool(x, PoolDims(x, 2))
  ampx = AdaptiveMeanPool((5,5))
  @test ampx(x) == meanpool(x, PoolDims(x, 2))
  ampy = AdaptiveMaxPool((10, 5))
  @test ampy(y) == maxpool(y, PoolDims(y, (2, 4)))
  ampy = AdaptiveMeanPool((10, 5))
  @test ampy(y) == meanpool(y, PoolDims(y, (2, 4)))
  gmp = GlobalMaxPool()
  @test size(gmp(x)) == (1, 1, 3, 2)
  gmp = GlobalMeanPool()
  @test size(gmp(x)) == (1, 1, 3, 2)
  mp = MaxPool((2, 2))
  @test mp(x) == maxpool(x, PoolDims(x, 2))
  mp = MeanPool((2, 2))
  @test mp(x) == meanpool(x, PoolDims(x, 2))
end

@testset "CNN" begin
  r = zeros(Float32, 28, 28, 1, 5)
  m = Chain(
    Conv((2, 2), 1=>16, relu),
    MaxPool((2,2)),
    Conv((2, 2), 16=>8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  @test size(m(r)) == (10, 5)

  # Test bias switch
  bias = Conv(ones(Float32, 2, 2, 1, 3), ones(Float32, 3))
  ip = zeros(Float32, 28,28,1,1)

  op = bias(ip)
  @test sum(op) == prod(size(op))

  @testset "No bias mapped through $lmap" for lmap in (identity, cpu, f32)
    bias = Conv((2,2), 1=>3, bias = false) |> lmap
    op = bias(ip)
    @test sum(op) ≈ 0.f0
    gs = gradient(() -> sum(bias(ip)), Flux.params(bias))
    @test bias.bias ∉ gs.params
  end

  # Train w/o bias and make sure no convergence happens
  # when only bias can be converged
  bias = Conv((2, 2), 1=>3, bias = false);
  ip = zeros(Float32, 28,28,1,1)
  op = zeros(Float32, 27,27,3,1) .+ 2.f0
  opt = Descent()

  for _ = 1:10^3
    gs = gradient(params(bias)) do
      Flux.Losses.mse(bias(ip), op)
    end
    Flux.Optimise.update!(opt, params(bias), gs)
  end

  @test Flux.Losses.mse(bias(ip), op) ≈ 4.f0
end

@testset "asymmetric padding" begin
  r = ones(Float32, 28, 28, 1, 1)
  m = Conv((3, 3), 1=>1, relu; pad=(0,1,1,2))
  m.weight[:] .= 1.0
  m.bias[:] .= 0.0
  y_hat = m(r)[:,:,1,1]
  @test size(y_hat) == (27, 29)
  @test y_hat[1, 1] ≈ 6.0
  @test y_hat[2, 2] ≈ 9.0
  @test y_hat[end, 1] ≈ 4.0
  @test y_hat[1, end] ≈ 3.0
  @test y_hat[1, end-1] ≈ 6.0
  @test y_hat[end, end] ≈ 2.0
end

@testset "Depthwise Conv" begin
  r = zeros(Float32, 28, 28, 3, 5)
  m1 = DepthwiseConv((2, 2), 3=>15)
  @test size(m1(r), 3) == 15

  m2 = DepthwiseConv((2, 3), 3=>9)
  @test size(m2(r), 3) == 9

  m3 = DepthwiseConv((2, 3), 3=>9; bias=false)
  @test size(m2(r), 3) == 9

  # Test that we cannot ask for non-integer multiplication factors
  @test_throws AssertionError DepthwiseConv((2,2), 3=>10)
end

@testset "ConvTranspose" begin
  x = zeros(Float32, 5, 5, 1, 1)
  y = Conv((3,3), 1 => 1)(x)
  x_hat1 = ConvTranspose((3, 3), 1 => 1)(y)
  x_hat2 = ConvTranspose((3, 3), 1 => 1, bias=false)(y)
  @test size(x_hat1) == size(x_hat2) == size(x)

  m = ConvTranspose((3,3), 1=>1)
  # Test that the gradient call does not throw: #900
  @test gradient(()->sum(m(x)), params(m)) isa Flux.Zygote.Grads

  x = zeros(Float32, 5, 5, 2, 4)
  m = ConvTranspose((3,3), 2=>3)
  @test gradient(()->sum(m(x)), params(m)) isa Flux.Zygote.Grads
end

@testset "CrossCor" begin
  x = rand(Float32, 28, 28, 1, 1)
  w = rand(2,2,1,1)
  y = CrossCor(w, [0.0])

  @test isapprox(sum(w .* x[1:2, 1:2, :, :]), y(x)[1, 1, 1, 1], rtol=1e-7)

  r = zeros(Float32, 28, 28, 1, 5)
  m = Chain(
    CrossCor((2, 2), 1=>16, relu),
    MaxPool((2,2)),
    CrossCor((2, 2), 16=>8, relu; bias=false),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  @test size(m(r)) == (10, 5)
  @test y(x) != Conv(w, [0.0])(x)
  @test CrossCor(w[end:-1:1, end:-1:1, :, :], [0.0])(x) == Conv(w, [0.0])(x)
end

@testset "Conv with non quadratic window #700" begin
  data = zeros(Float32, 7,7,1,1)
  data[4,4,1,1] = 1

  l = Conv((3,3), 1=>1)
  expected = zeros(eltype(l.weight),5,5,1,1)
  expected[2:end-1,2:end-1,1,1] = l.weight
  @test expected ≈ l(data)

  l = Conv((3,1), 1=>1)
  expected = zeros(eltype(l.weight),5,7,1,1)
  expected[2:end-1,4,1,1] = l.weight
  @test expected ≈ l(data)

  l = Conv((1,3), 1=>1)
  expected = zeros(eltype(l.weight),7,5,1,1)
  expected[4,2:end-1,1,1] = l.weight
  @test expected ≈ l(data)

  @test begin
    # we test that the next expression does not throw
    randn(Float32, 10,10,1,1) |> Conv((6,1), 1=>1, Flux.σ)
    true
  end
end

@testset "$ltype SamePad kernelsize $k" for ltype in (Conv, ConvTranspose, DepthwiseConv, CrossCor), k in ( (1,), (2,), (3,), (4,5), (6,7,8))
  data = ones(Float32, (k .+ 3)..., 1,1)
  l = ltype(k, 1=>1, pad=SamePad())
  @test size(l(data)) == size(data)

  l = ltype(k, 1=>1, pad=SamePad(), dilation = k .÷ 2)
  @test size(l(data)) == size(data)

  stride = 3
  l = ltype(k, 1=>1, pad=SamePad(), stride = stride)
  if ltype == ConvTranspose
    @test size(l(data))[1:end-2] == stride .* size(data)[1:end-2]
  else
    @test size(l(data))[1:end-2] == cld.(size(data)[1:end-2], stride)
  end
end

@testset "$ltype SamePad windowsize $k" for ltype in (MeanPool, MaxPool), k in ( (1,), (2,), (3,), (4,5), (6,7,8))
  data = ones(Float32, (k .+ 3)..., 1,1)

  l = ltype(k, pad=SamePad())
  @test size(l(data))[1:end-2] == cld.(size(data)[1:end-2], k)
end

@testset "bugs fixed" begin
  # https://github.com/FluxML/Flux.jl/issues/1421
  @test Conv((5, 5), 10 => 20, identity; init = Base.randn).bias isa Vector{Float64}
end

@testset "constructors: $fun" for fun in [Conv, CrossCor, ConvTranspose, DepthwiseConv]
  @test fun(rand(2,3,4)).bias isa Vector{Float64}
  @test fun(rand(2,3,4,5), false).bias === false
  if fun == Conv
    @test fun(rand(2,3,4,5,6), rand(6)).bias isa Vector{Float64}
    @test fun(rand(2,3,4,5,6), 1:6).bias isa Vector{Float64}
  elseif fun == DepthwiseConv
    @test fun(rand(2,3,4,5,6), rand(30)).bias isa Vector{Float64}
  end
  @test_throws DimensionMismatch fun(rand(2,3,4), rand(6))
end
