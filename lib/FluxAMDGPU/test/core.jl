import .Flux: cpu, gpu
using Statistics: mean
using LinearAlgebra: I, cholesky, Cholesky

@testset "Core" begin
  x = randn(5, 5)
  cx = gpu(x)
  @test cx isa ROCArray

  @test Flux.onecold(gpu([1.0, 2.0, 3.0])) == 3

  x = Flux.onehotbatch([1, 2, 3], 1:3)
  cx = gpu(x)
  @test cx isa Flux.OneHotMatrix && cx.indices isa ROCArray
  @test (cx .+ 1) isa ROCArray

  m = Chain(Dense(10, 5, tanh), Dense(5, 2), softmax)
  cm = gpu(m)

  @test all(p isa ROCArray for p in params(cm))
  @test cm(gpu(rand(10, 10))) isa ROCArray{Float32,2}

  xs = rand(5, 5)
  ys = Flux.onehotbatch(1:5,1:5)
  @test collect(roc(xs) .+ roc(ys)) ≈ collect(xs .+ ys)

  c = gpu(Conv((2,2),3=>4))
  x = gpu(rand(10, 10, 3, 2))
  l = c(gpu(rand(10,10,3,2)))
  @test gradient(x -> sum(c(x)), x)[1] isa ROCArray

  c = gpu(CrossCor((2,2),3=>4))
  x = gpu(rand(10, 10, 3, 2))
  l = c(gpu(rand(10,10,3,2)))
  @test gradient(x -> sum(c(x)), x)[1] isa ROCArray

end

@testset "onecold gpu" begin
  y = Flux.onehotbatch(ones(3), 1:10) |> gpu;
  l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
  @test Flux.onecold(y) isa ROCArray
  @test y[3,:] isa ROCArray
  @test Flux.onecold(y, l) == ['a', 'a', 'a']
end

@testset "restructure gpu" begin
  dudt = Dense(1,1) |> gpu
  p,re = Flux.destructure(dudt)
  foo(x) = sum(re(p)(x))
  @test gradient(foo, roc(rand(1)))[1] isa ROCArray
end

@testset "GPU functors" begin
  @testset "Cholesky" begin
    M = 2.0*I(10) |> collect
    Q = cholesky(M)
    Q_gpu = Q |> gpu
    @test Q_gpu isa Cholesky{<:Any,<:ROCArray}
    Q_cpu = Q_gpu |> cpu
    @test Q_cpu == cholesky(eltype(Q_gpu).(M))
  end
end
