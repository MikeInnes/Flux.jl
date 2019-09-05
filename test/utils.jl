using Flux
using Flux: throttle, jacobian, glorot_uniform, glorot_normal, stack, unstack
using Flux: destructure
using StatsBase: std
using Random
using Test

@testset "Throttle" begin
  @testset "default behaviour" begin
    a = []
    f = throttle(()->push!(a, time()), 1, leading=true, trailing=false)
    f()
    f()
    f()
    sleep(1.01)
    @test length(a) == 1
  end

  @testset "leading behaviour" begin
    a = []
    f = throttle(()->push!(a, time()), 1, leading=true, trailing=false)
    f()
    @test length(a) == 1
    f()
    @test length(a) == 1
    sleep(1.01)
    f()
    @test length(a) == 2
  end

  @testset "trailing behaviour" begin
    a = []
    f = throttle(()->push!(a, time()), 1, leading=false, trailing=true)
    f()
    @test length(a) == 0
    f()
    @test length(a) == 0
    sleep(1.01)
    @test length(a) == 1
  end

  @testset "arguments" begin
    a = []
    f = throttle((x)->push!(a, x), 1, leading=true, trailing=true)
    f(1)
    @test a == [1]
    f(2)
    @test a == [1]
    f(3)
    @test a == [1]
    sleep(1.01)
    @test a == [1, 3]
  end
end

@testset "Jacobian" begin
  A = param(randn(2,2))
  x = randn(2)
  m(x) = A*x
  y = m(x)
  J = jacobian(m,x)
  @test J ≈ A.data
end

@testset "Initialization" begin
  # Set random seed so that these tests don't fail randomly
  Random.seed!(0)

  # glorot_uniform should yield a kernel with stddev ~= sqrt(6/(n_in + n_out)),
  # and glorot_normal should yield a kernel with stddev != 2/(n_in _ n_out)
  for (n_in, n_out) in [(100, 100), (100, 400)]
    v = glorot_uniform(n_in, n_out)
    @test minimum(v) > -1.1*sqrt(6/(n_in + n_out))
    @test minimum(v) < -0.9*sqrt(6/(n_in + n_out))
    @test maximum(v) >  0.9*sqrt(6/(n_in + n_out))
    @test maximum(v) <  1.1*sqrt(6/(n_in + n_out))

    v = glorot_normal(n_in, n_out)
    @test std(v) > 0.9*sqrt(2/(n_in + n_out))
    @test std(v) < 1.1*sqrt(2/(n_in + n_out))
  end
end

@testset "Params" begin
  m = Dense(10, 5)
  @test size.(params(m)) == [(5, 10), (5,)]
  m = RNN(10, 5)
  @test size.(params(m)) == [(5, 10), (5, 5), (5,), (5,)]

  # Layer duplicated in same chain, params just once pls.
  c = Chain(m, m)
  @test size.(params(c)) == [(5, 10), (5, 5), (5,), (5,)]

  # Self-referential array. Just want params, no stack overflow pls.
  r = Any[nothing,m]
  Flux.children(a::Vector{Any}) = Tuple(a)
  r[1] = r
  @test size.(params(r)) == [(5, 10), (5, 5), (5,), (5,)]
end

@testset "Basic Stacking" begin
  x = randn(3,3)
  stacked = stack([x, x], 2)
  @test size(stacked) == (3,2,3)
end

@testset "Precision" begin
  m = Chain(Dense(10, 5, relu), Dense(5, 2))
  x = rand(10)
  @test eltype(m[1].W.data) == Float32
  @test eltype(m(x).data) == Float32
  @test eltype(f64(m)(x).data) == Float64
  @test eltype(f64(m)[1].W.data) == Float64
  @test eltype(f32(f64(m))[1].W.data) == Float32
  @test Tracker.isleaf(f32(f64(m))[1].W)
end

@testset "Stacking" begin
  stacked_array=[ 8 9 3 5; 9 6 6 9; 9 1 7 2; 7 4 10 6 ]
  unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
  @test unstack(stacked_array, 2) == unstacked_array
  @test stack(unstacked_array, 2) == stacked_array
  @test stack(unstack(stacked_array, 1), 1) == stacked_array
end


@testset "De/Re-Structuring" begin
    #Single Dense Layer
    m_orig = Dense(1,2)
    ps, re = destructure(m_orig)
    m_reparamed = re(ps)
    ps_new = param(randn!(similar(ps))) # needs param o/w no restructure...
    m2=re(ps_new)
    @test destructure(m_orig)[1] == destructure(m_reparamed)[1]
    @test destructure(m2)[1] == ps_new
    # Chain
    m_orig = Chain(Dense(10,2,tanh),Dense(2,4))
    ps, re = destructure(m_orig)
    m_reparamed = re(ps)
    ps_new = param(randn!(similar(ps))) # needs param o/w no restructure...
    m2=re(ps_new)
    @test destructure(m_orig)[1] == destructure(m_reparamed)[1]
    @test destructure(m2)[1] == ps_new
end

@testset "Wrong length ps" begin
    #Single Dense Layer
    m_orig = Dense(1,2)
    ps, re = destructure(m_orig)
    m_reparamed = re([ps;ps])
end

@testset "Restructure with Untracked" begin
    m = Dense(2,3)
    ps, re = destructure(m)
    m2 = re(Flux.data(ps))
    ps2,_ = destructure(m2)
    println(ps)
    println(ps2)
    @test_broken ps == ps2
end
