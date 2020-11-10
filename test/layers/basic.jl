using Test, Random
import Flux: activations

@testset "basic" begin
  @testset "helpers" begin
    @testset "activations" begin
      dummy_model = Chain(x->x.^2, x->x .- 3, x -> tan.(x))
      x = randn(10)
      @test activations(dummy_model, x)[1] == x.^2
      @test activations(dummy_model, x)[2] == (x.^2 .- 3)
      @test activations(dummy_model, x)[3] == tan.(x.^2 .- 3)

      @test activations(Chain(), x) == ()
      @test activations(Chain(identity, x->:foo), x)[2] == :foo # results include `Any` type
    end
  end

  @testset "Chain" begin
    @test_nowarn Chain(Dense(10, 5, σ), Dense(5, 2))(randn(10))
    @test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
    # numeric test should be put into testset of corresponding layer
  end

  @testset "Activations" begin
    c = Chain(Dense(3,5,relu), Dense(5,1,relu))
    X = Float32.([1.0; 1.0; 1.0])
    @test_nowarn gradient(()->Flux.activations(c, X)[2][1], params(c))
  end

  @testset "Dense" begin
    @testset "constructors" begin
      @test size(Dense(10, 100).W) == (100, 10)
      @test Dense(rand(100,10), rand(10)).σ == identity

      @test_throws MethodError Dense(10, 10.5)
      @test_throws MethodError Dense(10, 10.5, tanh)
    end

    @test  length(Dense(10, 5)(randn(10))) == 5
    @test_throws DimensionMismatch Dense(10, 5)(randn(1))
    @test_throws MethodError Dense(10, 5)(1) # avoid broadcasting
    @test_throws MethodError Dense(10, 5).(randn(10)) # avoid broadcasting

    @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,1)) == 10*ones(1, 1)
    @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,2)) == 10*ones(1, 2)
    @test Dense(10, 2, identity, initW = ones, initb = zeros)(ones(10,1)) == 10*ones(2, 1)
    @test Dense(10, 2, identity, initW = ones, initb = zeros)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]
  end

  @testset "Diagonal" begin
    @test length(Flux.Diagonal(10)(randn(10))) == 10
    @test length(Flux.Diagonal(10)(1)) == 10
    @test length(Flux.Diagonal(10)(randn(1))) == 10
    @test_throws DimensionMismatch Flux.Diagonal(10)(randn(2))

    @test Flux.Diagonal(2)([1 2]) == [1 2; 1 2]
    @test Flux.Diagonal(2)([1,2]) == [1,2]
    @test Flux.Diagonal(2)([1 2; 3 4]) == [1 2; 3 4]
  end

  @testset "Maxout" begin
    # Note that the normal common usage of Maxout is as per the docstring
    # These are abnormal constructors used for testing purposes

    @testset "Constructor" begin
      mo = Maxout(() -> identity, 4)
      input = rand(40)
      @test mo(input) == input
    end

    @testset "simple alternatives" begin
      mo = Maxout((x -> x, x -> 2x, x -> 0.5x))
      input = rand(40)
      @test mo(input) == 2*input
    end

    @testset "complex alternatives" begin
      mo = Maxout((x -> [0.5; 0.1]*x, x -> [0.2; 0.7]*x))
      input = [3.0 2.0]
      target = [0.5, 0.7].*input
      @test mo(input) == target
    end

    @testset "params" begin
      mo = Maxout(()->Dense(32, 64), 4)
      ps = params(mo)
      @test length(ps) == 8  #4 alts, each with weight and bias
    end
  end

  @testset "SkipConnection" begin
    @testset "zero sum" begin
      input = randn(10, 10, 10, 10)
      @test SkipConnection(x -> zeros(size(x)), (a,b) -> a + b)(input) == input
    end

    @testset "concat size" begin
      input = randn(10, 2)
      @test size(SkipConnection(Dense(10,10), (a,b) -> cat(a, b, dims = 2))(input)) == (10,4)
    end
  end

  @testset "output dimensions" begin
    m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32))
    @test Flux.outdims(m, (10, 10)) == (6, 6)

    m = Dense(10, 5)
    @test_throws DimensionMismatch Flux.outdims(m, (5, 2)) == (5,)
    @test Flux.outdims(m, (10,)) == (5,)

    m = Chain(Dense(10, 8, σ), Dense(8, 5), Dense(5, 2))
    @test Flux.outdims(m, (10,)) == (2,)

    m = Chain(Dense(10, 8, σ), Dense(8, 4), Dense(5, 2))
    @test_throws DimensionMismatch Flux.outdims(m, (10,))

    m = Flux.Diagonal(10)
    @test Flux.outdims(m, (10,)) == (10,)

    m = Maxout(() -> Conv((3, 3), 3 => 16), 2)
    @test Flux.outdims(m, (10, 10)) == (8, 8)
  end
end
