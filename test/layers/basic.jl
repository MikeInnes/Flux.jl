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
      @test size(Dense(10, 100).weight) == (100, 10)
      @test size(Dense(10, 100).bias) == (100,)
      @test Dense(rand(100,10), rand(100)).σ == identity
      @test Dense(rand(100,10)).σ == identity

      @test Dense(rand(100,10), false).σ == identity
      @test Dense(rand(100,10), false, tanh).σ == tanh
      @test Dense(rand(100,10), rand(100)).σ == identity
      @test Dense(rand(Float16, 100,10), true).bias isa Vector{Float16}  # creates matching type
      @test_skip Dense(rand(Float16, 100,10), rand(100)).bias isa Vector{Float16}  # converts to match

      @test Dense(3,4; init=Base.randn, bias=true).bias isa Vector{Float64}
      @test_skip Dense(3,4; init=Base.randn, bias=[1,2,3,4]).bias isa Vector{Float64}

      @test_throws MethodError Dense(10, 10.5)
      @test_throws MethodError Dense(10, 10.5, tanh)
      @test_throws DimensionMismatch Dense(3,4; bias=rand(5))
      @test_throws DimensionMismatch Dense(rand(4,3), rand(5))
      @test_throws MethodError Dense(rand(5))
      @test_throws MethodError Dense(rand(5), rand(5))
      @test_throws MethodError Dense(rand(5), rand(5), tanh)
    end
    @testset "dimensions" begin
      @test  length(Dense(10, 5)(randn(10))) == 5
      @test_throws DimensionMismatch Dense(10, 5)(randn(1))
      @test_throws MethodError Dense(10, 5)(1) # avoid broadcasting
      @test_throws MethodError Dense(10, 5).(randn(10)) # avoid broadcasting
      @test size(Dense(10, 5)(randn(10))) == (5,)
      @test size(Dense(10, 5)(randn(10,2))) == (5,2)
      @test size(Dense(10, 5)(randn(10,2,3))) == (5,2,3)
      @test size(Dense(10, 5)(randn(10,2,3,4))) == (5,2,3,4)
      @test_throws DimensionMismatch Dense(10, 5)(randn(11,2,3))
    end
    @testset "zeros" begin
      @test Dense(10, 1, identity, init = ones)(ones(10,1)) == 10*ones(1, 1)
      @test Dense(10, 1, identity, init = ones)(ones(10,2)) == 10*ones(1, 2)
      @test Dense(10, 2, identity, init = ones)(ones(10,1)) == 10*ones(2, 1)
      @test Dense(10, 2, identity, init = ones)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]
      @test Dense(10, 2, identity, init = ones, bias = false)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]
    end
  end

  @testset "Diagonal" begin
    @test length(Flux.Diagonal(10)(randn(10))) == 10
    @test length(Flux.Diagonal(10)(1)) == 10
    @test length(Flux.Diagonal(10)(randn(1))) == 10
    @test_throws DimensionMismatch Flux.Diagonal(10)(randn(2))

    @test Flux.Diagonal(2)([1 2]) == [1 2; 1 2]
    @test Flux.Diagonal(2)([1,2]) == [1,2]
    @test Flux.Diagonal(2)([1 2; 3 4]) == [1 2; 3 4]

    @test Flux.Diagonal(2)(rand(2,3,4)) |> size == (2, 3, 4)
    @test Flux.Diagonal(2,3)(rand(2,3,4)) |> size == (2, 3, 4)
    @test Flux.Diagonal(2,3,4)(rand(2,3,4)) |> size == (2, 3, 4)
    @test Flux.Diagonal(2,3)(rand(2,1,4)) |> size == (2, 3, 4)
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

  @testset "Bilinear" begin
    @testset "SkipConnection recombinator" begin
      d = Dense(10, 10)
      b = Flux.Bilinear(10, 10, 5)
      x = randn(Float32,10,9)
      sc = SkipConnection(d, b)
      @test size(sc(x)) == (5,9)
    end

    @testset "Two-streams zero sum" begin
      x = zeros(Float32,10,9)
      y = zeros(Float32,2,9)
      b = Flux.Bilinear(10, 2, 3)
      @test size(b(x,y)) == (3,9)
      @test sum(abs2, b(x,y)) == 0f0
    end

    @testset "Inner interactions" begin
      x = randn(Float32,11,7)
      b = Flux.Bilinear(11, 11, 3)
      @test size(b(x)) == (3,7)
      @test_nowarn gs = gradient(() -> sum(abs2.(b(x))), params(b))
    end

    @testset "constructors" begin
      b1 = Flux.Bilinear(randn(3,4,5))
      @test b1.bias isa Vector{Float64}
      @test b1.σ == identity

      b2 = Flux.Bilinear(randn(3,4,5), false)
      @test b2.bias == Flux.Zeros()

      b3 = Flux.Bilinear(randn(Float16, 3,4,5), true, tanh)
      @test b3.σ == tanh
      @test b3.bias isa Vector{Float16}
      @test size(b3(rand(4), rand(5))) == (3,)

      b4 = Flux.Bilinear(3,3,7; bias=1:7, init=Flux.zeros32)
      @test_skip  b4.bias isa Vector{Float32}

      @test_throws ArgumentError Flux.Bilinear(rand(3)) # expects a 3-array
      @test_throws ArgumentError Flux.Bilinear(rand(3,4), false, tanh)
      @test_throws DimensionMismatch Flux.Bilinear(rand(3,4,5), rand(6), tanh) # wrong length bias
    end
  end

  @testset "Parallel" begin
    @testset "zero sum" begin
      input = randn(10, 10, 10, 10)
      @test Parallel(+, x -> zeros(size(x)), identity)(input) == input
    end

    @testset "concat size" begin
      input = randn(10, 2)
      @test size(Parallel((a, b) -> cat(a, b; dims=2), Dense(10, 10), identity)(input)) == (10, 4)
    end

    @testset "vararg input" begin
      inputs = randn(10), randn(5), randn(4)
      @test size(Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2))(inputs)) == (2,)
    end
  end

  @testset "Embedding" begin
    vocab_size, embed_size = 10, 4
    m = Flux.Embedding(vocab_size => embed_size)
    @test size(m.weight) == (embed_size, vocab_size)
    
    x = rand(1:vocab_size, 3)
    y = m(x)
    @test y isa Matrix{Float32}
    @test y ≈ m.weight[:,x]
    x2 = OneHotMatrix(x, vocab_size)
    y2 = m(x2)
    @test y2 isa Matrix{Float32}
    @test y2 ≈ y
    @test_throws DimensionMismatch m(OneHotMatrix(x, 1000))

    x = rand(1:vocab_size, 3, 4)
    y = m(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (embed_size, 3, 4)

    @test m(2) ≈ m.weight[:,2]
    @test m(OneHotVector(3, vocab_size)) ≈ m.weight[:,3]
    @test_throws DimensionMismatch m(OneHotVector(3, 1000))
  end
end
