using Flux
using Flux: throttle, nfan, glorot_uniform, glorot_normal, kaiming_normal, kaiming_uniform, orthogonal, sparse_init, stack, unstack, Zeros
using StatsBase: var, std
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

@testset "Initialization" begin
  # Set random seed so that these tests don't fail randomly
  Random.seed!(0)

  @testset "Fan in/out" begin
    @test nfan() == (1, 1) #For a constant
    @test nfan(100) == (1, 100) #For vector
    @test nfan(100, 200) == (200, 100) == nfan((100, 200)) #For Dense layer
    @test nfan(2, 30, 40) == (2 * 30, 2 * 40) #For 1D Conv layer
    @test nfan(2, 3, 40, 50) == (2 * 3 * 40, 2 * 3 * 50) #For 2D Conv layer
    @test nfan(2, 3, 4, 50, 60) == (2 * 3 * 4 * 50, 2 * 3 * 4 * 60) #For 3D Conv layer
  end

  @testset "glorot" begin
    # glorot_uniform and glorot_normal should both yield a kernel with
    # variance ≈ 2/(fan_in + fan_out)
    for dims ∈ [(1000,), (100, 100), (100, 400), (2, 3, 32, 64), (2, 3, 4, 32, 64)]
      for init ∈ [glorot_uniform, glorot_normal]
        v = init(dims...)
        fan_in, fan_out = nfan(dims...)
        σ2 = 2 / (fan_in + fan_out)
        @test 0.9σ2 < var(v) < 1.1σ2
        @test eltype(v) == Float32
      end
    end
  end

  @testset "kaiming" begin
    # kaiming_uniform should yield a kernel in range [-sqrt(6/n_out), sqrt(6/n_out)]
    # and kaiming_normal should yield a kernel with stddev ~= sqrt(2/n_out)
    for (n_in, n_out) in [(100, 100), (100, 400)]
      v = kaiming_uniform(n_in, n_out)
      σ2 = sqrt(6/n_out)
      @test -1σ2  < minimum(v) < -0.9σ2
      @test 0.9σ2  < maximum(v) < 1σ2
      @test eltype(v) == Float32

      v = kaiming_normal(n_in, n_out)
      σ2 = sqrt(2/n_out)
      @test 0.9σ2 < std(v) < 1.1σ2
      @test eltype(v) == Float32
    end
  end

  @testset "orthogonal" begin
    # A matrix of dim = (m,n) with m > n should produce a QR decomposition. In the other case, the transpose should be taken to compute the QR decomposition.
    for (rows,cols) in [(5,3),(3,5)]
      v = orthogonal(rows, cols)
      rows < cols ? (@test v * v' ≈ I(rows)) : (@test v' * v ≈ I(cols))
    end
    for mat in [(3,4,5),(2,2,5)]
      v = orthogonal(mat...)
      cols = mat[end]
      rows = div(prod(mat),cols)
      v = reshape(v, (rows,cols))
      rows < cols ? (@test v * v' ≈ I(rows)) : (@test v' * v ≈ I(cols))
    end
  end

  @testset "sparse_init" begin
    # sparse_init should yield an error for non 2-d dimensions
    # sparse_init should yield no zero elements if sparsity < 0
    # sparse_init should yield all zero elements if sparsity > 1
    # sparse_init should yield exactly ceil(n_in * sparsity) elements in each column for other sparsity values
    # sparse_init should yield a kernel in its non-zero elements consistent with the std parameter

    @test_throws ArgumentError sparse_init(100, 100, 100, sparsity=0.1)
    v = sparse_init(100, 100, sparsity=-0.1)
    @test sum(v .== 0) == 0
    @test eltype(v) == Float32
    v = sparse_init(100, 100, sparsity=1.1)
    @test sum(v .== 0) == length(v)
    @test eltype(v) == Float32

    for (n_in, n_out, sparsity, σ) in [(100, 100, 0.25, 0.1), (100, 400, 0.75, 0.01)]
      expected_zeros = ceil(Integer, n_in * sparsity)
      v = sparse_init(n_in, n_out, sparsity=sparsity, std=σ)
      @test all([sum(v[:,col] .== 0) == expected_zeros for col in 1:n_out])
      @test 0.9 * σ < std(v[v .!= 0]) < 1.1 * σ
      @test eltype(v) == Float32
    end
  end

  @testset "partial_application" begin
    big = 1e9

    partial_ku = kaiming_uniform(gain=big)
    @test maximum(partial_ku(8, 8)) > big / 2
    @test maximum(partial_ku(8, 8, gain=1)) < big / 2

    partial_kn = kaiming_normal(gain=big)
    @test maximum(partial_kn(8, 8)) > big / 2
    @test maximum(partial_kn(8, 8, gain=1)) < big / 2

    partial_si = sparse_init(sparsity=1)
    @test maximum(partial_si(8, 8)) == 0
    @test maximum(partial_si(8, 8, sparsity=0)) > 0
  end

  @testset "identity_init" begin
    import Flux: identity_init

    @testset "Basic" begin
      partial = identity_init(gain=3)
      @test partial(3, 3) == identity_init(3, 3; gain=3) == [3 0 0; 0 3 0; 0 0 3]
    end

    @testset "Non-identity sizes" begin
        @test identity_init(2, 3)[:, end] == zeros(Float32, 2)
        @test identity_init(3, 2; shift=1)[1, :] == zeros(Float32, 2)
        @test identity_init(1, 1, 3, 4)[:, :, :, end] == zeros(Float32, 1, 1, 3)
        @test identity_init(2, 1, 3, 3)[end, :, :, :] == zeros(Float32, 1, 3, 3)
        @test identity_init(1, 2, 3, 3)[:, end, :, :] == zeros(Float32, 1, 3, 3)
    end

    @testset "Dense ID mapping" begin
        l = Dense(3,3, init = identity_init)

        indata = reshape(collect(Float32, 1:9), 3, 3)
        @test l(indata) == indata
    end

    @testset "$layer ID mapping with kernelsize $kernelsize" for layer in (Conv, ConvTranspose, CrossCor), kernelsize in (
        (1,),
        (3,), 
        (1, 3), 
        (3, 5), 
        (3, 5, 7))   
        nch = 3
        l = layer(kernelsize, nch=>nch, init=identity_init, pad=SamePad())

        indata = randn(Float32, kernelsize..., nch, nch)
        @test l(indata) == indata
    end

    @testset "Inception identity" begin
      insize = 7
      path1 = Conv((1, 3), insize=>2; init=identity_init, pad=SamePad())
      path2 = Conv((3, 5), insize=>3; init=identity_init(shift=(0, 0, 2, 0)), pad=SamePad())
      path3 = Conv((5, 7), insize=>2; init=identity_init(shift=(0, 0, 5, 0)), pad=SamePad())
      block = Parallel((xs...) -> cat(xs...;dims=3), path1, path2, path3)

      indata = randn(Float32, 9, 9, 7, 2)
      @test block(indata) == indata
    end
  end
end

@testset "Params" begin
  m = Dense(10, 5)
  @test size.(params(m)) == [(5, 10), (5,)]
  m = RNN(10, 5)
  @test size.(params(m)) == [(5, 10), (5, 5), (5,), (5, 1)]

  # Layer duplicated in same chain, params just once pls.
  c = Chain(m, m)
  @test size.(params(c)) == [(5, 10), (5, 5), (5,), (5, 1)]

  # Self-referential array. Just want params, no stack overflow pls.
  r = Any[nothing,m]
  r[1] = r
  @test size.(params(r)) == [(5, 10), (5, 5), (5,), (5, 1)]
end

@testset "Basic Stacking" begin
  x = randn(3,3)
  stacked = stack([x, x], 2)
  @test size(stacked) == (3,2,3)
end

@testset "Precision" begin
  m = Chain(Dense(10, 5, relu), Dense(5, 2))
  x64 = rand(Float64, 10)
  x32 = rand(Float32, 10)
  @test eltype(m[1].W) == Float32
  @test eltype(m(x32)) == Float32
  @test eltype(m(x64)) == Float64
  @test eltype(f64(m)(x32)) == Float64
  @test eltype(f64(m)(x64)) == Float64
  @test eltype(f64(m)[1].W) == Float64
  @test eltype(f32(f64(m))[1].W) == Float32
end

@testset "Zeros" begin
  m = Dense(3, 2; bias = false)
  @test f64(m).b === m.b
  @test f32(m).b === m.b

  @testset "Gradients for broadcasted $op with sizes $s" for op in (+, -, *), s in ((1,), (2,3))
    o = ones(s)
    z = zeros(s)
    Z = Zeros(s...)
    a = ones(3,3)
    b = zeros(3,3)
    b′ = Zeros(3,3)


    @testset "Basic operations" begin
      a = rand(3,3)
      b = zeros(3,3)
      bz = Zeros(3,3)
      
      for op in (+, -)
        @test op(a, b) == op(a, bz)
      end

      for op in (+, -)
        gs = gradient((a, b) -> sum(op(a, b)), a, b)
        gsz = gradient((a, b) -> sum(op(a, b)), a, bz)
        @test gs[1] == gsz[1]
        @test gsz[2] === nothing
      end

      # Check with broadcasting
      b = zeros(3,3,3)
      bz = Zeros(3,3,3)

      for op in (+, -)
        @test broadcast(op, a, b) == broadcast(op, a, bz)
      end

      for op in (+, -)
        gs = gradient((a,b) -> sum(broadcast(op, a, b)), a, b)
        gsz = gradient((a,b) -> sum(broadcast(op, a, b)), a, bz)
        @test gs[1] == gsz[1]
        @test gsz[2] === nothing
      end
    end

    @testset "Explicit" begin
      gfun(args...) = gradient((x, y) -> sum(op.(x, y)), args...)
      g = gfun(o, z)
      @test gfun(o, Z) == (g[1], nothing)

      g = gfun(z, o) 
      @test gfun(Z, o) == (nothing, g[2])
    end

    @testset "Implicit" begin
      gfun(args...) = gradient(() -> sum(op.(args...)), params(collect(args)))
      g = gfun(o, z)

      gres = gfun(o, Z)
      @test gres[o] == g[o]
      # @test Z ∉ gres.params

      g = gfun(z, o)
      gres = gfun(Z, o)
      @test gres[o] == g[o]
      # @test Z ∉ gres.params
    end
  end

  @testset "Gradients for broadcasted / with sizes $s" for s in ((1,), (2,3))
    o = ones(s)
    z = zeros(s)
    Z = Zeros(s...) # Only defined for 0-dim

    @testset "Explicit" begin
      gfun(args...) = gradient((x, y) -> sum(x ./ y), args...)
      g = gfun(z, o)
      @test gfun(Z, o) == (nothing, g[2])
    end

    @testset "Implicit" begin
      gfun(x,y) = gradient(() -> sum(x ./ y), params([x,y]))

      g = gfun(z, o)
      gres = gfun(Z, o)
      @test gres[o] == g[o]
      # @test Z ∉ gres.params
    end
  end

  @testset "Gradients for $op with sizes $s" for op in (+, -), s in (tuple(), (1,), (2,3))
    o = ones(s)
    z = zeros(s)
    Z = Zeros(s...)


    @testset "Explicit" begin
      gfun(args...) = gradient((x, y) -> sum(op(x,y)), args...)

      g = gfun(o, z)
      @test gfun(o, Z) == (g[1], nothing)

      g = gfun(z, o)
      @test gfun(Z, o) == (nothing, g[2])
    end

    @testset "Implicit" begin
      gfun(args...) = gradient(() -> sum(op(args...)), params(collect(args)))
      g = gfun(o, z)
      gres = gfun(o, Z)
      @test gres[o] == g[o]
      # @test Z ∉ gres.params

      g = gfun(z, o)
      gres = gfun(Z, o)
      @test gres[o] == g[o]
      # @test Z ∉ gres.params
    end
  end
end

@testset "Stacking" begin
  stacked_array=[ 8 9 3 5; 9 6 6 9; 9 1 7 2; 7 4 10 6 ]
  unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
  @test unstack(stacked_array, 2) == unstacked_array
  @test stack(unstacked_array, 2) == stacked_array
  @test stack(unstack(stacked_array, 1), 1) == stacked_array
end

@testset "Param remapping" begin
  ls(dims...) = reshape(collect(Float32, 1:prod(dims)), dims...) # accepts dims in reverse order to Dense
  dl(nin, nout, bias) = Dense(ls(nout, nin), bias(nout))
  dm(bias) = Chain(
    dl(3, 5, bias),
    dl(5, 4, bias),
    dl(4, 3, bias)
  )

  nobias(n) = Zeros(n)
  testdense(m, bt) = @testset "Check layer $i" for (i, (l1, l2)) in enumerate(zip(m, dm(bt)))
    @test l1.W == l2.W
    @test l1.b == l2.b
    @test_skip typeof(l1.b) === typeof(l2.b)
  end

  @testset "loadparams!" begin
    import Flux: loadparams!
    pars(w, b) = [w, b]
    import Flux: loadparams!, Zeros
    pars(w, b::Zeros) = [w, Flux.zeros32(size(w,1))]
    pars(l) = pars(l.W, l.b)
    pararray(m) = mapreduce(pars, vcat, m)
    weights(m) = mapreduce(l -> [l.W], vcat, m)
    @testset "Bias type $bt" for bt in (Flux.zeros32, nobias)
      m = dm(bt)
      loadparams!(m, params(m))
      testdense(m, bt)
    end

    @testset "$b1 to $b2" for (b1, b2, be) in (
      (Flux.zeros32, Flux.ones32, Flux.ones32),   # Load ones as bias to a model with zeros as bias -> model gets ones as bias
      (Flux.ones32, nobias, Flux.zeros32), # Load Zeros as bias to a model with ones as bias-> model gets zeros as bias
      (nobias, Flux.ones32, nobias),     # Load ones as bias to a model with Zeros as bias-> model bias does not change
    )
      m1 = dm(b1)
      m2 = dm(b2)
      loadparams!(m1, b1 == nobias ? weights(m2) : pararray(m2))
      testdense(m1, be)
    end
  end

  @testset "destructure" begin
    import Flux: destructure
    @testset "Bias type $bt" for bt in (zeros, nobias)
      m = dm(bt)
      p, re = destructure(m)
      testdense(re(p), bt)
    end

    @testset "restructure in gradient" begin
      x = rand(Float32, 3, 1)
      m = dm(zeros)
      ∇m = gradient(m -> sum(m(x)), m)[1]
      p, re = destructure(m)
      ∇p = gradient(θ -> sum(re(θ)(x)), p)[1]
      @test ∇p ≈ destructure(∇m)[1]
    end
  end
end

@testset "Train and test mode" begin
  mutable struct DummyLayer
    testing::Bool
  end
  Flux.testmode!(m::DummyLayer, testing=true) = (m.testing = testing; m)

  c = Chain(DummyLayer(true))
  testmode!(c)
  @test c[1].testing
  trainmode!(c)
  @test !c[1].testing
end

@testset "modules" begin
  m1 = Conv((2,3), 4=>5; pad=6, stride=7)
  m2 = LayerNorm(8)
  m3 = m2.diag
  m4 = SkipConnection(m1, +)
  m5 =  Chain(m4, m2)
  modules = Flux.modules(m5)
  # Depth-first descent
  @test length(modules) == 5
  @test modules[1] === m5
  @test modules[2] === m4
  @test modules[3] === m1
  @test modules[4] === m2
  @test modules[5] === m3

  modules = Flux.modules(Chain(Dense(2,3), BatchNorm(3), LSTM(3,4)))
  @test length(modules) == 5

  modules = Flux.modules(Chain(SkipConnection(
                                  Conv((2,3), 4=>5; pad=6, stride=7),
                                  +), 
                                LayerNorm(8)))
  @test length(modules) == 5
end

@testset "Patience triggers" begin
  @testset "patience" begin
    trigger = Flux.patience(() -> true, 3)

    @test trigger() == false
    @test trigger() == false
    @test trigger() == true

    v = [false, true, false, true, true, true]
    trigger = let v = v
      Flux.patience(i -> v[i], 3)
    end

    n_iter = 0
    for i in 1:length(v)
      trigger(i) && break
      n_iter += 1
    end

    @test n_iter == 5
  end

  @testset "early stopping" begin
    @testset "args & kwargs" begin
      es = Flux.early_stopping((x; y = 1) -> x + y, 10; min_dist=3)
  
      n_iter = 0
      while n_iter < 99
        es(-n_iter; y=-n_iter) && break
        n_iter += 1
      end
  
      @test n_iter == 9
    end
  
    @testset "distance" begin
      es = Flux.early_stopping(identity, 10; distance=(best_score, score) -> score - best_score)

      n_iter = 0
      while n_iter < 99
        es(n_iter) && break
        n_iter += 1
      end

      @test n_iter == 99
    end
  
    @testset "init_score" begin
      es = Flux.early_stopping(identity, 10; init_score=10)

      n_iter = 0
      while n_iter < 99
        es(n_iter) && break
        n_iter += 1
      end

      @test n_iter == 10
    end
  end

  @testset "plateau" begin
    f = let v = 10
      () -> v = v / abs(v) - v
    end

    trigger = Flux.plateau(f, 3, init_score=10, min_dist=18)

    n_iter = 0
    while n_iter < 99
      trigger() && break
      n_iter += 1
    end

    @test n_iter == 3
  end
end
