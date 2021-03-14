using Flux.Optimise
using Flux.Optimise: runall
using Flux: Params, gradient
using Test
using Random

@testset "Optimise" begin
  # Ensure rng has different state inside and outside the inner @testset
  # so that w and w' are different
  Random.seed!(84)
  w = randn(10, 10)
  @testset for opt in [ADAMW(), ADAGrad(0.1), AdaMax(), ADADelta(0.9), AMSGrad(),
                       NADAM(), RADAM(), Descent(0.1), ADAM(), OADAM(), AdaBelief(),
                       Nesterov(), RMSProp(), Momentum()]
    Random.seed!(42)
    w′ = randn(10, 10)
    b = false
    loss(x) = Flux.Losses.mse(w*x, w′*x .+ b)
    for t = 1: 10^5
      θ = params([w′, b])
      x = rand(10)
      θ̄ = gradient(() -> loss(x), θ)
      Optimise.update!(opt, θ, θ̄)
    end
    @test loss(rand(10, 10)) < 0.01
  end
end

@testset "Optimiser" begin
  Random.seed!(84)
  w = randn(10, 10)
  @testset for Opt in [InvDecay, WeightDecay, ExpDecay]
    Random.seed!(42)
    w′ = randn(10, 10)
    loss(x) = Flux.Losses.mse(w*x, w′*x)
    opt = Optimiser(Opt(), ADAM(0.001))
    for t = 1:10^5
      θ = Params([w′])
      x = rand(10)
      θ̄ = gradient(() -> loss(x), θ)
      Optimise.update!(opt, θ, θ̄)
    end
    @test loss(rand(10, 10)) < 0.01
  end
end

@testset "Training Loop" begin
  i = 0
  l = 1
  Flux.train!(
              () -> (sleep(0.1); Flux.skip(); i+=1),
              (),
              Iterators.repeated((), 10),
              Descent()
             )

  @test i==0 #all skipped

  Flux.train!(
              () -> (sleep(0.1); i==8 && Flux.skip(); i+=1),
              (),
              Iterators.repeated((), 10),
              Descent()
             )

  @test i==8 #skip after i hit 8

  i = 0
  Flux.train!(() -> (sleep(0.1); i += 1; l),
              (),
              Iterators.repeated((), 100),
              Descent(),
              cb = Flux.throttle(() -> (i > 3 && Flux.stop()), 1))

  @test 3 < i < 50

  # Test multiple callbacks
  x = 0
  fs = [() -> (), () -> x = 1]
  cbs = runall(fs)
  cbs()
  @test x == 1

  r = rand(3, 3)
  loss(x) = sum(x .* x)
  Flux.train!(loss, Flux.params(r), (r,), Descent())
end

@testset "ExpDecay" begin

  @testset "Sanity Check" begin
    o = ExpDecay(0.2, 0.5, 1, 1e-3)
    p = [0.0]
    steps = 1:8
    eta_expected = @. max(o.eta * 0.5 ^ steps, o.clip)
    eta_actual = [Optimise.apply!(o, p, [1.0])[1] for _ in steps]
    @test eta_actual == eta_expected
  end

  w = randn(10, 10)
  o = ExpDecay(0.1, 0.1, 1000, 1e-4)
  w1 = randn(10,10)
  loss(x) = Flux.Losses.mse(w*x, w1*x)
  flag = 1
  decay_steps = []
  for t = 1:10^5
    prev_eta = o.eta
    θ = Params([w1])
    x = rand(10)
    θ̄ = gradient(() -> loss(x), θ)
    prev_grad = collect(θ̄[w1])
    delta = Optimise.apply!(o, w1, θ̄[w1])
    w1 .-= delta
    new_eta = o.eta
    if new_eta != prev_eta
      push!(decay_steps, t)
    end
    array = fill(o.eta, size(prev_grad))
    if array .* prev_grad != delta
      flag = 0
    end
  end
  @test flag == 1
  # Test to check if decay happens at decay steps. Eta reaches clip value (1e-4) after 4000 steps (decay by 0.1 every 1000 steps starting at 0.1).
  ground_truth = []
  for i in 1:4
    push!(ground_truth, 1000*i)  # Expected decay steps for this example.
  end
  @test decay_steps == ground_truth
  @test o.eta == o.clip
end

@testset "Clipping" begin
    w = randn(10, 10)
    loss(x) = sum(w * x)
    θ = Params([w])
    x = 1000 * randn(10)
    w̄ = gradient(() -> loss(x), θ)[w]
    w̄_value = Optimise.apply!(ClipValue(1.0), w, copy(w̄))
    @test all(w̄_value .<= 1)
    w̄_norm = Optimise.apply!(ClipNorm(1.0), w, copy(w̄))
    @test norm(w̄_norm) <= 1
end
