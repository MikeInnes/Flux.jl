using Test
using Flux
using Flux.Losses: ctc_loss
using Zygote: gradient
using LinearAlgebra

# Custom function to check numerical gradient of ctc loss,
# based on `ngradient` in `Tracker.jl`
function ctc_ngradient(x, y)
  f = Flux.Losses.ctc_loss
  grads = zero(x)
  for i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(x, y)
    x[i] = tmp + δ/2
    y2 = f(x, y)
    x[i] = tmp
    grads[i] = (y2-y1)/δ
  end
  return grads
end

@testset "ctc-gpu" begin
  x = rand(10, 50)
  y = rand(1:9, 30)
  x_gpu = gpu(x)
  g1 = gradient(ctc_loss, x_gpu, y)[1]
  g1 = g1 |> collect
  g2 = ctc_ngradient(x, y)
  @test g1 ≈ g2 rtol=1e-5 atol=1e-5

  # test that GPU loss matches CPU implementation
  l1 = ctc_loss(x_gpu, y)
  l2 = ctc_loss(x, y)
  @test l1 ≈ l2

  # tests using hand-calculated values
  x_gpu = [1. 2. 3.; 2. 1. 1.; 3. 3. 2.] |> gpu
  y = [1, 2]
  @test ctc_loss(x_gpu, y) ≈ 3.6990738275138035

  g = [-0.317671 -0.427729 0.665241; 0.244728 -0.0196172 -0.829811; 0.0729422 0.447346 0.16457]
  ghat = gradient(ctc_loss, x_gpu, y)[1] |> collect
  @test g ≈ ghat rtol=1e-5 atol=1e-5

  x_gpu = [-3. 12. 8. 15.; 4. 20. -2. 20.; 8. -33. 6. 5.] |> gpu
  y = [1, 2] |> gpu
  @test ctc_loss(x_gpu, y) ≈ 8.02519869363453

  g = [-2.29294774655333e-06 -0.999662657278862 1.75500863563993e-06 0.00669284889063; 0.017985914969696 0.999662657278861 -1.9907078755387e-06 -0.006693150917307; -0.01798362202195 -2.52019580677916e-20 2.35699239251042e-07 3.02026677058789e-07]
  ghat = gradient(ctc_loss, x_gpu, y)[1] |> collect
  @test g ≈ ghat rtol=1e-5 atol=1e-5
end
