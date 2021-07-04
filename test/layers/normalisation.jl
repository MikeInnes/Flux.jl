using Flux, Test, Statistics
using Zygote: pullback

evalwgrad(f, x...) = pullback(f, x...)[1]

@testset "Dropout" begin
  x = [1.,2.,3.]
  @test x == Dropout(0.1)(x)
  @test x == evalwgrad(Dropout(0), x)
  @test zero(x) == evalwgrad(Dropout(1), x)

  x = rand(100)
  m = Dropout(0.9)
  y = m(x)
  # By default no dropout is performed outside training
  # @test count(a -> a == 0, y) > 50
  testmode!(m, true)
  y = m(x) # should override istraining
  @test count(a -> a == 0, y) == 0
  testmode!(m, false)
  y = m(x)
  @test count(a -> a == 0, y) > 50

  x = rand(Float32, 100)
  m = Chain(Dense(100,100),
            Dropout(0.9))
  y = m(x)
  # by default no dropout is performed outside training
  # @test count(a -> a == 0, y) > 50
  testmode!(m, true)
  y = m(x) # should override istraining
  @test count(a -> a == 0, y) == 0

  x = rand(100, 50)
  m = Dropout(0.5, dims = 2)
  y = m(x)
  c = map(i -> count(a -> a == 0, @view y[i, :]), 1:100)
  @test minimum(c) == maximum(c)
  m = Dropout(0.5, dims = 1)
  y = m(x)
  c = map(i -> count(a -> a==0, @view y[:, i]), 1:50)
  @test minimum(c) == maximum(c)

  # issue #1084
  m = Dropout(0.9)
  x = rand(100)

  testmode!(m)
  y = m(x)
  @test count(a -> a == 0, y) == 0
  trainmode!(m)
  y = m(x)
  @test count(a -> a == 0, y) > 50

  y = Flux.dropout(x, 0.9, active = true)
  @test count(a -> a == 0, y) > 50

  y = Flux.dropout(x, 0.9, active = false)
  @test count(a -> a == 0, y) == 0
end

@testset "BatchNorm" begin
  let m = BatchNorm(2, track_stats = false), x = reshape(1:6, 1,1,2,3)

    @test m.β == [0, 0]  # initβ(2)
    @test m.γ == [1, 1]  # initγ(2)
    # initial m.σ is 1
    # initial m.μ is 0

    y, _ = pullback((m,x) -> m(x), m, x)
    @test isapprox(y, reshape([-1.22474 0 1.22474; -1.22474 0 1.22474], 1, 1, 2, 3), atol = 1.0e-5)
    # julia> x
    #  2×3 Array{Float64,2}:
    #  1.0  3.0  5.0
    #  2.0  4.0  6.0
    #
    # μ of batch will be
    #  (1. + 3. + 5.) / 3 = 3
    #  (2. + 4. + 6.) / 3 = 4
    #
    # ∴ update rule with momentum:
    #  .1 * 3 + 0 = .3
    #  .1 * 4 + 0 = .4
    m = BatchNorm(2, track_stats = true)
    gs = gradient((m,x) -> sum(m(x)), m, x)
    @test m.μ ≈ reshape([0.3, 0.4], 2, 1)

    # julia> .1 .* var(x, dims = 4, corrected = true) .* (3 / 2).+ .9 .* [1., 1.]
    # 2×1 Array{Float64,2}:
    #  1.3
    #  1.3
    v = mean((0.1 .* var(x, dims = 4, corrected = false)) .* (3 / 2) .+ 0.9 .* [1.0, 1.0], dims = 3) |> x -> dropdims(x, dims = (3,4))
    @test m.σ² ≈ v
   
    x′ = m(x)
    @test isapprox(x′[1], (1 .- 0.3) / sqrt(1.3), atol = 1.0e-5)
  end

  # with activation function
  let m = trainmode!(BatchNorm(3, sigmoid)), x = reshape(1:6, 1,1,3,2)
    y = m(x)
    @test_broken isapprox(y, mean(sigmoid.((x .- m.μ) ./ sqrt.(m.σ² .+ m.ϵ)), dims = 1), atol = 1.0e-7)
  end

  let m = BatchNorm(2), x = reshape(Float32.(1:6), 3, 2, 1)
    y = reshape(permutedims(x, [2, 1, 3]), 2, :)
    y = permutedims(reshape(m(y), 2, 3, 1), [2, 1, 3])
    @test m(x) ≈ y
  end

  let m = BatchNorm(2), x = reshape(Float32.(1:12), 2, 3, 2, 1)
    y = reshape(permutedims(x, [3, 1, 2, 4]), 2, :)
    y = permutedims(reshape(m(y), 2, 2, 3, 1), [2, 3, 1, 4])
    @test m(x) ≈ y
  end

  let m = BatchNorm(2), x = reshape(Float32.(1:24), 2, 2, 3, 2, 1)
    y = reshape(permutedims(x, [4, 1, 2, 3, 5]), 2, :)
    y = permutedims(reshape(m(y), 2, 2, 2, 3, 1), [2, 3, 4, 1, 5])
    @test m(x) ≈ y
  end

  # let m = BatchNorm(32), x = randn(Float32, 416, 416, 32, 1);
  #   m(x)
  #   @test (@allocated m(x)) <  100_000_000
  # end

  # @test length(Flux.params(BatchNorm(10))) == 2
  # @test length(Flux.params(BatchNorm(10, affine=true))) == 2
  # @test length(Flux.params(BatchNorm(10, affine=false))) == 0
end

@testset "InstanceNorm" begin
  # begin tests
  m = InstanceNorm(2; affine = true, track_stats = true)
  sizes = (3, 2, 2)
  x = reshape(1:prod(sizes), sizes)

  # @test length(params(m)) == 2
  x = Float32.(x)
  @test m.β == [0, 0]  # initβ(2)
  @test m.γ == [1, 1]  # initγ(2)
  y, back = pullback((m,x) -> m(x), m, x)

  #julia> x
  #[:, :, 1] =
  # 1.0  4.0
  # 2.0  5.0
  # 3.0  6.0
  #
  #[:, :, 2] =
  # 7.0  10.0
  # 8.0  11.0
  # 9.0  12.0
  #
  # μ will be
  # (1. + 2. + 3.) / 3 = 2.
  # (4. + 5. + 6.) / 3 = 5.
  #
  # (7. + 8. + 9.) / 3 = 8.
  # (10. + 11. + 12.) / 3 = 11.
  #
  # ∴ update rule with momentum:
  # (1. - .1) * 0 + .1 * (2. + 8.) / 2 = .5
  # (1. - .1) * 0 + .1 * (5. + 11.) / 2 = .8
  N = ndims(x)
  @test m.μ ≈ [0.5, 0.8]
  n = prod(size(x,i) for i in 1:N-2)
  corr = n / (n-1)
  σ² = var(x, dims = 1:N-2, corrected = false)
  @test m.σ² ≈ 0.1 * corr * vec(mean(σ², dims = N)) .+ 0.9 * 1

  y = m(x)
  @test length(m.μ) == 2
  @test length(m.σ²) == 2
  @test y ≈ (x .- reshape(m.μ, 1,2,1)) ./ sqrt.(reshape(m.σ², 1,2,1) .+ 1f-5)   atol=1.0e-5

  # with activation function
  let m = InstanceNorm(2, sigmoid; affine = true, track_stats = true), sizes = (3, 2, 2),
      x = reshape(collect(1:prod(sizes)), sizes)
    x = Float64.(x)
    affine_shape = collect(sizes)
    affine_shape[[1,3]] .= 1

    y = m(x)
    y = m(x) # inference time after a training step    
    μ = reshape(m.μ, affine_shape...)
    σ² = reshape(m.σ², affine_shape...)
    @test y ≈ sigmoid.((x .- μ) ./ sqrt.(σ² .+ m.ϵ))   atol=1.0e-7
  end

  # with activation function
  let m = InstanceNorm(2, sigmoid; affine = true, track_stats = false), sizes = (3, 2, 2),
      x = reshape(collect(1:prod(sizes)), sizes)

    x = Float64.(x)
    y = m(x)
    μ = mean(x, dims=1)
    σ² = var(x, dims=1, corrected = false)
    @test y ≈ sigmoid.((x .- μ) ./ sqrt.(σ² .+ m.ϵ))   atol=1.0e-7
  end

  let m = InstanceNorm(2, sigmoid), sizes = (3, 2, 2),
      x = reshape(collect(1:prod(sizes)), sizes)
    
    x = Float64.(x)
    y, back = pullback((m,x) -> m(x), m, x)
    μ = mean(x, dims = 1)
    σ² = var(x, dims = 1, corrected = false)
    @test y ≈ sigmoid.((x .- μ) ./ sqrt.(σ² .+ m.ϵ))   atol=1.0e-7
  end

  # check trainmode!
  let m = trainmode!(InstanceNorm(2; affine = true)), sizes = (2, 4, 1, 2, 3),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
    y = reshape(m(y), sizes...)
    @test m(x) == y
  end

  # check that μ, σ², and the output are the correct size for higher rank tensors
  let m = InstanceNorm(2; affine = true, track_stats = true), sizes = (5, 5, 3, 4, 2, 6),
      x = reshape(Float32.(1:prod(sizes)), sizes)
    y, _ = pullback((m,x) -> m(x), m, x)
    @test size(m.μ) == (sizes[end - 1], )
    @test size(m.σ²) == (sizes[end - 1], )
    @test size(y) == sizes
  end

  # show that instance norm is equal to batch norm when channel and batch dims are squashed
  let m_inorm = trainmode!(InstanceNorm(2; affine=true)), m_bnorm = trainmode!(BatchNorm(12)), sizes = (5, 5, 3, 4, 2, 6),
      x = reshape(Float32.(collect(1:prod(sizes))), sizes)
    @test m_inorm(x) == reshape(m_bnorm(reshape(x, (sizes[1:end - 2]..., :, 1))), sizes)
  end

  # @test length(Flux.params(InstanceNorm(10))) == 0
  # @test length(Flux.params(InstanceNorm(10, affine = true))) == 2
  # @test length(Flux.params(InstanceNorm(10, affine =false))) == 0
end

@testset "LayerNorm" begin
  x = rand(2,3)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims = 1)
  x = rand(2,3,4)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims = 1)
  x = rand(2,3,4,5)
  @test LayerNorm(2)(x) ≈ Flux.normalise(x, dims = 1)
  x = rand(2)
  @test LayerNorm(2, tanh)(x) ≈ tanh.(Flux.normalise(x, dims = 1))

  x = rand(2,3,4,5)
  @test LayerNorm((2,3))(x) ≈ Flux.normalise(x, dims = (1,2))
  x = rand(2,3,4,5)
  @test LayerNorm((2,3,4))(x) ≈ Flux.normalise(x, dims = 1:3)

  m = LayerNorm((2,3,4))
  m = LayerNorm((2,3,4), affine = false)
end

@testset "GroupNorm" begin
  # begin tests
  squeeze(x) = dropdims(x, dims = tuple(findall(size(x) .== 1)...)) # To remove all singular dimensions

  let m = GroupNorm(4,2, track_stats=true), sizes = (3,4,2),
        x = reshape(collect(1:prod(sizes)), sizes)

      # @test length(params(m)) == 2
      x = Float32.(x)
      @test m.β == [0, 0, 0, 0]  # initβ(32)
      @test m.γ == [1, 1, 1, 1]  # initγ(32)

      y, back = pullback((m,x) -> m(x), m, x)

      #julia> x
      #[:, :, 1]  =
      # 1.0  4.0  7.0  10.0
      # 2.0  5.0  8.0  11.0
      # 3.0  6.0  9.0  12.0
      #
      #[:, :, 2] =
      # 13.0  16.0  19.0  22.0
      # 14.0  17.0  20.0  23.0
      # 15.0  18.0  21.0  24.0
      #
      # μ will be
      # (1. + 2. + 3. + 4. + 5. + 6.) / 6 = 3.5
      # (7. + 8. + 9. + 10. + 11. + 12.) / 6 = 9.5
      #
      # (13. + 14. + 15. + 16. + 17. + 18.) / 6 = 15.5
      # (19. + 20. + 21. + 22. + 23. + 24.) / 6 = 21.5
      #
      # μ =
      # 3.5   15.5
      # 9.5   21.5
      #
      # ∴ update rule with momentum:
      # (1. - .1) * 0 + .1 * (3.5 + 15.5) / 2 = 0.95
      # (1. - .1) * 0 + .1 * (9.5 + 21.5) / 2 = 1.55
      @test m.μ ≈ [0.95, 1.55]
      n = prod(size(x)) ÷ m.G ÷ size(x)[end]
      corr = n / (n-1)
      z = reshape(x,3,2,2,2)
      σ² = var(z, dims=(1,2), corrected=false)
      @test m.σ² ≈ 0.1*corr*vec(mean(σ², dims=4)) .+ 0.9 * 1

      y = m(x)
      out = (z .- reshape(m.μ, 1,1,2,1)) ./ sqrt.(reshape(m.σ², 1,1,2,1) .+ 1f-5)
      @test y ≈ reshape(out, size(x))   atol=1.0e-5
  end
  # with activation function
  let m = GroupNorm(4,2, sigmoid, track_stats=true), sizes = (3, 4, 2),
      x = reshape(collect(1:prod(sizes)), sizes)
    x = Float32.(x)
    μ_affine_shape = ones(Int,length(sizes) + 1)
    μ_affine_shape[end-1] = 2 # Number of groups

    affine_shape = ones(Int,length(sizes) + 1)
    affine_shape[end-2] = 2 # Channels per group
    affine_shape[end-1] = 2 # Number of groups
    affine_shape[1] = sizes[1]
    affine_shape[end] = sizes[end]

    og_shape = size(x)

    y = m(x)
    x_ = reshape(x,affine_shape...)
    out = reshape(sigmoid.((x_ .- reshape(m.μ,μ_affine_shape...)) ./ sqrt.(reshape(m.σ²,μ_affine_shape...) .+ m.ϵ)),og_shape)
    @test y ≈ out   atol=1e-7
  end

  let m = trainmode!(GroupNorm(2,2, track_stats=true)), sizes = (2, 4, 1, 2, 3),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
    y = reshape(m(y), sizes...)
    @test m(x) == y
  end

  # check that μ, σ², and the output are the correct size for higher rank tensors
  let m = GroupNorm(4,2, track_stats=true), sizes = (5, 5, 3, 4, 4, 6),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    y = m(x)
    @test size(m.μ) == (m.G,)
    @test size(m.σ²) == (m.G,)
    @test size(y) == sizes
  end

  # show that group norm is the same as instance norm when the group size is the same as the number of channels
  let IN = trainmode!(InstanceNorm(4; affine=true)), GN = trainmode!(GroupNorm(4,4)), sizes = (2,2,3,4,5),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    @test IN(x) ≈ GN(x)
  end

  # show that group norm is the same as batch norm for a group of size 1 and batch of size 1
  let BN = trainmode!(BatchNorm(4)), GN = trainmode!(GroupNorm(4,4)), sizes = (2,2,3,4,1),
      x = Float32.(reshape(collect(1:prod(sizes)), sizes))
    @test BN(x) ≈ GN(x)
  end
end
