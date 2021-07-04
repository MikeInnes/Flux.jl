istraining() = false

@adjoint istraining() = true, _ -> nothing

_isactive(m) = isnothing(m.active) ? istraining() : m.active

_dropout_shape(s, ::Colon) = size(s)
_dropout_shape(s, dims) = tuple((i ∉ dims ? 1 : si for (i, si) ∈ enumerate(size(s)))...)

_dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

"""
    dropout(x, p; dims=:, active=true)

The dropout function. If `active` is `true`,
for each input, either sets that input to `0` (with probability
`p`) or scales it by `1 / (1 - p)`. `dims` specifies the unbroadcasted dimensions,
e.g. `dims=1` applies dropout along columns and `dims=2` along rows.
This is used as a regularisation, i.e. it reduces overfitting during training.

If `active` is `false`, it just returns the input `x`.

Warning: when using this function, you have to manually manage the activation
state. Usually in fact, dropout is used while training
but is deactivated in the inference phase. This can be
automatically managed using the [`Dropout`](@ref) layer instead of the
`dropout` function.

The [`Dropout`](@ref) layer is what you should use in most scenarios.
"""
function dropout(x, p; dims=:, active::Bool=true)
  active || return x
  y = dropout_mask(x, p, dims=dims)
  return x .* y
end

@adjoint function dropout(x, p; dims=:, active::Bool=true)
  active || return x, Δ -> (Δ, nothing)
  y = dropout_mask(x, p, dims=dims)
  return x .* y, Δ -> (Δ .* y, nothing)
end

function dropout_mask(x, p; dims=:)
  y = rand!(similar(x, _dropout_shape(x, dims)))
  y .= _dropout_kernel.(y, p, 1 - p)
  return y
end

"""
    Dropout(p; dims=:)

Dropout layer. In the forward pass, apply the [`Flux.dropout`](@ref) function on the input.

Does nothing to the input once [`Flux.testmode!`](@ref) is set to `true`.
To apply dropout along certain dimension(s), specify the `dims` keyword.
e.g. `Dropout(p; dims = 3)` will randomly zero out entire channels on WHCN input
(also called 2D dropout).

Does nothing to the input once [`Flux.testmode!`](@ref) is `true`.
"""
mutable struct Dropout{F,D}
  p::F
  dims::D
  active::Union{Bool, Nothing}
end

function Dropout(p; dims=:)
  @assert 0 ≤ p ≤ 1
  Dropout(p, dims, nothing)
end

function (a::Dropout)(x)
  _isactive(a) || return x
  return dropout(x, a.p; dims=a.dims, active=true)
end

testmode!(m::Dropout, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, d::Dropout)
  print(io, "Dropout(", d.p)
  d.dims != (:) && print(io, ", dims = $(repr(d.dims))")
  print(io, ")")
end

"""
    AlphaDropout(p)

A dropout layer. Used in
[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).
The AlphaDropout layer ensures that mean and variance of activations
remain the same as before.

Does nothing to the input once [`testmode!`](@ref) is true.
"""
mutable struct AlphaDropout{F}
  p::F
  active::Union{Bool, Nothing}
  function AlphaDropout(p, active = nothing)
    @assert 0 ≤ p ≤ 1
    new{typeof(p)}(p, active)
  end
end

function (a::AlphaDropout)(x)
  _isactive(a) || return x
  λ = eltype(x)(1.0507009873554804934193349852946)
  α = eltype(x)(1.6732632423543772848170429916717)
  α1 = eltype(x)(-λ*α)
  noise = randn(eltype(x), size(x))
  x = @. x*(noise > (1 - a.p)) + α1 * (noise < (1 - a.p))
  A = sqrt(a.p + a.p * (1 - a.p) * α1^2)
  B = -A * α1 * (1 - a.p)
  x = @. A * x + B
  return x
end

testmode!(m::AlphaDropout, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

"""
    LayerNorm(sz, λ = identity; affine = Diagonal(sz...), ϵ = 1fe-5)

A [normalisation layer](https://arxiv.org/abs/1607.06450) designed to be
used with recurrent hidden states. 
The argument `sz` should be an integer or a tuple of integers. 
In the forward pass, the layer normalises the mean and standard 
deviation of the input, the applied the elementwise activation `λ`.
The input is normalised along the first `length(sz)` dimensions
for tuple `sz`, along the first dimension for integer `sz`.
The input  is expected to have first dimensions' size equal to `sz`. 

By default, LayerNorm also applies a learnable shift and rescaling
as in the [`Diagonal`](@ref) layer. To disable this, pass `affine = identity`.


Se also [`BatchNorm`](@ref), [`InstanceNorm`](@ref), [`GroupNorm`](@ref), and [`normalise`](@ref).
"""
struct LayerNorm{F,D,T,S}
  λ::F
  diag::D
  ϵ::T
  sz::S
end

function LayerNorm(sz, λ = identity; affine = Diagonal(sz...), ϵ = 1f-5)
  # diag = affine ? Diagonal(sz...) : identity
  return LayerNorm(λ, affine, ϵ, sz)
end

@functor LayerNorm

function (a::LayerNorm)(x)
  x = normalise(x, dims = 1:length(a.sz), ϵ = a.ϵ)
  a.λ.(a.diag(x))
end

function Base.show(io::IO, l::LayerNorm)
  print(io, "LayerNorm($(l.size)")
  print(io, ", $(l.λ)")
  af = l.diag == identity ? false : true
  print(io, ", affine = $(af)")
  print(io, ")")
end

struct NormConfig{A,T}
  dims::Vector{Int}
end

NormConfig(affine, track_stats, reduce_dims) = NormConfig{affine, track_stats}(reduce_dims)

getaffine(nc::NormConfig{true}, sz_x; dims = length(sz_x) - 1) =
  ntuple(i -> i in dims ? sz_x[i] : 1, length(sz_x))

getaffine(nc::NormConfig{false}, args...; kwargs...) = ()

# For InstanceNorm, GroupNorm, and BatchNorm.
# Compute the statistics on the slices specified by reduce_dims.
# reduce_dims=[1,...,N-2,N] for BatchNorm
# reduce_dims=[1,...,N-2] for InstanceNorm and GroupNorm
function norm_forward(l, x::AbstractArray{T,N}, nc::NormConfig{A, true}) where {A, T, N}
  if !_isactive(l) # testmode with tracked stats
    stats_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    μ = reshape(l.μ, stats_shape)
    σ² = reshape(l.σ², stats_shape)
  else  # trainmode or testmode without tracked stats
    μ = mean(x; dims = nc.dims)
    σ² = mean((x .- μ) .^ 2; dims = nc.dims) # ./ l.chs

    μnew, σ²new = track_stats(x, (l.μ, l.σ²), (μ,σ²),
                              l.momentum, reduce_dims = nc.dims)

    Zygote.ignore() do
      l.μ = reshape(μnew, :)
      l.σ² = reshape(σ²new, :)
    end
  end
  μ, σ²
end

function norm_forward(l, x::AbstractArray{T,N}, nc::NormConfig{A, false}) where {A, T, N}
  μ = mean(x; dims = nc.dims)
  σ² = mean((x .- μ) .^ 2; dims = nc.dims)
  μ, σ²
end

function track_stats(x::AbstractArray{T,N}, (μprev, σ²prev), (μ, σ²), mtm; reduce_dims) where {T,N}
  m = prod(size(x)[collect(reduce_dims)])
  μnew = vec((N in reduce_dims) ? μ : mean(μ, dims = N))
  σ²new = vec((N in reduce_dims) ? σ² : mean(σ², dims = N))
  μ_ = (1 - mtm) .* μprev .+ mtm .* μnew
  σ²_ = (1 - mtm) .* σ²prev .+ mtm .* (m / (m - one(T))) .* σ²new
  μ_, σ²_
end
@nograd track_stats

function affine(l, x::AbstractArray{T,N}, μ, σ², nc::NormConfig{true}; dims = N - 1) where {T,N}
  affine_shape = getaffine(nc, size(x), dims = dims)
  γ = reshape(l.γ, affine_shape)
  β = reshape(l.β, affine_shape)
  x̂ = (x .- μ) ./ sqrt.(σ² .+ l.ϵ)
  l.λ.(γ .* x̂ .+ β)
end

function affine(l, x, μ, σ², nc::NormConfig{false}; dims = :) 
  l.λ.((x .- μ) ./ sqrt.(σ² .+ l.ϵ))
end

# function affine(l, x, μ, σ², affine_shape)
#   res = (x .- μ) ./ sqrt.(σ² .+ l.ϵ)
#   _affine(l.λ, res, affine_shape)
# end

"""
    BatchNorm(channels::Integer, λ = identity;
              initβ = zeros, initγ = ones,
              ϵ = 1f-5, momentum = 0.1f0)

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.
`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension.

`BatchNorm` computes the mean and variance for each `D_1×...×D_{N-2}×1×D_N` 
input slice and normalises the input accordingly.

If `affine = true`, it also applies  a shift and a rescale to the input 
through to learnable per-channel bias β and scale γ parameters.

After normalisation, elementwise activation `λ` is applied.  

If `track_stats = true`, accumulates mean and var statistics in training phase 
that will be used to renormalize the input in test phase.

Use [`testmode!`](@ref) during inference.

# Examples
```julia
m = Chain(
  Dense(28^2, 64),
  BatchNorm(64, relu),
  Dense(64, 10),
  BatchNorm(10),
  softmax)
```
"""
mutable struct BatchNorm{F,V,N,W}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W     # moving mean
  σ²::W    # moving var
  ϵ::N
  momentum::N
  affine::Bool
  track_stats::Bool
  active::Union{Bool, Nothing}
  chs::Int # number of channels
end

function BatchNorm(chs::Int, λ = identity;
                   initβ = i -> zeros(Float32, i), 
                   initγ = i -> ones(Float32, i), 
                   affine = true, track_stats = true,
                   ϵ = 1f-5, momentum = 0.1f0)

  β = initβ(chs)
  γ = initγ(chs)
  μ = zeros(Float32, chs)
  σ² = ones(Float32, chs)

  BatchNorm(λ, β, γ,
            μ, σ², ϵ, momentum, 
            affine, track_stats, 
            nothing, chs)
end

@functor BatchNorm
trainable(bn::BatchNorm) = bn.affine ? (bn.β, bn.γ) : ()

function (BN::BatchNorm)(x)
  N = ndims(x)::Int
  @assert size(x, N - 1) == BN.chs
  reduce_dims = [1:N-2; N]
  nc = NormConfig(BN.affine, BN.track_stats, reduce_dims)
  μ, σ² = norm_forward(BN, x, nc)
  affine(BN, x, μ, σ², nc)
end

testmode!(m::BatchNorm, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(l.chs)")
  print(io, ", $(l.λ)")
  print(io, ", affine = $(l.affine)")
  print(io, ")")
end


"""
    InstanceNorm(channels::Integer, λ = identity;
                 initβ = zeros, initγ = ones,
                 affine = false, track_stats = false,
                 ϵ = 1f-5, momentum = 0.1f0)

[Instance Normalization](https://arxiv.org/abs/1607.08022) layer.
`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N > 2` dimensions, call the `N-1`th the channel dimension. 
For `WHCN` images it's the usual channel dimension.

`InstanceNorm` computes the mean and variance for each `D_1×...×D_{N-2}×1×1` 
input slice and normalises the input accordingly.

If `affine=true`, it also applies  a shift and a rescale to the input 
through to learnable per-channel bias `β` and scale `γ` parameters.

If `track_stats=true`, accumulates mean and var statistics in training phase 
that will be used to renormalize the input in test phase.

**Warning**: the defaults for `affine` and `track_stats` used to be `true` 
in previous Flux versions (< v0.12).
"""
mutable struct InstanceNorm{F,V,N,W}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving var
  ϵ::N
  momentum::N
  affine::Bool
  track_stats::Bool
  active::Union{Bool, Nothing}
  chs::Int # number of channels
end

function InstanceNorm(chs::Int, λ = identity;
                    initβ = i -> zeros(Float32, i), 
                    initγ = i -> ones(Float32, i), 
                    affine = true, track_stats = true,
                    ϵ = 1f-5, momentum = 0.1f0)

  β = initβ(chs)
  γ = initγ(chs)
  μ = zeros(Float32, chs)
  σ² = ones(Float32, chs)
  InstanceNorm(λ, β, γ,
               μ, σ², ϵ, momentum, 
               affine, track_stats,
               nothing, chs)
end

@functor InstanceNorm
trainable(in::InstanceNorm) = in.affine ? (in.β, in.γ) : ()

function (l::InstanceNorm)(x)
  @assert ndims(x) > 2
  @assert size(x, ndims(x)-1) == l.chs
  N = ndims(x)
  reduce_dims = 1:N-2
  nc = NormConfig(l.affine, l.track_stats, reduce_dims)
  μ, σ² = norm_forward(l, x, nc)
  affine(l, x, μ, σ², nc)
end

testmode!(m::InstanceNorm, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, l::InstanceNorm)
  print(io, "InstanceNorm($(l.chs)")
  print(io, ", $(l.λ)")
  print(io, ", affine = $(l.affine)")
  print(io, ")")
end

"""
    GroupNorm(channels::Integer, G::Integer, λ=identity;
              initβ = (i) -> zeros(Float32, i), 
              initγ = (i) -> ones(Float32, i),
              affine=true, track_stats=false,
              ϵ=1f-5, momentum=0.1f0)

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

`chs` is the number of channels, the channel dimension of your input.
For an array of N dimensions, the `N-1`th index is the channel dimension.

`G` is the number of groups along which the statistics are computed.
The number of channels must be an integer multiple of the number of groups.

`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N > 2` dimensions, call the `N-1`th the channel dimension. 
For `WHCN` images it's the usual channel dimension.

If `affine=true`, it also applies  a shift and a rescale to the input 
through to learnable per-channel bias `β` and scale `γ` parameters.

If `track_stats=true`, accumulates mean and var statistics in training phase 
that will be used to renormalize the input in test phase.
"""
mutable struct GroupNorm{F,V,N,W}
  G::Int  # number of groups
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W     # moving mean
  σ²::W    # moving std
  ϵ::N
  momentum::N
  affine::Bool
  track_stats::Bool
  active::Union{Bool, Nothing}
  chs::Int # number of channels
end

@functor GroupNorm
trainable(gn::GroupNorm) = gn.affine ? (gn.β, gn.γ) : ()

function GroupNorm(chs::Int, G::Int, λ = identity;
                   initβ = i -> zeros(Float32, i), 
                   initγ = i -> ones(Float32, i), 
                   affine = true, track_stats = false,
                   ϵ = 1f-5, momentum = 0.1f0) 

  chs % G == 0 || error("The number of groups ($(G)) must divide the number of channels ($chs)")

  β = initβ(chs)
  γ = initγ(chs)
  μ = zeros(Float32, G)
  σ² = ones(Float32, G)

  GroupNorm(G, λ, 
            β, γ,
            μ, σ², 
            ϵ, momentum, 
            affine, track_stats, 
            nothing, chs)
end

function (gn::GroupNorm)(x)
  @assert ndims(x) > 2
  @assert size(x, ndims(x) - 1) == gn.chs
  sz = size(x)
  N = ndims(x)
  x = reshape(x, sz[1:N-2]..., sz[N-1] ÷ gn.G, gn.G, sz[N])
  n = ndims(x)
  reduce_dims = 1:n-2
  nc = NormConfig(gn.affine, gn.track_stats, reduce_dims)
  μ, σ² = norm_forward(gn, x, nc)
  res = affine(gn, x, μ, σ², nc, dims = (n - 1, n - 2))
  return reshape(res, sz)
end

testmode!(m::GroupNorm, mode = true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, l::GroupNorm)
  print(io, "GroupNorm($(l.chs), $(l.G)")
  print(io, ", $(l.λ)")
  print(io, ", affine = $(l.affine)")
  print(io, ")")
end
