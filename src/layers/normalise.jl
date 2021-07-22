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
    LayerNorm(sz, λ=identity; affine=true, ϵ=1fe-5)

A [normalisation layer](https://arxiv.org/abs/1607.06450) designed to be
used with recurrent hidden states. 
The argument `sz` should be an integer or a tuple of integers. 
In the forward pass, the layer normalises the mean and standard 
deviation of the input, the applied the elementwise activation `λ`.
The input is normalised along the first `length(sz)` dimensions
for tuple `sz`, along the first dimension for integer `sz`.
The input  is expected to have first dimensions' size equal to `sz`. 

If `affine=true` also applies a learnable shift and rescaling
as in the [`Diagonal`](@ref) layer.


Se also [`BatchNorm`](@ref), [`InstanceNorm`](@ref), [`GroupNorm`](@ref), and [`normalise`](@ref).
"""
struct LayerNorm{F,D,T,N}
  λ::F
  diag::D
  ϵ::T
  size::NTuple{N,Int}
  affine::Bool
end

function LayerNorm(sz, λ=identity; affine=true, ϵ=1f-5)
  sz = sz isa Integer ? (sz,) : sz
  diag = affine ? Diagonal(sz...) : nothing
  return LayerNorm(λ, diag, ϵ, sz, affine)
end

@functor LayerNorm

function (a::LayerNorm)(x)
  x = normalise(x, dims=1:length(a.size), ϵ=a.ϵ)
  a.diag === nothing ? a.λ.(x) : a.λ.(a.diag(x))
end

function Base.show(io::IO, l::LayerNorm)
  print(io, "LayerNorm($(l.size)")
  l.λ == identity || print(io, ", $(l.λ)")
  hasaffine(l) || print(io, ", affine=false")
  print(io, ")")
end

# For InstanceNorm, GroupNorm, and BatchNorm.
# Compute the statistics on the slices specified by reduce_dims.
# reduce_dims=[1,...,N-2,N] for BatchNorm
# reduce_dims=[1,...,N-2] for InstanceNorm and GroupNorm
function _norm_layer_forward(l, x::AbstractArray{T,N}; reduce_dims, affine_shape) where {T, N}
  isnothing(l.dim) ? dim = N-1 : dim = l.dim
  if !_isactive(l) && l.track_stats # testmode with tracked stats
    stats_shape = ntuple(i -> i == dim ? size(x, dim) : 1, N)
    μ = reshape(l.μ, stats_shape)
    σ² = reshape(l.σ², stats_shape)
  else  # trainmode or testmode without tracked stats
    μ = mean(x; dims=reduce_dims)
    σ² = mean((x .- μ).^2; dims=reduce_dims)
    if l.track_stats
      ## update moving mean/std
      Zygote.ignore() do
        mtm = l.momentum
        m = prod(size(x, i) for i in reduce_dims)  # needed for computing corrected var
        μnew = vec(dim+1 ∈ reduce_dims ? μ : mean(μ, dims=dim+1))
        σ²new = vec(dim+1 ∈ reduce_dims ? σ² : mean(σ², dims=dim+1))
        l.μ = (1-mtm) .* l.μ .+ mtm .* μnew
        l.σ² = (1-mtm) .* l.σ² .+ mtm .* (m / (m - one(eltype(l.σ²)))) .* σ²new
      end
    end
  end
  if hasaffine(l)
    γ = reshape(l.γ, affine_shape)
    β = reshape(l.β, affine_shape)
    return l.λ.(γ .* (x .- μ) ./ sqrt.(σ² .+ l.ϵ) .+ β)
  else
    return l.λ.((x .- μ) ./ sqrt.(σ² .+ l.ϵ))
  end
end

"""
    BatchNorm(channels::Integer, λ=identity;
              dim = nothing,
              initβ=zeros32, initγ=ones32,
              ϵ=1f-5, momentum= 0.1f0)

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.
`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension. Use `dim=dim` to change the channel dimension to `dim`.

`BatchNorm` computes the mean and variance for each `D_1×...×D_{N-2}×1×D_N` 
input slice and normalises the input accordingly.

If `affine=true`, it also applies  a shift and a rescale to the input 
through to learnable per-channel bias β and scale γ parameters.

After normalisation, elementwise activation `λ` is applied.  

If `track_stats=true`, accumulates mean and var statistics in training phase 
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
  dim::Union{Int, Nothing} # channel dimension
end

function BatchNorm(chs::Int, λ=identity;
          dim = nothing,
          initβ=zeros32, initγ=ones32, 
          affine=true, track_stats=true,
          ϵ=1f-5, momentum=0.1f0)

  β = affine ? initβ(chs) : nothing
  γ = affine ? initγ(chs) : nothing
  μ = track_stats ? zeros32(chs) : nothing
  σ² = track_stats ? ones32(chs) : nothing

  return BatchNorm(λ, β, γ,
            μ, σ², ϵ, momentum, 
            affine, track_stats, 
            nothing, chs, dim)
end

@functor BatchNorm
trainable(bn::BatchNorm) = hasaffine(bn) ? (bn.β, bn.γ) : ()

function (BN::BatchNorm)(x)
  N = ndims(x)
  isnothing(BN.dim) ? dim = N-1 : dim = BN.dim
  @assert dim < N 
  @assert size(x, dim) == BN.chs
  reduce_dims = [1:dim-1;dim+1:N]
  affine_shape = ntuple(i -> i == dim ? size(x, dim) : 1, N)
  return _norm_layer_forward(BN, x; reduce_dims, affine_shape)
end

testmode!(m::BatchNorm, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(l.chs)")
  (l.λ == identity) || print(io, ", $(l.λ)")
  hasaffine(l) || print(io,  ", affine=false")
  print(io, ")")
end


"""
    InstanceNorm(channels::Integer, λ=identity;
                 initβ=zeros32, initγ=ones32,
                 affine=false, track_stats=false,
                 ϵ=1f-5, momentum=0.1f0)

[Instance Normalization](https://arxiv.org/abs/1607.08022) layer.
`channels` should be the size of the channel dimension in your data (see below).

Given an array with `N > 2` dimensions, call the `N-1`th the channel dimension. 
For `WHCN` images it's the usual channel dimension.
Use `dim=dim` to change the channel dimension to `dim`.

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
  dim::Union{Int, Nothing} # channel dimension
end

function InstanceNorm(chs::Int, λ=identity;
                    dim = nothing,
                    initβ=zeros32, initγ=ones32,
                    affine=false, track_stats=false,
                    ϵ=1f-5, momentum=0.1f0)

  β = affine ? initβ(chs) : nothing
  γ = affine ? initγ(chs) : nothing
  μ = track_stats ? zeros32(chs) : nothing
  σ² = track_stats ? ones32(chs) : nothing

  return InstanceNorm(λ, β, γ,
            μ, σ², ϵ, momentum, 
            affine, track_stats,
            nothing, chs, dim)
end

@functor InstanceNorm
trainable(in::InstanceNorm) = hasaffine(in) ? (in.β, in.γ) : ()

function (l::InstanceNorm)(x)
  N = ndims(x)
  @assert N > 2
  isnothing(l.dim) ? dim = N-1 : dim = l.dim
  @assert dim < N 
  @assert size(x, dim) == l.chs
  reduce_dims = [1:dim-1;dim+2:N];
  affine_shape = ntuple(i -> i == dim ? size(x, dim) : 1, N)
  return _norm_layer_forward(l, x; reduce_dims, affine_shape)
end

testmode!(m::InstanceNorm, mode=true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, l::InstanceNorm)
  print(io, "InstanceNorm($(l.chs)")
  l.λ == identity || print(io, ", $(l.λ)")
  hasaffine(l) || print(io,  ", affine=false")
  print(io, ")")
end

"""
    GroupNorm(channels::Integer, G::Integer, λ=identity;
              dim = nothing
              initβ=zeros32, initγ=ones32,
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
Use `dim=dim` to change the channel dimension to `dim`.

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
  dim::Union{Int, Nothing} # channel dimension
end

@functor GroupNorm
trainable(gn::GroupNorm) = hasaffine(gn) ? (gn.β, gn.γ) : ()

function GroupNorm(chs::Int, G::Int, λ=identity;
              dim = nothing,
              initβ=zeros32, initγ=ones32, 
              affine=true, track_stats=false,
              ϵ=1f-5, momentum=0.1f0)

  chs % G == 0 || error("The number of groups ($(G)) must divide the number of channels ($chs)")

  β = affine ? initβ(chs) : nothing
  γ = affine ? initγ(chs) : nothing
  μ = track_stats ? zeros32(G) : nothing
  σ² = track_stats ? ones32(G) : nothing

  return GroupNorm(G, λ, 
            β, γ,
            μ, σ², 
            ϵ, momentum, 
            affine, track_stats, 
            nothing, chs, dim)
end

function (gn::GroupNorm)(x)
  N = ndims(x)
  @assert N > 2
  isnothing(gn.dim) ? dim = N-1 : dim = gn.dim
  @assert dim < N 
  @assert size(x, dim) == gn.chs
  sz = size(x)
  x = reshape(x, sz[1:dim-1]..., sz[dim]÷gn.G, gn.G, sz[dim+1:N]...)
  N = ndims(x)
  reduce_dims = [1:dim;dim+3:N];
  affine_shape = ntuple(i -> i ∈ (dim+1, dim) ? size(x, i) : 1, N)
  x = _norm_layer_forward(gn, x; reduce_dims, affine_shape)
  return reshape(x, sz)
end

testmode!(m::GroupNorm, mode = true) =
  (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, l::GroupNorm)
  # print(io, "GroupNorm($(join(size(l.β), ", "))", ", ", l.G)
  print(io, "GroupNorm($(l.chs), $(l.G)")
  l.λ == identity || print(io, ", ", l.λ)
  hasaffine(l) || print(io,  ", affine=false")
  print(io, ")")
end

"""
  hasaffine(l)

Return `true` if a normalisation layer has trainable shift and 
scale parameters, `false` otherwise.

See [`BatchNorm`](@ref), [`InstanceNorm`](@ref), [`GroupNorm`](@ref), and [`LayerNorm`](@ref).
"""
hasaffine(l::Union{BatchNorm, InstanceNorm, LayerNorm, GroupNorm}) = l.affine
