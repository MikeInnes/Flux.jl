"""
    testmode!(m)
    testmode!(m, false)

Put layers like [`Dropout`](@ref) and [`BatchNorm`](@ref) into testing mode
(or back to training mode with `false`).
"""
function testmode!(m, val::Bool=true)
  prefor(x -> _testmode!(x, val), m)
  return m
end

_testmode!(m, test) = nothing

"""
    Dropout(p)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) or scales it by `1/(1-p)`. This is used as a regularisation, i.e. it
reduces overfitting during training.

Does nothing to the input once in [`testmode!`](@ref).
"""
mutable struct Dropout{F}
  p::F
  active::Bool
end

function Dropout(p)
  @assert 0 ≤ p ≤ 1
  Dropout{typeof(p)}(p, true)
end

_dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

function (a::Dropout)(x)
  a.active || return x
  y = similar(x)
  rand!(y)
  y .= _dropout_kernel.(y, a.p, 1 - a.p)
  return x .* y
end

_testmode!(a::Dropout, test) = (a.active = !test)

"""
    LayerNorm(h::Integer)

A [normalisation layer](https://arxiv.org/pdf/1607.06450.pdf) designed to be
used with recurrent hidden states of size `h`. Normalises the mean/stddev of
each input before applying a per-neuron gain/bias.
"""
struct LayerNorm{T}
  diag::Diagonal{T}
end

LayerNorm(h::Integer) =
  LayerNorm(Diagonal(h))

@treelike LayerNorm

(a::LayerNorm)(x) = a.diag(normalise(x))

function Base.show(io::IO, l::LayerNorm)
  print(io, "LayerNorm(", length(l.diag.α), ")")
end

"""
    BatchNorm(channels::Integer, running_var(σ²) = identity;
              initβ(bias) = zeros, initγ(scale) = ones,
              ϵ(eps) = 1e-8, momentum = .1)

Batch Normalization layer. The `channels` input should be the size of the
channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. (For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension.)

`BatchNorm` computes the mean and variance for each each `W×H×1×N` slice and
shifts them to have a new mean and variance (corresponding to the learnable,
per-channel `bias` and `scale` parameters).

See [Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf).

Example:
```julia
m = Chain(
  Dense(28^2, 64),
  BatchNorm(64, relu),
  Dense(64, 10),
  BatchNorm(10),
  softmax)
```
"""
mutable struct BatchNorm{F,V,W,N}
  activation_function::F  # λ
  bias::V  # β
  scale::V  #γ
  running_mean::W  # μ
  running_var::W  #σ²
  eps::N  #ϵ
  momentum::N
  active::Bool
end

BatchNorm(chs::Integer, activation_function = identity;
          initbias = (i) -> zeros(Float32, i), initscale = (i) -> ones(Float32, i), eps = 1f-5, momentum = 0.1f0) =
  BatchNorm(activation_function, param(initbias(chs)), param(initscale(chs)),
            zeros(chs), ones(chs), eps, momentum, true)

function (BN::BatchNorm)(x)
  size(x, ndims(x)-1) == length(BN.bias) ||
    error("BatchNorm expected $(length(BN.bias)) channels, got $(size(x, ndims(x)-1))")
  dims = length(size(x))
  channels = size(x, dims-1)
  affine_shape = ones(Int, dims)
  affine_shape[end-1] = channels
  m = prod(size(x)[1:end-2]) * size(x)[end]
  scale = reshape(BN.scale, affine_shape...)
  bias = reshape(BN.bias, affine_shape...)

  if !BN.active
    running_mean = reshape(BN.running_mean, affine_shape...)
    running_var = reshape(BN.running_var, affine_shape...)
    eps = BN.eps
  else
    T = eltype(x)
    axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
    running_mean = mean(x, dims = axes)
    running_var = sum((x .- running_mean) .^ 2, dims = axes) ./ m
    eps = data(convert(T, BN.eps))
    # update moving mean/std
    mtm = data(convert(T, BN.momentum))
    BN.running_mean = (1 - mtm) .* BN.running_mean .+ mtm .* reshape(data(running_mean), :)
    BN.running_var = (1 - mtm) .* BN.running_var .+ (mtm * m/(m-1)) .* reshape(data(running_var), :) 
  end

  let activation_function = BN.activation_function
    x̂ = (x .- running_mean) ./ sqrt.(running_var .+ eps)
    activation_function.(scale .* x̂ .+ bias)
  end
end

children(BN::BatchNorm) =
  (BN.activation_function, BN.bias, BN.scale, BN.running_mean, BN.running_var, BN.eps, BN.momentum, BN.active)

mapchildren(f, BN::BatchNorm) =  # e.g. mapchildren(cu, BN)
  BatchNorm(BN.activation_function, f(BN.bias), f(BN.scale), f(BN.running_mean), f(BN.running_var), BN.eps, BN.momentum, BN.active)

_testmode!(BN::BatchNorm, test) = (BN.active = !test)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(join(size(l.bias), ", "))")
  (l.activation_function == identity) || print(io, ", activation_function = $(l.activation_function)")
  print(io, ")")
end
