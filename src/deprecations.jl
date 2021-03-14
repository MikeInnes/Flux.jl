# v0.12 deprecations
@deprecate Dropout(p, dims) Dropout(p; dims=dims)
@deprecate InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum) InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, nothing)
@deprecate BatchNorm(λ, β, γ, μ, σ², ϵ, momentum) BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, nothing)
@deprecate GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum) GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum, nothing)

@deprecate outdims(f, inputsize) outputsize(f, inputsize)

@deprecate Conv(; weight,  bias, activation=identity, kws...) Conv(weight, bias, activation; kws...) 
@deprecate ConvTranspose(; weight, bias, activation=identity, kws...) ConvTranspose(weight, bias, activation; kws...) 
@deprecate DepthwiseConv(; weight, bias, activation=identity, kws...) DepthwiseConv(weight, bias, activation; kws...) 

function Base.getproperty(a::Dense, s::Symbol)
  if s === :W
    Base.depwarn("field name dense.W is deprecated in favour of dense.weight", :Dense)
    return getfield(a, :weight)
  elseif s === :b
    Base.depwarn("field name dense.b is deprecated in favour of dense.bias", :Dense)
    return getfield(a, :bias)
  end
  return getfield(a, s)
end

struct Zeros # was used both Dense(10, 2, initb = Zeros) and Dense(rand(2,10), Zeros())
  function Zeros()
    Base.depwarn("Zeros() and Zeros(dims...) are deprecated, please simply use bias=false instead", :Zeros)
    false
  end
end
Zeros(args...) = Zeros()
