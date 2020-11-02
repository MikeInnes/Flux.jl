import Adapt: adapt, adapt_storage
using  LinearAlgebra: Cholesky
using Zygote: IdSet
import Functors: @functor, functor, fmap

trainable(m) = functor(m)[1]

"""
    testmode!(m, mode = true)

Set a layer or model's test mode (see below).
Using `:auto` mode will treat any gradient computation as training.

_Note_: if you manually set a model into test mode, you need to manually place
it back into train mode during training phase.

Possible values include:
- `false` for training
- `true` for testing
- `:auto` or `nothing` for Flux to detect the mode automatically
"""
testmode!(m, mode = true) = m

"""
    trainmode!(m, mode = true)

Set a layer of model's train mode (see below).
Symmetric to [`testmode!`](@ref) (i.e. `trainmode!(m, mode) == testmode!(m, !mode)`).

_Note_: if you manually set a model into train mode, you need to manually place
it into test mode during testing phase.

Possible values include:
- `true` for training
- `false` for testing
- `:auto` or `nothing` for Flux to detect the mode automatically
"""
trainmode!(m, mode = true) = mode isa Bool ? testmode!(m, !mode) : testmode!(m, mode)

params!(p::Params, x::AbstractArray{<:Number}, seen = IdSet()) = push!(p, x)

function params!(p::Params, x, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for child in trainable(x)
    params!(p, child, seen)
  end
end

function params(m...)
  ps = Params()
  params!(ps, m)
  return ps
end

namedparams!(p, x::AbstractArray{<:Number}, name = :nothing, seen = IdSet()) = p[x] = name

function namedparams!(p, x, name = :nothing, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for (name, child) in pairs(trainable(x))
    name = name isa Symbol ? string(name) : ""
    namedparams!(p, child, name, seen)
  end
end

"""
    namedparams(m...)

Gate named parameters of the model

# Examples
```julia
julia> Flux.namedparams(Chain(LSTM(1, 1)))
IdDict{Any,String} with 5 entries:
  Float32[0.0415536; 0.334981; 0.49804; 0.258496]    => "Wi"
  Float32[0.0]                                       => ""
  Float32[0.336735; -0.167129; -0.695561; -0.743882] => "Wh"
  Float32[-0.979939, 1.0, 1.06309, -0.0981269]       => "b"
  Float32[0.0]                                       => ""
```
"""
function namedparams(m...)
  ps = IdDict{Any, String}()
  namedparams!(ps, m, :nothing)
  return ps
end

# Deprecated stuff
macro treelike(args...)
  functorm(args...)
end
mapleaves(f, x) = fmap(f, x)

function loadparams!(m, xs)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end

# CPU/GPU movement conveniences

cpu(m) = fmap(x -> adapt(Array, x), m)

gpu(x) = use_cuda[] ? fmap(CUDA.cu, x) : x

# Precision

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs)

paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)

f32(m) = paramtype(Float32, m)
f64(m) = paramtype(Float64, m)

# Functors for certain Julia data structures
@functor Cholesky
trainable(c::Cholesky) = ()