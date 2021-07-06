module NilNumber

using NNlib
import Random

"""
    Nil <: Number

Nil is a singleton type with a single instance `nil`.
Unlike `Nothing` and `Missing` it subtypes `Number`.
"""
struct Nil <: Number end

const nil = Nil()

Nil(::T) where T<:Number = nil
(::Type{T})(::Nil) where T<:Number = nil
Base.convert(::Type{Nil}, ::Number) = nil

Base.float(::Type{Nil}) = Nil

for f in [:copy, :zero, :one, :oneunit,
          :+, :-, :abs, :abs2, :inv,
          :exp, :log, :log1p, :log2, :log10,
          :sqrt, :tanh, :conj]
  @eval Base.$f(::Nil) = nil
end

for f in [:+, :-, :*, :/, :^, :mod, :div, :rem]
  @eval Base.$f(::Nil, ::Nil) = nil
end

Base.isless(::Nil, ::Nil) = true
Base.isless(::Nil, ::Number) = true
Base.isless(::Number, ::Nil) = true

Base.isnan(::Nil) = false
Base.isfinite(::Nil) = true
Base.typemin(::Type{Nil}) = nil
Base.typemax(::Type{Nil}) = nil

Base.promote_rule(x::Type{Nil}, y::Type{<:Number}) = Nil

Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{Nil}) = nil

end  # module

using .NilNumber: Nil, nil

"""
    outputsize(m, inputsize::Tuple; padbatch=false)

Calculate the size of the output from model `m`, given the size of the input.
Obeys `outputsize(m, size(x)) == size(m(x))` for valid input `x`.

Keyword `padbatch=true` is equivalent to using `(inputsize..., 1)`, and
returns the final size including this extra batch dimension.

This should be faster than calling `size(m(x))`. It uses a trivial number type,
which should work out of the box for custom layers.

If `m` is a `Tuple` or `Vector`, its elements are applied in sequence, like `Chain(m...)`.

# Examples
```julia-repl
julia> using Flux: outputsize

julia> outputsize(Dense(10, 4), (10,); padbatch=true)
(4, 1)

julia> m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32));

julia> m(randn(Float32, 10, 10, 3, 64)) |> size
(6, 6, 32, 64)

julia> outputsize(m, (10, 10, 3); padbatch=true)
(6, 6, 32, 1)

julia> outputsize(m, (10, 10, 3, 64))
(6, 6, 32, 64)

julia> try outputsize(m, (10, 10, 7, 64)) catch e println(e) end
┌ Error: layer Conv((3, 3), 3=>16), index 1 in Chain, gave an error with input of size (10, 10, 7, 64)
└ @ Flux ~/.julia/dev/Flux/src/outputsize.jl:114
DimensionMismatch("Input channels must match! (7 vs. 3)")

julia> outputsize([Dense(10, 4), Dense(4, 2)], (10, 1)) # Vector of layers becomes a Chain
(2, 1)
```
"""
function outputsize(m, inputsizes::Tuple...; padbatch=false)
  x = nil_input(padbatch, inputsizes...)
  return size(m(x))
end

nil_input(pad::Bool, s::Tuple{Vararg{Integer}}) = pad ? fill(nil, (s...,1)) : fill(nil, s)
nil_input(pad::Bool, multi::Tuple{Vararg{Integer}}...) = nil_input.(pad, multi)
nil_input(pad::Bool, tup::Tuple{Vararg{Tuple}}) = nil_input(pad, tup...)

function outputsize(m::Chain, inputsizes::Tuple{Vararg{Integer}}...; padbatch=false)
  x = nil_input(padbatch, inputsizes...)
  for (i,lay) in enumerate(m.layers)
    try
      x = lay(x)
    catch err
      str = x isa AbstractArray ? "with input of size $(size(x))" : ""
      @error "layer $lay, index $i in Chain, gave an error $str"
      rethrow(err)
    end
  end
  return size(x)
end

"""
    outputsize(m, x_size, y_size, ...; padbatch=false)

For model or layer `m` accepting multiple arrays as input,
this returns `size(m((x, y, ...)))` given `size_x = size(x)`, etc.

# Examples
```jldoctest
julia> x, y = rand(Float32, 5, 64), rand(Float32, 7, 64);

julia> par = Parallel(vcat, Dense(5, 9), Dense(7, 11));

julia> Flux.outputsize(par, (5, 64), (7, 64))
(20, 64)

julia> m = Chain(par, Dense(20, 13), softmax);

julia> Flux.outputsize(m, (5,), (7,); padbatch=true)
(13, 1)

julia> par(x, y) == par((x, y)) == Chain(par, identity)((x, y))
true
```
Notice that `Chain` only accepts multiple arrays as a tuple,
while `Parallel` also accepts them as multiple arguments;
`outputsize` always supplies the tuple.
"""
outputsize

## make tuples and vectors be like Chains

outputsize(m::Tuple, input::Tuple...; padbatch=false) = outputsize(Chain(m...), input...; padbatch=padbatch)
outputsize(m::AbstractVector, input::Tuple...; padbatch=false) = outputsize(Chain(m...), input...; padbatch=padbatch)

## bypass statistics in normalization layers

for layer in (:LayerNorm, :BatchNorm, :InstanceNorm, :GroupNorm)
  @eval (l::$layer)(x::AbstractArray{Nil}) = x
end

## fixes for layers that don't work out of the box

for (fn, Dims) in ((:conv, DenseConvDims), (:depthwiseconv, DepthwiseConvDims))
  @eval begin
    function NNlib.$fn(a::AbstractArray{Nil}, b::AbstractArray{Nil}, dims::$Dims)
      fill(nil, NNlib.output_size(dims)..., NNlib.channels_out(dims), size(a)[end])
    end

    function NNlib.$fn(a::AbstractArray{<:Real}, b::AbstractArray{Nil}, dims::$Dims)
      NNlib.$fn(fill(nil, size(a)), b, dims)
    end

    function NNlib.$fn(a::AbstractArray{Nil}, b::AbstractArray{<:Real}, dims::$Dims)
      NNlib.$fn(a, fill(nil, size(b)), dims)
    end
  end
end

(m::Embedding)(x::AbstractVector{<:Nil}) = fill(nil, size(m.weight, 1), length(x))
(m::Embedding)(x::AbstractArray{<:Nil}) = fill(nil, size(m.weight, 1), size(x)...)
