"""
    Chain(layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.

# Examples
```jldoctest
julia> m = Chain(x -> x^2, x -> x+1);

julia> m(5) == 26
true

julia> m = Chain(Dense(10, 5), Dense(5, 2));

julia> x = rand(10);

julia> m(x) == m[2](m[1](x))
true
```
"""
struct Chain{T<:Tuple}
  layers::T
  Chain(xs...) = new{typeof(xs)}(xs)
end

@forward Chain.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex

functor(::Type{<:Chain}, c) = c.layers, ls -> Chain(ls...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(c.layers, x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

testmode!(m::Chain, mode = true) = (map(x -> testmode!(x, mode), m.layers); m)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

"""
    outdims(c::Chain, isize)

Calculate the output dimensions given the input dimensions, `isize`.

```julia
m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32))
outdims(m, (10, 10)) == (6, 6)
```
"""
outdims(c::Chain, isize) = foldr(outdims, reverse(c.layers), init = isize)

# This is a temporary and naive implementation
# it might be replaced in the future for better performance
# see issue https://github.com/FluxML/Flux.jl/issues/702
# Johnny Chen -- @johnnychen94
# only slightly changed to better handle interaction with Zygote @dsweber2
"""
    activations(c::Chain, input)

Calculate the forward results of each layers in Chain `c` with `input` as model input.
"""
function activations(c::Chain, input)
    extraChain(c.layers, input)
end

function extraChain(fs::Tuple, x)
    res = first(fs)(x)
    return (res, extraChain(Base.tail(fs), res)...)
end

extraChain(::Tuple{}, x) = ()



"""
    Dense(in::Integer, out::Integer, σ = identity)

Create a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

# Example
```
julia> d = Dense(5, 2)
Dense(5, 2)

julia> d(rand(5))
2-element Array{Float32,1}:
 -0.16210233
  0.123119034
```
"""
struct Dense{F,S<:AbstractArray,T<:AbstractArray}
  W::S
  b::T
  σ::F
end

Dense(W, b) = Dense(W, b, identity)

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return Dense(initW(out, in), initb(out), σ)
end

@functor Dense

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(a::Dense{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Dense{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

"""
    outdims(l::Dense, isize)

Calculate the output dimensions given the input dimensions, `isize`.

```julia
m = Dense(10, 5)
outdims(m, (5, 2)) == (5,)
outdims(m, (10,)) == (5,)
```
"""
function outdims(l::Dense, isize)
    first(isize) == size(l.W, 2) || throw(DimensionMismatch("input size should equal to ($(size(l.W, 2)),), got $isize"))
    return (size(l.W, 1),)
end

"""
    Diagonal(in::Integer)

Create an element-wise linear transformation layer with learnable
vectors `α` and `β`:

    y = α .* x .+ β

The input `x` must be a array where `size(x, 1) == in`.
"""
struct Diagonal{T}
  α::T
  β::T
end

Diagonal(in::Integer; initα = ones, initβ = zeros) =
  Diagonal(initα(in), initβ(in))

@functor Diagonal

function (a::Diagonal)(x)
  α, β = a.α, a.β
  α.*x .+ β
end

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", length(l.α), ")")
end

outdims(l::Diagonal, isize) = (length(l.α),)

"""
    Maxout(over)

The [Maxout](https://arxiv.org/abs/1302.4389) layer has a number of
internal layers which all receive the same input. It returns the elementwise
maximum of the internal layers' outputs.

Maxout over linear dense layers satisfies the univeral approximation theorem.
"""
struct Maxout{FS<:Tuple}
    over::FS
end

"""
    Maxout(f, n_alts)

Construct a Maxout layer over `n_alts` instances of the layer given by `f`.
The function takes no arguments and should return some callable layer.
Conventionally, this is a linear dense layer.

# Examples

This constructs a `Maxout` layer over 4 internal dense linear layers, each
identical in structure (784 inputs, 128 outputs):
```julia
insize = 784
outsize = 128
Maxout(()->Dense(insize, outsize), 4)
```
"""
function Maxout(f, n_alts)
  over = Tuple(f() for _ in 1:n_alts)
  return Maxout(over)
end

@functor Maxout

function (mo::Maxout)(input::AbstractArray)
    mapreduce(f -> f(input), (acc, out) -> max.(acc, out), mo.over)
end

outdims(l::Maxout, isize) = outdims(first(l.over), isize)

"""
    SkipConnection(layer, connection)

Create a skip connection which consists of a layer or `Chain` of consecutive
layers and a shortcut connection linking the block's input to the output
through a user-supplied 2-argument callable. The first argument to the callable
will be propagated through the given `layer` while the second is the unchanged,
"skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`,
and requires the output of the layers to be the same shape as the input.
Here is a more complicated example:
```julia
m = Conv((3,3), 4=>7, pad=(1,1))
x = ones(5,5,4,10);
size(m(x)) == (5, 5, 7, 10)

sm = SkipConnection(m, (mx, x) -> cat(mx, x, dims=3))
size(sm(x)) == (5, 5, 11, 10)
```
"""
struct SkipConnection
  layers
  connection  #user can pass arbitrary connections here, such as (a,b) -> a + b
end

@functor SkipConnection

function (skip::SkipConnection)(input)
  skip.connection(skip.layers(input), input)
end

function Base.show(io::IO, b::SkipConnection)
  print(io, "SkipConnection(", b.layers, ", ", b.connection, ")")
end
