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

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

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
    Dense(in => out, σ=identity; bias=true, init=glorot_uniform)
    Dense(W::AbstractMatrix, [bias, σ])

Create a traditional `Dense` layer, whose forward pass is given by:

    y = σ.(W * x .+ bias)

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`

Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`](@doc Flux.glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

# Examples
```jldoctest
julia> d = Dense(5, 2)
Dense(5 => 2)       # 12 parameters

julia> d(rand(Float32, 5, 64)) |> size
(2, 64)

julia> d(rand(Float32, 5, 1, 1, 64)) |> size  # treated as three batch dimensions
(2, 1, 1, 64)

julia> d1 = Dense(ones(2, 5), false, tanh)  # using provided weight matrix
Dense(5 => 2, tanh; bias=false)  # 10 parameters

julia> d1(ones(5))
2-element Vector{Float64}:
 0.9999092042625951
 0.9999092042625951

julia> Flux.params(d1)  # no trainable bias
Params([[1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0]])
```
"""
struct Dense{F, M<:AbstractMatrix, B}
  weight::M
  bias::B
  σ::F
  function Dense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
    b = create_bias(W, bias, size(W,1))
    new{F,M,typeof(b)}(W, b, σ)
  end
end

Dense(in::Integer, out::Integer, σ = identity; kw...) = Dense(in => out, σ; kw...)

function Dense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
               initW = nothing, initb = nothing,
               init = glorot_uniform, bias=true)

  W = if initW !== nothing
    Base.depwarn("keyword initW is deprecated, please use init (which similarly accepts a funtion like randn)", :Dense)
    initW(out, in)
  else
    init(out, in)
  end

  b = if bias === true && initb !== nothing
    Base.depwarn("keyword initb is deprecated, please simply supply the bias vector, bias=initb(out)", :Dense)
    initb(out)
  else
    bias
  end

  return Dense(W, b, σ)
end

@functor Dense

function (a::Dense)(x::AbstractVecOrMat)
  W, b, σ = a.weight, a.bias, a.σ
  return σ.(W*x .+ b)
end

(a::Dense)(x::AbstractArray) = 
  reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.weight, 2), " => ", size(l.weight, 1))
  l.σ == identity || print(io, ", ", l.σ)
  l.bias == Zeros() && print(io, "; bias=false")
  print(io, ")")
end

"""
    Diagonal(α, β)
    Diagonal(size::Integer...)

Create an element-wise linear layer, which performs

    y = α .* x .+ β

The learnable arrays are initialised `α = ones(Float32, size)` and
`β = zeros(Float32, size)`.

Used by [`LayerNorm`](@ref).
"""
struct Diagonal{T}
  α::T
  β::T
end

function Diagonal(sz::Integer...; initα = nothing, initβ = nothing)
  α = if initα !== nothing
    Base.depwarn("keyword initα is deprecated, please simply supply the desired vectors", :Diagonal)
    initα(sz...)
  else
    ones(sz...)
  end
  β = if initβ !== nothing
    Base.depwarn("keyword initβ is deprecated, please simply supply the desired vectors", :Diagonal)
    initβ(sz...)
  else
    zeros(sz...)
  end
  Diagonal(α, β)
end

@functor Diagonal

(a::Diagonal)(x) = a.α .* x .+ a.β

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", join(size(l.α), ", "), ")")
end

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
```jldoctest
julia> insize = 784;

julia> outsize = 128;

julia> Maxout(()->Dense(insize, outsize), 4);
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

"""
    SkipConnection(layer, connection)

Create a skip connection which consists of a layer or `Chain` of consecutive
layers and a shortcut connection linking the block's input to the output
through a user-supplied 2-argument callable. The first argument to the callable
will be propagated through the given `layer` while the second is the unchanged,
"skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.
Here is a more complicated example:
```jldoctest
julia> m = Conv((3,3), 4 => 7, pad=(1,1));

julia> x = ones(Float32, 5, 5, 4, 10);

julia> size(m(x)) == (5, 5, 7, 10)
true

julia> sm = SkipConnection(m, (mx, x) -> cat(mx, x, dims=3));

julia> size(sm(x)) == (5, 5, 11, 10)
true
```
"""
struct SkipConnection{T,F}
  layers::T
  connection::F  #user can pass arbitrary connections here, such as (a,b) -> a + b
end

@functor SkipConnection

function (skip::SkipConnection)(input)
  skip.connection(skip.layers(input), input)
end

function Base.show(io::IO, b::SkipConnection)
  print(io, "SkipConnection(", b.layers, ", ", b.connection, ")")
end

"""
    Bilinear((in1, in2) => out, σ=identity; bias=true, init=glorot_uniform)
    Bilinear(W::AbstractArray, [bias, σ])

Creates a Bilinear layer, which operates on two inputs at the same time.
Its output, given vectors `x` & `y`, is another vector `z` with,
for all `i ∈ 1:out`:

    z[i] = σ(x' * W[i,:,:] * y + bias[i])

If `x` and `y` are matrices, then each column of the output `z = B(x, y)` is of this form,
with `B` the Bilinear layer.

If the second input `y` is not given, it is taken to be equal to `x`, i.e. `B(x) == B(x, x)`

The two inputs may also be provided as a tuple, `B((x, y)) == B(x, y)`,
which is accepted as the input to a `Chain`.

If the two input sizes are the same, `in1 == in2`, then you may write `Bilinear(in => out, σ)`.

The initialisation works as for [`Dense`](@ref) layer, with `W = init(out, in1, in2)`.
By default the bias vector is `zeros(Float32, out)`, option `bias=false` will switch off
trainable bias. Either of these may be provided explicitly.

# Examples
```jldoctest
julia> x, y = randn(Float32, 5, 32), randn(Float32, 5, 32);

julia> B = Flux.Bilinear((5, 5) => 7)
Bilinear(5 => 7)    # 182 parameters

julia> B(x) |> size  # interactions based on one input
(7, 32)

julia> B(x,y) == B((x,y))  # two inputs, may be given as a tuple
true

julia> sc = SkipConnection(
                Chain(Dense(5 => 20, tanh), Dense(20, 9, tanh)),
                Flux.Bilinear((9, 5) => 3, bias=false),
            );  # used as the recombinator, with skip as the second input

julia> sc(x) |> size
(3, 32)

julia> Flux.Bilinear(rand(4,8,16), false, tanh)  # first dim of weight is the output
Bilinear((8, 16) => 4, tanh; bias=false)  # 512 parameters
```
"""
struct Bilinear{F,A,B}
  weight::A
  bias::B
  σ::F
  function Bilinear(W::A, bias = true, σ::F = identity) where {A<:AbstractArray, F}
    ndims(A) == 3 || throw(ArgumentError("expected a 3-array of weights"))
    b = create_bias(W, bias, size(W,1))
    new{F,A,typeof(b)}(W, b, σ)
  end
end

@functor Bilinear

function Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity;
                  init = glorot_uniform, bias = true)
  Bilinear(init(out, in1, in2), bias, σ)
end

Bilinear(((in1, in2), out)::Pair{<:Tuple, <:Integer}, σ = identity; kw...) = Bilinear(in1, in2, out, σ; kw...)
Bilinear((in12, out)::Pair{<:Integer, <:Integer}, σ = identity; kw...) = Bilinear(in12, in12, out, σ; kw...)

function (a::Bilinear)(x::AbstractMatrix, y::AbstractMatrix)
  W, b, σ = a.weight, a.bias, a.σ

  d_z, d_x, d_y = size(W)
  d_x == size(x,1) && d_y == size(y,1) || throw(DimensionMismatch("number of rows in data must match W"))
  size(x,2) == size(y,2) || throw(DimensionMismatch("Data inputs must agree on number of columns, got $(size(x,2)) and $(size(y,2))"))

  # @einsum Wy[o,i,s] := W[o,i,j] * y[j,s]
  Wy = reshape(reshape(W, (:, d_y)) * y, (d_z, d_x, :))

  # @einsum Z[o,s] := Wy[o,i,s] * x[i,s]
  Wyx = batched_mul(Wy, reshape(x, (d_x, 1, :)))
  Z = reshape(Wyx, (d_z, :))

  # @einsum out[o,s] := σ(Z[o,i] + b[o])
  σ.(Z .+ b)
end

(a::Bilinear)(x::AbstractVecOrMat) = a(x, x)
(a::Bilinear)(x::AbstractVector, y::AbstractVector) = vec(a(reshape(x, :,1), reshape(y, :,1)))
(a::Bilinear)(x::NTuple{2, AbstractArray}) = a(x[1], x[2])

function Base.show(io::IO, l::Bilinear)
  if size(l.weight, 2) == size(l.weight, 3)
    print(io, "Bilinear(", size(l.weight, 2), " => ", size(l.weight, 1))
  else
    print(io, "Bilinear((", size(l.weight, 2), ", ", size(l.weight, 3), ") => ", size(l.weight, 1))
  end
  l.σ == identity || print(io, ", ", l.σ)
  l.bias == Flux.Zeros() && print(io, "; bias=false")
  print(io, ")")
end

"""
    Parallel(connection, layers...)

Create a 'Parallel' layer that passes an input array to each path in
`layers`, reducing the output with `connection`.

Called with one input `x`, this is equivalent to `reduce(connection, [l(x) for l in layers])`.
If called with multiple inputs, they are `zip`ped with the layers, thus `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

# Examples

```jldoctest
julia> model = Chain(Dense(3 => 5),
                     Parallel(vcat, Dense(5 => 4), Chain(Dense(5 => 7), Dense(7 => 4))),
                     Dense(8 => 17));

julia> size(model(rand(3)))
(17,)

julia> model = Parallel(+, Dense(10, 2), Dense(5, 2))
Parallel(
  +,
  Dense(10 => 2),                       # 22 parameters
  Dense(5 => 2),                        # 12 parameters
)                   # Total: 4 arrays, 34 parameters, 392 bytes.

julia> size(model(rand(10), rand(5)))
(2,)
```
"""
struct Parallel{F, T}
  connection::F
  layers::T
end

Parallel(connection, layers...) = Parallel(connection, layers)

@functor Parallel

(m::Parallel)(x::AbstractArray) = mapreduce(f -> f(x), m.connection, m.layers)
(m::Parallel)(xs::Vararg{<:AbstractArray}) = mapreduce((f, x) -> f(x), m.connection, m.layers, xs)
(m::Parallel)(xs::Tuple) = m(xs...)

Base.getindex(m::Parallel, i::Integer) = m.layers[i]
Base.getindex(m::Parallel, i::AbstractVector) = Parallel(m.connection, m.layers[i]...)

trainable(m::Parallel) = (m.connection, m.layers...)

function Base.show(io::IO, m::Parallel)
  print(io, "Parallel(", m.connection, ", ")
  join(io, m.layers, ", ")
  print(io, ")")
end
