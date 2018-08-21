"""
    Chain(layers...)

Chain multiple layers / functions together, so that they are called in sequence
on a given input.

```julia
m = Chain(x -> x^2, x -> x+1)
m(5) == 26

m = Chain(Dense(10, 5), Dense(5, 2))
x = rand(10)
m(x) == m[2](m[1](x))
```

`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.
"""
struct Chain
  layers::Vector{Any}
  Chain(xs...) = new([xs...])
end

@forward Chain.layers Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!, Base.length
@forward Chain.layers Base.iterate

children(c::Chain) = c.layers
mapchildren(f, c::Chain) = Chain(f.(c.layers)...)
adapt(T, c::Chain) = Chain(map(x -> adapt(T, x), c.layers)...)

(c::Chain)(x) = foldl((x, m) -> m(x), c.layers; init = x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end

"""
    activations(c::Union{Chain,Any}, x)

The input `c` must be a Chain or any layer that supports operation `c(x)`.

Creates an Array that stores activation of each layers

# Examples
```julia
julia> c = Chain(Dense(10,5,σ),Dense(5,2),softmax)
Chain(Dense(10, 5, NNlib.σ), Dense(5, 2), NNlib.softmax)
julia> activations(c,ones(10))
3-element Array{Any,1}:
 Flux.Tracker.TrackedReal{Float64}[0.520429 (tracked), 0.706467 (tracked), 0.276672 (tracked), 0.563502 (tracked), 0.371877 (tracked)]
 Flux.Tracker.TrackedReal{Float64}[-0.119249 (tracked), 0.461743 (tracked)]
 Flux.Tracker.TrackedReal{Float64}[0.358704 (tracked), 0.641296 (tracked)]
```
# Examples
```julia
julia> m = Dense(5,2)
Dense(5, 2)
julia> activations(m,randn(5))
1-element Array{Any,1}:
 Flux.Tracker.TrackedReal{Float64}[-0.942021 (tracked), 1.07021 (tracked)]
```
"""
activations(m::Any, x, rst=[]) = push!(rst,m(x))
activations(c::Chain, x, rst=[]) = begin
    rst = activations(c[1], x, rst)
    if length(c) >= 2
        rst = activations(c[2:end], rst[end], rst)
    end
    return rst
end

"""
    Dense(in::Integer, out::Integer, σ = identity)

Creates a traditional `Dense` layer with parameters `W` and `b`.

    y = σ.(W * x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

```julia
julia> d = Dense(5, 2)
Dense(5, 2)

julia> d(rand(5))
Tracked 2-element Array{Float64,1}:
  0.00257447
  -0.00449443
```
"""
struct Dense{F, S <: AbstractArray,T <: AbstractArray}
  W::S
  b::T
  σ::F
end

Dense(W, b) = Dense(W, b, identity)

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return Dense(param(initW(out, in)), param(initb(out)), σ)
end

@treelike Dense

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end
(a::Dense)(x::Number) = a([x]) # prevent broadcasting of scalar

function Base.show(io::IO, l::Dense)
  print(io, "Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    Diagonal(in::Integer)

Creates an element-wise linear transformation layer with learnable
vectors `α` and `β`:

    y = α .* x .+ β

The input `x` must be a array where `size(x, 1) == in`.
"""
struct Diagonal{T}
  α::T
  β::T
end

Diagonal(in::Integer; initα = ones, initβ = zeros) =
  Diagonal(param(initα(in)), param(initβ(in)))

@treelike Diagonal

function (a::Diagonal)(x)
  α, β = a.α, a.β
  α.*x .+ β
end

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", length(l.α), ")")
end
