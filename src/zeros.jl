import Base: +, -, *, reshape, size
import Base.Broadcast: broadcasted, Broadcasted, BroadcastStyle

"""
    Zeros()
    Zeros(size...)
    Zeros(Type, size...)

Acts as a stand-in for an array of zeros that can be
used during training which is ignored by the optimisers.

Useful to turn bias off for a forward pass of a layer.

## Examples

```julia
julia> Flux.Zeros(3,3)
3×3 Flux.Zeros{Bool,2}:
 false  false  false
 false  false  false
 false  false  false

julia> Flux.Zeros(Float32, 3,3)
3×3 Flux.Zeros{Float32,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> rand(3,3) .+ Flux.Zeros()
3×3 Array{Float64,2}:
 0.198739  0.490459  0.785386
 0.779074  0.39986   0.66383
 0.854981  0.447292  0.314497

julia> bias_less_conv = Conv((2,2), 1=>3, bias = Flux.Zeros())
Conv((2, 2), 1=>3)
```
"""
struct Zeros{T,N} <: AbstractArray{T,N}
  size::Tuple
end

Zeros(::Type{T}, sz...) where T = Zeros{T,length(sz)}(sz)
Zeros(sz::Integer...) = Zeros(Bool, sz...)

Base.size(xs::Zeros) = xs.size
Base.axes(xs::Zeros) = Base.OneTo.(size(xs))

Base.IndexStyle(::Type{<:Zeros}) = IndexLinear()

Base.getindex(xs::Zeros{T,N}, I::Int) where {T,N} = zero(T)
Base.getindex(xs::Zeros{T,N}, inds::Union{Base.OneTo, Base.UnitRange}) where {T,N} =
              Zeros(T, length(inds))

Base.collect(xs::Zeros{T,N}) where {T,N} = fill(zero(T), size(xs))

@adjoint reshape(xs::Zeros{T}, dims...) where T =
                reshape(xs, dims...), _ -> nothing

# Define basic ops
for f in (:+, :-)
  @eval @inline function $f(a::Union{AbstractArray{<:Number}, Zeros}, b::Zeros)
    @assert size(a) == size(b) throw(DimensionMismatch("dimensions must match"))
    a
  end
end

+(a::Zeros, b::AbstractArray) = b + a
-(a::Zeros, b::AbstractArray) = -b + a

Base.copy(xs::Zeros{T,N}) where {T,N} = xs

# Define broadcasting behaviour
for op in (:+, :-)
  @eval function broadcasted(::typeof($op), a::AbstractArray, b::Zeros)
    bs = Broadcast.broadcast_shape(size(a), size(b))
    size(a) == bs && return a
    sz = similar(a, bs)
    sz .= a
  end
end

broadcasted(::typeof(+), a::Zeros, b::AbstractArray) = broadcasted(+, b, a)
broadcasted(::typeof(-), a::Zeros, b::AbstractArray) = broadcasted(+, -b, a)

function broadcasted(::typeof(*), a::AbstractArray, b::Zeros)
  Zeros(Broadcast.broadcast_shape(size(a), size(b))...)
end

broadcasted(::typeof(*), a::Zeros, b::AbstractArray) = broadcasted(*, b, a)

for op in (:+, :-, :*)
  @eval broadcasted(::typeof($op), a::Zeros, b::Zeros) = Zeros(Broadcast.broadcast_shape(size(a), size(b))...)
end

# Some opportunities to avoid scalar indexing, intermediaries
# Since it replicates a little of what we expect Base to do,
# it should be possible to remove in the future, but for now,
# these help with performance.
broadcasted(::typeof(+), a::AbstractArray, b::Zeros{T,0}) where T = a
broadcasted(::typeof(+), a::Zeros{T,0}, b::AbstractArray) where T = b
broadcasted(::typeof(-), a::AbstractArray, b::Zeros{T,0}) where T = a
broadcasted(::typeof(-), a::Zeros{T,0}, b::AbstractArray) where T = -b
broadcasted(::typeof(*), a::AbstractArray, b::Zeros{T,0}) where T = zero(a)
broadcasted(::typeof(*), a::Zeros{T,0}, b::AbstractArray) where T = zero(b)
broadcasted(::typeof(/), a::Zeros{T,0}, b::AbstractArray) where T = zero(b)
