import Base: +, -, *,/, reshape, broadcasted

"""
    Zeros()

Acts as a stand-in for an array of zeros that can be
used during training which is ignored by the optimisers.

Useful to turn bias off for a forward pass of a layer.

## Examples

```julia-repl
julia> bias_less_conv = Conv((2,2), 1=>3; bias = false)
Conv((2, 2), 1=>3)

julia> bias_less_conv.bias
Flux.Zeros()
```
"""
mutable struct Zeros{T,N} <: AbstractArray{T,N}
  dims::NTuple{N,Int}
end

Zeros(::Type{T}, dims...) where T = Zeros{T,length(dims)}(dims)
Zeros(dims...) = Zeros(Bool, dims...)

Base.reshape(x::Zeros{T}, dims::Union{Colon,Int}...) where T = Zeros(T, Base._reshape_uncolon(x, dims)...)
Base.getindex(z::Zeros, args...) = error("Calling getindex on Zeros object, materialize to normal array or check for correctness")
# Base.getindex(z::Zeros{T}, args...) where T = zero(T)
Base.collect(x::Zeros{T}) where T = zeros(T, x.dims...)

Base.size(xs::Zeros) = xs.dims
Base.copyto!(a::Zeros, b::Zeros) = b

# Base.print_array(io::IO, z::Zeros{T}) where T = print(io, "Zeros object with size $(z.dims)")

Flux.CUDA.Adapt.adapt(to, x::Zeros) = x

@adjoint reshape(xs::Zeros{T}, dims...) where T =
                reshape(xs, dims...), _ -> nothing

# @adjoint Zeros(args...) = Zeros(args...), _ -> nothing

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

for op in (:+, :-)
  @eval function broadcasted(::typeof($op), a::AbstractArray, b::Zeros)
    bs = Broadcast.broadcast_shape(size(a), size(b))
    size(a) == bs && return a
    sz = similar(a, bs)
    sz .= a
  end
end

function broadcasted(::typeof(*), a::AbstractArray{T}, b::Zeros) where {T}
  bs = Broadcast.broadcast_shape(size(a), size(b))
  fill!(similar(a, bs), zero(T))
end
broadcasted(::typeof(*), a::Zeros, b::AbstractArray) = b .* a

broadcasted(::typeof(+), a::Zeros, b::AbstractArray) = broadcasted(+, b, a)
broadcasted(::typeof(-), a::Zeros, b::AbstractArray) = broadcasted(+, -b, a)

for op in (:+, :-, :*)
  @eval @adjoint function Base.broadcasted(::typeof($op), a::AbstractArray{T,N}, b::Zeros{S,M}) where {T <: Number, S <: Number, N,M}
    Base.broadcasted($op, a, b), Δ -> begin
      dims = M > N ? tuple(setdiff(1:M, 1:N)...) : tuple(setdiff(1:N, 1:M)...)
      da = dims == Tuple(1:N) ? Δ : dropdims(sum(Δ, dims = dims), dims = dims)
      (nothing, da, nothing)
    end
  end

  @eval @adjoint function Base.broadcasted(::typeof($op), a::Zeros{<:Any, N}, b::AbstractArray{<: Number, M}) where {M, N}
    a .* b, Δ -> begin
      dims = M > N ? tuple(setdiff(1:M, 1:N)...) : tuple(setdiff(1:N, 1:M)...)
      da = dims == Tuple(1:N) ? Δ : dropdims(sum(Δ, dims = dims), dims = dims)
      (nothing, nothing, da)
    end
  end
end

Base.sum(z::Zeros{T}) where T = zero(T)

for op in (:+, :-, :*)
  @eval @adjoint function $op(a::AbstractArray{T,N}, b::Zeros{S,M}) where {T <: Number, S <: Number, N,M}
    $op(a, b), Δ -> begin
      (Δ, nothing)
    end
  end

  @eval @adjoint function $op(a::Zeros, b::AbstractArray)
    $op(a, b), Δ -> (nothing, Δ)
  end
end

# Some opportunities to avoid scalar indexing, intermediaries
# Since it replicates a little of what we expect Base to do,
# it should be possible to remove in the future, but for now,
# these help with performance.
broadcasted(::typeof(+), a::AbstractArray, b::Zeros{T,0}) where T = a
broadcasted(::typeof(+), a::Zeros{T,0}, b::AbstractArray) where T = b
broadcasted(::typeof(-), a::AbstractArray, b::Zeros{T,0}) where T = a
broadcasted(::typeof(-), a::Zeros{T,0}, b::AbstractArray) where T = -b
# broadcasted(::typeof(*), a::Zeros{T,0}, b::AbstractArray) where T = zero(b)
# broadcasted(::typeof(*), a::AbstractArray, b::Zeros{T,0}) where T = zero(b)

broadcasted(::typeof(conj), z::Zeros) = z

@adjoint broadcasted(::typeof(*), a::Zeros{S,0}, b::AbstractArray{T}) where {S, T <: Number} = a .* b, Δ -> (nothing, nothing, Δ)
