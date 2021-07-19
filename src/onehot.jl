import Adapt
import .CUDA

"""
    OneHotArray{T,L,N,M,I} <: AbstractArray{Bool,M}

These are constructed by [`onehot`](@ref) and [`onehotbatch`](@ref).
Parameter `I` is the type of the underlying storage, and `T` its eltype.
"""
struct OneHotArray{T<:Integer, L, N, var"N+1", I<:Union{T, AbstractArray{T, N}}} <: AbstractArray{Bool, var"N+1"}
  indices::I
end
OneHotArray{T, L, N, I}(indices) where {T, L, N, I} = OneHotArray{T, L, N, N+1, I}(indices)
OneHotArray(indices::T, L::Integer) where {T<:Integer} = OneHotArray{T, L, 0, 1, T}(indices)
OneHotArray(indices::I, L::Integer) where {T, N, I<:AbstractArray{T, N}} = OneHotArray{T, L, N, N+1, I}(indices)

_indices(x::OneHotArray) = x.indices
_indices(x::Base.ReshapedArray{<: Any, <: Any, <: OneHotArray}) =
  reshape(parent(x).indices, x.dims[2:end])

const OneHotVector{T, L} = OneHotArray{T, L, 0, 1, T}
const OneHotMatrix{T, L, I} = OneHotArray{T, L, 1, 2, I}

@doc @doc(OneHotArray)
OneHotVector(idx, L) = OneHotArray(idx, L)
@doc @doc(OneHotArray)
OneHotMatrix(indices, L) = OneHotArray(indices, L)

# use this type so reshaped arrays hit fast paths
# e.g. argmax
const OneHotLike{T, L, N, var"N+1", I} =
  Union{OneHotArray{T, L, N, var"N+1", I},
        Base.ReshapedArray{Bool, var"N+1", <:OneHotArray{T, L, <:Any, <:Any, I}}}

const OneHotLikeVector{T, L} = OneHotLike{T, L, 0, 1, T}
const OneHotLikeMatrix{T, L, I} = OneHotLike{T, L, 1, 2, I}  

_isonehot(x::OneHotArray) = true
_isonehot(x::Base.ReshapedArray{<:Any, <:Any, <:OneHotArray{<:Any, L}}) where L = (size(x, 1) == L)

Base.size(x::OneHotArray{<:Any, L}) where L = (Int(L), size(x.indices)...)

_onehotindex(x, i) = (x == i)

Base.getindex(x::OneHotVector, i::Integer) = _onehotindex(x.indices, i)
Base.getindex(x::OneHotVector{T, L}, ::Colon) where {T, L} = x

Base.getindex(x::OneHotArray, i::Integer, I...) = _onehotindex.(x.indices[I...], i)
Base.getindex(x::OneHotArray{<:Any, L}, ::Colon, I...) where L = OneHotArray(x.indices[I...], L)
Base.getindex(x::OneHotArray{<:Any, <:Any, <:Any, N}, ::Vararg{Colon, N}) where N = x
Base.getindex(x::OneHotArray, I::CartesianIndex{N}) where N = x[I[1], Tuple(I)[2:N]...]

function Base.showarg(io::IO, x::OneHotArray, toplevel)
    print(io, ndims(x) == 1 ? "OneHotVector(" : ndims(x) == 2 ? "OneHotMatrix(" : "OneHotArray(")
    Base.showarg(io, x.indices, false)
    print(io, ')')
    toplevel && print(io, " with eltype Bool")
    return nothing
end

# this is from /LinearAlgebra/src/diagonal.jl, official way to print the dots:
function Base.replace_in_print_matrix(x::OneHotLike, i::Integer, j::Integer, s::AbstractString)
    x[i,j] ? s : _isonehot(x) ? Base.replace_with_centered_mark(s) : s
end

# copy CuArray versions back before trying to print them:
Base.print_array(io::IO, X::OneHotLike{T, L, N, var"N+1", <:CuArray}) where {T, L, N, var"N+1"} = 
  Base.print_array(io, cpu(X))
Base.print_array(io::IO, X::LinearAlgebra.AdjOrTrans{Bool, <:OneHotLike{T, L, N, var"N+1", <:CuArray}}) where {T, L, N, var"N+1"} = 
  Base.print_array(io, cpu(X))

_onehot_bool_type(x::OneHotLike{<:Any, <:Any, <:Any, N, <:Union{Integer, AbstractArray}}) where N = Array{Bool, N}
_onehot_bool_type(x::OneHotLike{<:Any, <:Any, <:Any, N, <:CuArray}) where N = CuArray{Bool, N}

function Base.cat(x::OneHotLike{<:Any, L}, xs::OneHotLike{<:Any, L}...; dims::Int) where L
  if isone(dims) || any(x -> !_isonehot(x), (x, xs...))
    return cat(map(x -> convert(_onehot_bool_type(x), x), (x, xs...))...; dims = dims)
  else
    return OneHotArray(cat(_indices(x), _indices.(xs)...; dims = dims - 1), L)
  end
end

Base.hcat(x::OneHotLike, xs::OneHotLike...) = cat(x, xs...; dims = 2)
Base.vcat(x::OneHotLike, xs::OneHotLike...) = cat(x, xs...; dims = 1)

# optimized concatenation for matrices and vectors of same parameters
Base.hcat(x::T, xs::T...) where {L, T <: OneHotLike{<:Any, L, <:Any, 2}} =
  OneHotMatrix(reduce(vcat, _indices.(xs); init = _indices(x)), L)
Base.hcat(x::T, xs::T...) where {L, T <: OneHotLike{<:Any, L, <:Any, 1}} =
  OneHotMatrix(reduce(vcat, _indices.(xs); init = _indices(x)), L)

batch(xs::AbstractArray{<:OneHotVector{<:Any, L}}) where L = OneHotArray(_indices.(xs), L)

Adapt.adapt_structure(T, x::OneHotArray{<:Any, L}) where L = OneHotArray(adapt(T, _indices(x)), L)

Base.BroadcastStyle(::Type{<:OneHotArray{<: Any, <: Any, <: Any, N, <: CuArray}}) where N = CUDA.CuArrayStyle{N}()

Base.map(f, x::OneHotLike) = Base.broadcast(f, x)

Base.argmax(x::OneHotLike; dims = Colon()) =
  (_isonehot(x) && dims == 1) ?
    reshape(CartesianIndex.(_indices(x), CartesianIndices(_indices(x))), 1, size(_indices(x))...) :
    invoke(argmax, Tuple{AbstractArray}, x; dims = dims)

"""
    onehot(x, labels, [default])

Return a `OneHotVector` which is roughly a sparse representation of `x .== labels`.

Instead of storing say `Vector{Bool}`, it stores the index of the first occurrence 
of `x` in `labels`. If `x` is not found in labels, then it either returns `onehot(default, labels)`,
or gives an error if no default is given.

See also [`onehotbatch`](@ref) to apply this to many `x`s, 
and [`onecold`](@ref) to reverse either of these, as well as to generalise `argmax`.

# Examples
```jldoctest
julia> β = Flux.onehot(:b, [:a, :b, :c])
3-element OneHotVector(::UInt32) with eltype Bool:
 ⋅
 1
 ⋅

julia> αβγ = (Flux.onehot(0, 0:2), β, Flux.onehot(:z, [:a, :b, :c], :c))  # uses default
(Bool[1, 0, 0], Bool[0, 1, 0], Bool[0, 0, 1])

julia> hcat(αβγ...)  # preserves sparsity
3×3 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
```
"""
function onehot(x, labels)
  i = something(findfirst(isequal(x), labels), 0)
  i > 0 || error("Value $x is not in labels")
  OneHotVector{UInt32, length(labels)}(i)
end

function onehot(x, labels, default)
  i = something(findfirst(isequal(x), labels), 0)
  i > 0 || return onehot(default, labels)
  OneHotVector{UInt32, length(labels)}(i)
end

"""
    onehotbatch(xs, labels, [default])

Returns a `OneHotMatrix` where `k`th column of the matrix is [`onehot(xs[k], labels)`](@ref onehot).
This is a sparse matrix, which stores just a `Vector{UInt32}` containing the indices of the
nonzero elements.

If one of the inputs in `xs` is not found in `labels`, that column is `onehot(default, labels)`
if `default` is given, else an error.

If `xs` has more dimensions, `M = ndims(xs) > 1`, then the result is an 
`AbstractArray{Bool, M+1}` which is one-hot along the first dimension, 
i.e. `result[:, k...] == onehot(xs[k...], labels)`.

# Examples
```jldoctest
julia> oh = Flux.onehotbatch(collect("abracadabra"), 'a':'e', 'e')
5×11 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 1  ⋅  ⋅  1  ⋅  1  ⋅  1  ⋅  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅

julia> reshape(1:15, 3, 5) * oh  # this matrix multiplication is done efficiently
3×11 Matrix{Int64}:
 1  4  13  1  7  1  10  1  4  13  1
 2  5  14  2  8  2  11  2  5  14  2
 3  6  15  3  9  3  12  3  6  15  3
```
"""
onehotbatch(ls, labels, default...) = batch([onehot(l, labels, default...) for l in ls])

"""
    onecold(y::AbstractArray, labels = 1:size(y,1))

Roughly the inverse operation of [`onehot`](@ref) or [`onehotbatch`](@ref): 
This finds the index of the largest element of `y`, or each column of `y`, 
and looks them up in `labels`.

If `labels` are not specified, the default is integers `1:size(y,1)` --
the same operation as `argmax(y, dims=1)` but sometimes a different return type.

# Examples
```jldoctest
julia> Flux.onecold([false, true, false])
2

julia> Flux.onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c

julia> Flux.onecold([ 1  0  0  1  0  1  0  1  0  0  1
                      0  1  0  0  0  0  0  0  1  0  0
                      0  0  0  0  1  0  0  0  0  0  0
                      0  0  0  0  0  0  1  0  0  0  0
                      0  0  1  0  0  0  0  0  0  1  0 ], 'a':'e') |> String
"abeacadabea"
```
"""
onecold(y::AbstractVector, labels = 1:length(y)) = labels[argmax(y)]
function onecold(y::AbstractArray, labels = 1:size(y, 1))
  indices = _fast_argmax(y)
  xs = isbits(labels) ? indices : collect(indices) # non-bit type cannot be handled by CUDA

  return map(xi -> labels[xi[1]], xs)
end

_fast_argmax(x::AbstractArray) = dropdims(argmax(x; dims = 1); dims = 1)
function _fast_argmax(x::OneHotLike)
  if _isonehot(x)
    return _indices(x)
  else
    return _fast_argmax(convert(_onehot_bool_type(x), x))
  end
end

@nograd OneHotArray, onecold, onehot, onehotbatch

function Base.:(*)(A::AbstractMatrix, B::OneHotLike{<:Any, L}) where L
  _isonehot(B) || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == L || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))
  return A[:, onecold(B)]
end
for wrapper in [:Adjoint, :Transpose]
  @eval begin
    function Base.:*(A::$wrapper{<:Any, <:AbstractMatrix{T}}, b::OneHotVector{<:Any, L}) where {L, T}
      size(A, 2) == L ||
        throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))

      return A[:, onecold(b)]
    end

    function Base.:*(A::$wrapper{<:Number, <:AbstractVector{T}}, b::OneHotVector{<:Any, L}) where {L, T}
      size(A, 2) == L ||
        throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))

      return A[onecold(b)]
    end
  end
end
