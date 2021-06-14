import Adapt
import .CUDA

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

OneHotVector(idx, L) = OneHotArray(idx, L)
OneHotMatrix(indices, L) = OneHotArray(indices, L)

function _show_elements(x::OneHotArray)
  xbool = convert(Array{Bool}, cpu(x))
  xrepr = join(split(repr(MIME("text/plain"), xbool; context= :limit => true), "\n")[2:end], "\n")

  return xrepr
end

function Base.show(io::IO, ::MIME"text/plain", x::OneHotArray{<:Any, L, <:Any, N, I}) where {L, N, I}
  join(io, string.(size(x)), "×")
  print(io, " Flux.OneHotArray{")
  join(io, string.([L, N, I]), ",")
  println(io, "}:")
  print(io, _show_elements(x))
end
function Base.show(io::IO, ::MIME"text/plain", x::OneHotVector{T, L}) where {T, L}
  print(io, string.(length(x)))
  print(io, "-element Flux.OneHotVector{")
  join(io, string.([L, T]), ",")
  println(io, "}:")
  print(io, _show_elements(x))
end

# use this type so reshaped arrays hit fast paths
# e.g. argmax
const OneHotLike{T, L, N, var"N+1", I} =
  Union{OneHotArray{T, L, N, var"N+1", I},
        Base.ReshapedArray{Bool, var"N+1", <:OneHotArray{T, L, <:Any, <:Any, I}}}

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

Base.argmax(x::OneHotLike; dims = Colon()) =
  (_isonehot(x) && dims == 1) ?
    reshape(CartesianIndex.(_indices(x), CartesianIndices(_indices(x))), 1, size(_indices(x))...) :
    invoke(argmax, Tuple{AbstractArray}, x; dims = dims)

"""
    onehot(l, labels[, unk])

Return a `OneHotVector` where only first occourence of `l` in `labels` is `1` and
all other elements are `0`.

If `l` is not found in labels and  `unk` is present, the function returns
`onehot(unk, labels)`; otherwise the function raises an error.

# Examples
```jldoctest
julia> Flux.onehot(:b, [:a, :b, :c])
3-element Flux.OneHotVector{3,UInt32}:
 0
 1
 0

julia> Flux.onehot(:c, [:a, :b, :c])
3-element Flux.OneHotVector{3,UInt32}:
 0
 0
 1
```
"""
function onehot(l, labels)
  i = something(findfirst(isequal(l), labels), 0)
  i > 0 || error("Value $l is not in labels")
  OneHotVector{UInt32, length(labels)}(i)
end

function onehot(l, labels, unk)
  i = something(findfirst(isequal(l), labels), 0)
  i > 0 || return onehot(unk, labels)
  OneHotVector{UInt32, length(labels)}(i)
end

"""
    onehotbatch(ls, labels[, unk...])

Return a `OneHotMatrix` where `k`th column of the matrix is `onehot(ls[k], labels)`.

If one of the input labels `ls` is not found in `labels` and `unk` is given,
return [`onehot(unk, labels)`](@ref) ; otherwise the function will raise an error.

# Examples
```jldoctest
julia> Flux.onehotbatch([:b, :a, :b], [:a, :b, :c])
3×3 Flux.OneHotArray{3,2,Vector{UInt32}}:
 0  1  0
 1  0  1
 0  0  0
```
"""
onehotbatch(ls, labels, unk...) = batch([onehot(l, labels, unk...) for l in ls])

"""
    onecold(y, labels = 1:size(y,1))

Inverse operations of [`onehot`](@ref). Similar to `argmax(y, dims=1)`.

# Examples
```jldoctest
julia> Flux.onecold([true, false, false], [:a, :b, :c])
:a

julia> Flux.onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c

julia> y = Flux.onehotbatch([1,3,2,1,3], 1:3)
3×5 OneHotArray{UInt32, 3, 1, 2, Vector{UInt32}}:
 1  0  0  1  0
 0  0  1  0  0
 0  1  0  0  1

julia> Flux.onecold(y)
5-element Vector{Int64}:
 1
 3
 2
 1
 3
```
"""
onecold(y::AbstractVector) = argmax(y) 

function onecold(y::AbstractArray) 
  x = dropdims(argmax(y; dims = 1); dims = 1)
  # convert from CartesianIndex
  return map(c -> c[1], x)
end
onecold(y::OneHotArray) = _indices(y)
onecold(y::OneHotLike) = _isonehot(y) ? _indices(y) : 
                            onecold(convert(_onehot_bool_type(y), y))

onecold(y::AbstractVector, labels) = labels[onecold(y)]
onecold(y::AbstractArray, labels) = _onecold(onecold(y), labels)

function _onecold(indices::CuArray, labels)
  all(isbits, labels) || 
    throw(ArgumentError("Only `all(isbits, labels)` allowed on gpu")) 
  l = cu(collect(labels))
  return reshape(l[vec(indices)], size(indices))
end

# _onecold(indices::AbstractArray, labels) = map(c -> labels[c], indices)
_onecold(indices::AbstractArray, labels) = reshape(labels[vec(indices)], size(indices))

@nograd OneHotArray, onecold, onehot, onehotbatch

function Base.:(*)(A::AbstractMatrix, B::OneHotLike{<:Any, L}) where L
  _isonehot(B) || return invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
  size(A, 2) == L || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $L"))
  return A[:, onecold(B)]
end
