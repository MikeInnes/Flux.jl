using NNlib: conv, ∇conv_data, depthwiseconv, output_size

# pad dims of x with dims of y until ndims(x) == ndims(y)
_paddims(x::Tuple, y::Tuple) = (x..., y[(end - (length(y) - length(x) - 1)):end]...)

_convtransoutdims(isize, ksize, ssize, dsize, pad) = (isize .- 1).*ssize .+ 1 .+ (ksize .- 1).*dsize .- (pad[1:2:end] .+ pad[2:2:end])

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

"""
    SamePad

Padding for convolutional layers will be calculated so that outputshape == inputshape when stride = 1.

For stride > 1 the output shape depends on the type of convolution layer.
"""
struct SamePad end

calc_padding(lt, pad, k::NTuple{N,T}, dilation, stride) where {T,N}= expand(Val(2*N), pad)
function calc_padding(lt, ::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
  #Ref: "A guide to convolution arithmetic for deep learning" https://arxiv.org/abs/1603.07285

  # Effective kernel size, including dilation
  k_eff = @. k + (k - 1) * (dilation - 1)
  # How much total padding needs to be applied?
  pad_amt = @. k_eff - 1
  # In case amount of padding is odd we need to apply different amounts to each side.
  return Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, pad_amt))
end

"""
    Conv(filter, in => out, σ = identity; init = glorot_uniform,
         stride = 1, pad = 0, dilation = 1)

    filter = (2,2)
    in = 1
    out = 16
    Conv((2, 2), 1=>16, relu)

Standard convolutional layer. `filter` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order (width, height, # channels, batch size).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.
Use `pad=SamePad()` to apply padding so that outputsize == inputsize / stride.

# Examples

Apply a `Conv` layer to a 1-channel input using a 2×2 window filter size, giving us a
16-channel output. Output is activated with ReLU.
```julia
filter = (2,2)
in = 1
out = 16
Conv(filter, in => out, relu)
```
"""
struct Conv{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

"""
    Conv(weight::AbstractArray, bias::AbstractArray)
    Conv(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the convolutional layer with user defined weight and bias arrays.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.

There is also a keyword-only constuctor available for all convoultional
layers.

```julia
weight = rand(Float32, 3, 3, 5)
bias = zeros(Float32, 5)
Conv(weight = weight,
    bias = bias,
    σ = sigmoid)
```
"""
function Conv(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
              stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(Conv, pad, size(w)[1:N-2], dilation, stride)
  return Conv(σ, w, b, stride, pad, dilation)
end

function Conv(;weight::AbstractArray{T,N}, bias::Union{Zeros, AbstractVector{T}},
              activation = identity, stride = 1, pad = 0, dilation = 1) where {T,N}
  Conv(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

"""
    convfilter(filter::Tuple, in=>out)

Constructs a standard convolutional weight matrix with given `filter` and
channels from `in` to `out`.

Accepts the keyword `init` (default: `glorot_uniform`) to control the sampling
distribution.

See also: [`depthwiseconvfilter`](@ref)
"""
convfilter(filter::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
          init = glorot_uniform) where N = init(filter..., ch...)

function Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
            init = glorot_uniform,  stride = 1, pad = 0, dilation = 1,
            weight = convfilter(k, ch, init = init), bias = zeros(ch[2])) where N

  Conv(weight, bias, σ,
      stride = stride, pad = pad, dilation = dilation)
end

@functor Conv

function (c::Conv)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(conv(x, c.weight, cdims) .+ b)
end

function Base.show(io::IO, l::Conv)
  print(io, "Conv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::Conv{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Conv{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

"""
    outdims(l::Conv, isize::Tuple)

Calculate the output dimensions given the input dimensions `isize`.
Batch size and channel size are ignored as per [NNlib.jl](https://github.com/FluxML/NNlib.jl).

```julia
m = Conv((3, 3), 3 => 16)
outdims(m, (10, 10)) == (8, 8)
outdims(m, (10, 10, 1, 3)) == (8, 8)
```
"""
outdims(l::Conv, isize) =
  output_size(DenseConvDims(_paddims(isize, size(l.weight)), size(l.weight); stride = l.stride, padding = l.pad, dilation = l.dilation))

"""
    ConvTranspose(filter, in=>out)
    ConvTranspose(filter, in=>out, activation)
    ConvTranspose(filter, in => out, σ = identity; init = glorot_uniform,
                  stride = 1, pad = 0, dilation = 1)

Standard convolutional transpose layer. `filter` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order (width, height, # channels, batch size).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.
Use `pad=SamePad()` to apply padding so that outputsize == stride * inputsize.
"""
struct ConvTranspose{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

"""
    ConvTranspose(weight::AbstractArray, bias::AbstractArray)
    ConvTranspose(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the convolutional transpose layer with user defined weight and bias arrays.
forward pass.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.

For keyword-only constuctor, see also [`Conv`](@ref)
"""
function ConvTranspose(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
                      stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(ConvTranspose, pad, size(w)[1:N-2], dilation, stride)
  return ConvTranspose(σ, w, b, stride, pad, dilation)
end

function ConvTranspose(;weight::AbstractArray{T,N}, bias::Union{Zeros, AbstractVector{T}},
                        activation = identity, stride = 1, pad = 0, dilation = 1) where {T,N}
  ConvTranspose(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

function ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                      init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                      weight = convfilter(k, reverse(ch), init = init), bias = zeros(ch[2])) where N

  ConvTranspose(weight, bias, σ,
              stride = stride, pad = pad, dilation = dilation)
end

@functor ConvTranspose

function conv_transpose_dims(c::ConvTranspose, x::AbstractArray)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    combined_pad = (c.pad[1:2:end] .+ c.pad[2:2:end])
    I = (size(x)[1:end-2] .- 1).*c.stride .+ 1 .+ (size(c.weight)[1:end-2] .- 1).*c.dilation .- combined_pad
    C_in = size(c.weight)[end-1]
    batch_size = size(x)[end]
    # Create DenseConvDims() that looks like the corresponding conv()
    return DenseConvDims((I..., C_in, batch_size), size(c.weight);
                        stride=c.stride,
                        padding=c.pad,
                        dilation=c.dilation,
    )
end

# TODO: Find proper fix for https://github.com/FluxML/Flux.jl/issues/900
@nograd conv_transpose_dims

function (c::ConvTranspose)(x::AbstractArray)
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = conv_transpose_dims(c, x)
  σ.(∇conv_data(x, c.weight, cdims) .+ b)
end

function Base.show(io::IO, l::ConvTranspose)
  print(io, "ConvTranspose(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)), "=>", size(l.weight, ndims(l.weight)-1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::ConvTranspose{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::ConvTranspose{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

outdims(l::ConvTranspose{N}, isize) where N = _convtransoutdims(isize[1:2], size(l.weight)[1:N], l.stride, l.dilation, l.pad)

function calc_padding(::Type{ConvTranspose}, pad::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
  calc_padding(Conv, pad, k .- stride .+ 1, dilation, stride)
end

"""
    DepthwiseConv(filter::Tuple, in=>out)
    DepthwiseConv(filter::Tuple, in=>out, activation)
    DepthwiseConv(filter, in => out, σ = identity; init = glorot_uniform,
                  stride = 1, pad = 0, dilation = 1)

Depthwise convolutional layer. `filter` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.
Note that `out` must be an integer multiple of `in`.

Data should be stored in WHCN order (width, height, # channels, batch size).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.
Use `pad=SamePad()` to apply padding so that outputsize == inputsize / stride.
"""
struct DepthwiseConv{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

"""
    DepthwiseConv(weight::AbstractArray, bias::AbstractArray)
    DepthwiseConv(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the `DepthwiseConv` layer with user defined weight and bias arrays.
forward pass.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.

For keyword-only constuctor, see also [`Conv`](@ref)
"""
function DepthwiseConv(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
                      stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(DepthwiseConv, pad, size(w)[1:N-2], dilation, stride)
  return DepthwiseConv(σ, w, b, stride, pad, dilation)
end

function DepthwiseConv(;weight::AbstractArray{T,N}, bias::Union{Zeros, AbstractVector{T}},
                      activation = identity, stride = 1, pad = 0, dilation = 1) where {T,N}
  DepthwiseConv(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

"""
    depthwiseconvfilter(filter::Tuple, in=>out)

Constructs a depthwise convolutional weight array defined by `filter` and channels
from `in` to `out`.

Accepts the keyword `init` (default: `glorot_uniform`) to control the sampling
distribution.

See also: [`convfilter`](@ref)
"""
depthwiseconvfilter(filter::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
                    init = glorot_uniform) where N = init(filter..., div(ch[2], ch[1]), ch[1])

function DepthwiseConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                      init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                      weight = depthwiseconvfilter(k, ch, init = init), bias = zeros(ch[2])) where N
  @assert ch[2] % ch[1] == 0 "Output channels must be integer multiple of input channels"

  return DepthwiseConv(
    weight,
    bias,
    σ;
    stride = stride,
    pad = pad,
    dilation = dilation
  )
end

@functor DepthwiseConv

function (c::DepthwiseConv)(x)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = DepthwiseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(depthwiseconv(x, c.weight, cdims) .+ b)
end

function Base.show(io::IO, l::DepthwiseConv)
  print(io, "DepthwiseConv(", size(l.weight)[1:end-2])
  print(io, ", ", size(l.weight)[end], "=>", prod(size(l.weight)[end-1:end]))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::DepthwiseConv{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::DepthwiseConv{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

outdims(l::DepthwiseConv, isize) =
  output_size(DepthwiseConvDims(_paddims(isize, (1, 1, size(l.weight)[end], 1)), size(l.weight); stride = l.stride, padding = l.pad, dilation = l.dilation))

"""
    CrossCor(filter, in=>out)
    CrossCor(filter, in=>out, activation)
    CrossCor(filter, in => out, σ = identity; init = glorot_uniform,
             stride = 1, pad = 0, dilation = 1)

Standard cross convolutional layer. `filter` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order (width, height, # channels, batch size).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.
Use `pad=SamePad()` to apply padding so that outputsize == inputsize / stride.

# Examples

Apply a `CrossCor` layer to a 1-channel input using a 2×2 window filter size, giving us a
16-channel output. Output is activated with ReLU.
```julia
filter = (2,2)
in = 1
out = 16
CrossCor((2, 2), 1=>16, relu)
```
"""
struct CrossCor{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

"""
    CrossCor(weight::AbstractArray, bias::AbstractArray)
    CrossCor(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the standard cross convolutional layer with user defined weight and bias
arrays.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
For input dimension N,
`pad` should be a single Integer indicating equal padding value for each spatial dimension,
a tuple of length (N-2) to apply symmetric padding or a tuple of length 2*(N-2)
indicating padding values for each spatial dimension at both the ends.
`stride` and `dilation` should be a single Integer or a tuple with N-2 parameters.

For keyword-only constuctor, see also [`Conv`](@ref)
"""
function CrossCor(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
                  stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(CrossCor, pad, size(w)[1:N-2], dilation, stride)
  return CrossCor(σ, w, b, stride, pad, dilation)
end

function CrossCor(;weight::AbstractArray{T,N}, bias::Union{Zeros, AbstractVector{T}},
                      activation = identity, stride = 1, pad = 0, dilation = 1) where {T,N}
  CrossCor(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

function CrossCor(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                  init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                  weight = convfilter(k, ch, init = init), bias = zeros(ch[2])) where N

  CrossCor(weight, bias, σ,
       stride = stride, pad = pad, dilation = dilation)
end

@functor CrossCor

function crosscor(x, w, ddims::DenseConvDims)
  ddims = DenseConvDims(ddims, F=true)
  return conv(x, w, ddims)
end

function (c::CrossCor)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(crosscor(x, c.weight, cdims) .+ b)
end

function Base.show(io::IO, l::CrossCor)
  print(io, "CrossCor(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::CrossCor{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::CrossCor{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

outdims(l::CrossCor, isize) =
  output_size(DenseConvDims(_paddims(isize, size(l.weight)), size(l.weight); stride = l.stride, padding = l.pad, dilation = l.dilation))

"""
    AdaptiveMaxPool(out)

Adaptive max pooling layer. `out` is the desired output size (batch and channel dimension excluded).
"""
struct AdaptiveMaxPool{S, O}
  out::NTuple{O, Int}
  AdaptiveMaxPool(out::NTuple{O, Int}) where O = new{O + 2, O}(out)
end

function (a::AdaptiveMaxPool{S})(x::AbstractArray{T, S}) where {S, T}
  insize = size(x)[1:end-2]
  outsize = a.out
  stride = insize .÷ outsize
  k = insize .- (outsize .- 1) .* stride
  pad = 0
  pdims = PoolDims(x, k; padding=pad, stride=stride)
  return maxpool(x, pdims)
end

function Base.show(io::IO, a::AdaptiveMaxPool)
  print(io, "AdaptiveMaxPool(", a.out, ")")
end

"""
    AdaptiveMeanPool(out)

Adaptive mean pooling layer. `out` is the desired output size (batch and channel dimension excluded).
"""
struct AdaptiveMeanPool{S, O}
  out::NTuple{O, Int}
  AdaptiveMeanPool(out::NTuple{O, Int}) where O = new{O + 2, O}(out)
end

function (a::AdaptiveMeanPool{S})(x::AbstractArray{T, S}) where {S, T}
  insize = size(x)[1:end-2]
  outsize = a.out
  stride = insize .÷ outsize
  k = insize .- (outsize .- 1) .* stride
  pad = 0
  pdims = PoolDims(x, k; padding=pad, stride=stride)
  return meanpool(x, pdims)
end

function Base.show(io::IO, a::AdaptiveMeanPool)
  print(io, "AdaptiveMeanPool(", a.out, ")")
end

"""
    GlobalMaxPool()

Global max pooling layer.

Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output,
by performing max pooling on the complete (w,h)-shaped feature maps.
"""
struct GlobalMaxPool end

function (g::GlobalMaxPool)(x)
  # Input size
  x_size = size(x)
  # Kernel size
  k = x_size[1:end-2]
  # Pooling dimensions
  pdims = PoolDims(x, k)

  return maxpool(x, pdims)
end

function Base.show(io::IO, g::GlobalMaxPool)
  print(io, "GlobalMaxPool()")
end

"""
    GlobalMeanPool()

Global mean pooling layer.

Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output,
by performing mean pooling on the complete (w,h)-shaped feature maps.
"""
struct GlobalMeanPool end

function (g::GlobalMeanPool)(x)
  # Input size
  x_size = size(x)
  # Kernel size
  k = x_size[1:end-2]
  # Pooling dimensions
  pdims = PoolDims(x, k)

  return meanpool(x, pdims)
end

function Base.show(io::IO, g::GlobalMeanPool)
  print(io, "GlobalMeanPool()")
end

"""
    MaxPool(k; pad = 0, stride = k)

Max pooling layer. `k` is the size of the window for each dimension of the input.

Use `pad=SamePad()` to apply padding so that outputsize == inputsize / stride.
=======
"""
struct MaxPool{N,M}
  k::NTuple{N,Int}
  pad::NTuple{M,Int}
  stride::NTuple{N,Int}
end

function MaxPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
  stride = expand(Val(N), stride)
  pad = calc_padding(MaxPool ,pad, k, 1, stride)
  return MaxPool(k, pad, stride)
end

function (m::MaxPool)(x)
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return maxpool(x, pdims)
end

function Base.show(io::IO, m::MaxPool)
  print(io, "MaxPool(", m.k, ", pad = ", m.pad, ", stride = ", m.stride, ")")
end

outdims(l::MaxPool{N}, isize) where N = output_size(PoolDims(_paddims(isize, (l.k..., 1, 1)), l.k; stride = l.stride, padding = l.pad))

"""
    MeanPool(k; pad = 0, stride = k)

Mean pooling layer. `k` is the size of the window for each dimension of the input.

Use `pad=SamePad()` to apply padding so that outputsize == inputsize / stride.
"""
struct MeanPool{N,M}
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end

function MeanPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
  stride = expand(Val(N), stride)
  pad = calc_padding(MeanPool, pad, k, 1, stride)
  return MeanPool(k, pad, stride)
end

function (m::MeanPool)(x)
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return meanpool(x, pdims)
end

function Base.show(io::IO, m::MeanPool)
  print(io, "MeanPool(", m.k, ", pad = ", m.pad, ", stride = ", m.stride, ")")
end

outdims(l::MeanPool{N}, isize) where N = output_size(PoolDims(_paddims(isize, (l.k..., 1, 1)), l.k; stride = l.stride, padding = l.pad))
