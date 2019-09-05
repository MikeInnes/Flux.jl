module Flux

# Zero Flux Given

using Base: tail
using MacroTools, Juno, Reexport, Statistics, Random
using MacroTools: @forward

export Chain, Dense, Maxout, RNN, LSTM, GRU, Conv, CrossCor, ConvTranspose, MaxPool, MeanPool,
       DepthwiseConv, Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       SkipConnection,
       HyperNet, HyperDense,
       params, mapleaves, cpu, gpu, f32, f64

@reexport using NNlib

using Tracker
using Tracker: data
export Tracker, TrackedArray, TrackedVector, TrackedMatrix, param

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
  ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM,
  ADAMW, RADAM, InvDecay, ExpDecay, WeightDecay

using CUDAapi
if has_cuda()
  try
    using CuArrays
    @eval has_cuarrays() = true
  catch ex
    @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
    @eval has_cuarrays() = false
  end
else
  has_cuarrays() = false
end

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")
include("layers/hypernet.jl")

include("data/Data.jl")

if has_cuarrays()
  include("cuda/cuda.jl")
end

end # module
