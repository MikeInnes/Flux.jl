module Flux

# Zero Flux Given

using Base: tail
using MacroTools, Juno, Requires, Reexport, Statistics, Random
using MacroTools: @forward

export Chain, Dense,
       RNN, LSTM, PLSTM, FCLSTM, GRU,
       Conv, ConvTranspose, MaxPool, MeanPool,
       DepthwiseConv, Dropout, LayerNorm, BatchNorm,
       params, mapleaves, cpu, gpu, f32, f64

@reexport using NNlib

include("tracker/Tracker.jl")
using .Tracker
using .Tracker: data
export Tracker, TrackedArray, TrackedVector, TrackedMatrix, param

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
  ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM,
  ADAMW, InvDecay, ExpDecay, WeightDecay

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

include("data/Data.jl")

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda/cuda.jl")

end # module
