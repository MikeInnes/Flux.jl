module Flux

# Zero Flux Given

using Base: tail
using Zygote, MacroTools, Juno, Reexport, Statistics, Random
using MacroTools: @forward
@reexport using NNlib
using Zygote: Params, @adjoint, gradient, pullback, @nograd
export gradient

export Chain, Dense, Maxout, RNN, LSTM, GRU, Conv, CrossCor, ConvTranspose, MaxPool, MeanPool,
       DepthwiseConv, Dropout, AlphaDropout, LayerNorm, BatchNorm, InstanceNorm, GroupNorm,
       SkipConnection, params, fmap, cpu, gpu, f32, f64

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
  ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, Lookahead,
  ADAMW, RADAM, InvDecay, ExpDecay, WeightDecay


using CuArrays
const use_cuda = Ref(false)

include("utils.jl")
include("onehot.jl")
include("functor.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

include("data/Data.jl")

include("deprecations.jl")

function __init__()
  precompiling = ccall(:jl_generating_output, Cint, ()) != 0

  # we don't want to include the CUDA module when precompiling,
  # or we could end up replacing it at run time (triggering a warning)
  precompiling && return

  if !CuArrays.functional()
    # nothing to do here, and either CuArrays or one of its dependencies will have warned
  else
    use_cuda[] = true

    # FIXME: this functionality should be conditional at run time by checking `use_cuda`
    #        (or even better, get moved to CuArrays.jl as much as possible)
    if CuArrays.has_cudnn()
      include(joinpath(@__DIR__, "cuda/cuda.jl"))
    else
      @warn "CuArrays.jl did not find libcudnn. Some functionality will not be available."
    end
  end
end

end # module
