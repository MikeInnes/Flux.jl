module Optimise

export update!, train!,
  SGD, ADAM, Momentum, Nesterov, RMSProp, ADAGrad, ADADelta, AMSGrad

struct Param{T}
  x::T
  Δ::T
end

Base.convert(::Type{Param}, x::AbstractArray) = Param(x, zeros(x))

include("optimisers.jl")
include("interface.jl")
include("train.jl")
include("train_threaded.jl")

using Flux.Tracker: TrackedArray

Base.convert(::Type{Param}, x::TrackedArray) = Param(x.data, x.grad)

end
