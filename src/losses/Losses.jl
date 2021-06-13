module Losses

using Statistics
using Zygote
using Zygote: @adjoint
using ..Flux: ofeltype, epseltype
using NNlib: logsoftmax, logσ
import Base.Broadcast: broadcasted

export mse, mae, msle,
    label_smoothing,
    crossentropy, logitcrossentropy,
    binarycrossentropy, logitbinarycrossentropy,
    kldivergence,
    huber_loss,
    tversky_loss,
    dice_coeff_loss,
    poisson_loss,
    hinge_loss, squared_hinge_loss,
    ctc_loss,
    binary_focal_loss, focal_loss

include("utils.jl")
include("functions.jl")
include("ctc.jl")

end #module
