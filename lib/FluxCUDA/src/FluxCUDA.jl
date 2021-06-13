module FluxCUDA

using Flux, CUDA
using NNlib, NNlibCUDA
using Zygote
using Zygote: @adjoint

include("onehot.jl")
include("ctc.jl")
include("cudnn.jl")

function __init__()
    if Flux.default_gpu_converter[] === identity
        @info "Registering CUDA.jl as the default GPU converter"
        Flux.default_gpu_converter[] = cu
    else
        @warn "Not registering CUDA.jl as the default GPU converter as another one has been registered already."
    end
end

end # module
