module FluxAMDGPU

using Flux
using AMDGPU

### onehot

using Flux: OneHotArray, OneHotLike

Base.BroadcastStyle(::Type{<:OneHotArray{<:Any, <:Any, <:Any, N, <:ROCArray}}) where N =
    AMDGPU.ROCArrayStyle{N}()

function __init__()
    if Flux.default_gpu_converter[] === identity
        @info "Registering AMDGPU.jl as the default GPU converter"
        Flux.default_gpu_converter[] = roc
    else
        @warn "Not registering AMDGPU.jl as the default GPU converter as another one has been registered already."
    end
end

end # module
