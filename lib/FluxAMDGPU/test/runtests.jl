using Test

using FluxAMDGPU
using FluxAMDGPU: Flux, AMDGPU

using .AMDGPU
# XXX: allowscalar is currently not inherited by child tasks, so set it globally
#AMDGPU.allowscalar(false)
ENV["JULIA_GPU_ALLOWSCALAR"] = "false"

using .Flux
@assert Flux.default_gpu_converter[] == roc

using Zygote
using Zygote: pullback

include("test_utils.jl")
include("core.jl")
include("losses.jl")
include("layers.jl")
