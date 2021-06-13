using Flux: OneHotArray, OneHotLike

Base.BroadcastStyle(::Type{<:OneHotArray{<: Any, <: Any, <: Any, N, <: CuArray}}) where N =
    CUDA.CuArrayStyle{N}()
