import NNlibCUDA: batchnorm, ∇batchnorm

function (BN::Flux.BatchNorm)(x::CuArray{T},
                              cache = nothing) where T<:Union{Float32, Float64}
  
  @assert BN.affine throw(ArgumentError("BatchNorm: only affine = true supported on gpu"))
  @assert BN.track_stats throw(ArgumentError("BatchNorm: only track_stats = true supported on gpu"))
  return BN.λ.(batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; 
                  cache = cache, alpha = 1, beta = 0, eps = BN.ϵ, 
                  training = Flux._isactive(BN)))
end

@adjoint function batchnorm(g, b, x, running_mean, running_var, momentum; kw...)
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kw...) 
  function batchnorm_pullback(Δ)
    ∇batchnorm(g, b, x, Δ, running_mean, running_var, momentum; kw...)..., nothing, nothing, nothing
  end
  y, batchnorm_pullback
end
