function (a::Dense{F,<:CUDA.CuArray{wT},<:CUDA.CuArray{bT}}(x::CUDA.CuArray{T}) where
							    F, T <: Float16, wT <: Float16, bT <: Float16
  # low precision mul! with accumulating into Float32
  # needed for dispatching to the correct low precision
  # kernel in CUDA.jl to use the tensor cores
  y = _lowprecmul(a.W, x)
  a.σ.(y .+ a.b)
end

function _lowprecmul(A::CUDA.CuArray{T}, B::CUDA.CuArray{T}) where T <: Float16
  y = similar(x, Float32, (size(A, 1), size(B, 2)))
  CUDA.mul!(y, A, B)
  y
end

@adjoint function _lowprecmul(A, B)
  _lowprecmul(A, B), Δ -> (Δ * B', A' * Δ)
end
