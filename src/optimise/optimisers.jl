function descent(p::Param, η::Real)
  function ()
    @. p.x -= η * p.Δ
    @. p.Δ = 0
  end
end

function momentum(p::Param, ρ, η)
  v = zeros(p.x)
  function ()
    @. v = ρ * v - η * p.Δ
    @. p.Δ = -v
  end
end

# Ref. https://arxiv.org/pdf/1212.0901.pdf
function nesterov(p::Param, ρ, η)
  v = zeros(p.x)
  function ()
    d = @. ρ^2 * v - (1+ρ) * η * p.Δ
    @. v = ρ*v - η*p.Δ
    @. p.Δ = -d
  end
end

"""
    Spectral norm regularization as proposed in 
    Spectral Norm Regularization for Improving the Generalizability of Deep Learning, Yuichi Yoshida, Takeru Miyato, 2017
    https://arxiv.org/pdf/1705.10941.pdf
"""
function spectral(p::Param, λ::Real)
  n,m = size(p.x)
  u = similar(p.x,n)
  u .= randn(n)
  v = similar(p.x,m)
  v .= randn(m)
  function ()
    u .= p.x * v
    v .= (u' * p.x )'
    σ = norm(v)/norm(u)
    v ./=norm(v)
    u ./=norm(u)
    p.Δ .+= λ*σ*u*v'
  end
end

function rmsprop(p::Param; η::Real = 0.001, ρ::Real = 0.9, ϵ::Real = 1e-8)
  acc  = zeros(p.x)
  function ()
    @. acc = ρ * acc + (1 - ρ) * p.Δ^2
    @. p.Δ *= η / (√acc + ϵ)
  end
end

function adagrad(p::Param; η::Real = 0.01, ϵ::Real = 1e-8)
  acc = zeros(p.x) .+ ϵ
  function ()
    @. acc += p.Δ^2
    @. p.Δ *= η / √acc
  end
end

function adadelta(p::Param; ρ::Real = 0.9, ϵ::Real = 1e-8)
  acc = zeros(p.x)
  Δacc = zeros(p.x)
  function ()
    @. acc = ρ * acc + (1 - ρ) * p.Δ^2
    @. p.Δ *= √(Δacc + ϵ) / √(acc + ϵ)
    @. Δacc = ρ * Δacc + (1 - ρ) * p.Δ^2
   end
end

function adam(p::Param; η::Real = 0.001, β1::Real = 0.9, β2::Real = 0.999, ϵ::Real = 1e-8)
  mt = zeros(p.x)
  vt = zeros(p.x)
  β1p, β2p = β1, β2
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ^2
    @. p.Δ =  mt / (1 - β1p) / (√(vt / (1 - β2p)) + ϵ) * η
    β1p *= β1
    β2p *= β2
  end
end

function adamax(p::Param; η::Real = 0.002, β1::Real = 0.9, β2::Real = 0.999, ϵ::Real = 1e-8)
  mt = zeros(p.x)
  ut = zeros(p.x)
  β1p = β1
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. ut = max(β2 * ut, abs(p.Δ))
    @. p.Δ = (η/(1 - β1p)) * mt/(ut + ϵ)
    β1p *= β1
  end
end

function amsgrad(p::Param; η::Real = 0.001, β1::Real = 0.9, β2::Real = 0.999, ϵ::Real = 1e-8)
  mt = zeros(p.x)
  vt = zeros(p.x) .+ ϵ
  v̂t = zeros(p.x) .+ ϵ
  function ()
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ ^ 2
    @. v̂t = max.(v̂t, vt)
    @. p.Δ = η * mt / √v̂t
  end
end

clip(p::Param, thresh::Real) = () -> clamp!(p.Δ, -thresh, thresh)

function expdecay(p::Param, γ::Real)
  if γ != 0
    return () -> p.Δ .+= γ .* p.x
  else
    return () -> nothing
  end
end

function invdecay(p::Param, γ::Real)
  if γ != 0
    n = 0
    return () -> begin
      p.Δ .*= 1 / (1 + γ * n)
      n += 1
    end
  else
    return () -> nothing
  end
end
