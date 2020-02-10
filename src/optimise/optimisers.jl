using Flux
using Base: @get!
using MacroTools: @forward

const ϵ = 1e-8

# TODO: should use weak refs


abstract type AbstractOptimiser end
abstract type AbstractGradientModifier <: AbstractOptimiser end

"""
  Descent(η)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`

## Parameters
  - Learning Rate (η): The amount by which the gradients are discounted before updating the weights. Defaults to `0.1`.

## Example
```julia-repl
opt = Descent() # uses default η (0.1)

opt = Descent(0.3) # use provided η

ps = params(model)

gs = gradient(ps) do
  loss(x, y)
end

Flux.Optimise.update!(opt, ps, gs)
```
"""
mutable struct Descent <: AbstractOptimiser
  eta::Float64
end

Descent() = Descent(0.1)

function apply!(o::Descent, x, Δ)
  Δ .*= o.eta
end

"""
    Momentum(η, ρ)

Gradient descent with learning rate `η` and momentum `ρ`.

## Parameters
  - Learning Rate (`η`): Amount by which gradients are discounted before updating the weights. Defaults to `0.01`.
  - Momentum (`ρ`): Parameter that accelerates descent in the relevant direction and dampens oscillations. Defaults to `0.9`.

## Examples
```julia
opt = Momentum() # uses defaults of η = 0.01 and ρ = 0.9

opt = Momentum(0.01, 0.99)
```
"""
mutable struct Momentum <: AbstractOptimiser
  eta::Float64
  rho::Float64
  velocity::IdDict
end

Momentum(η = 0.01, ρ = 0.9) = Momentum(η, ρ, IdDict())

init_state(o::Momentum, x) = zero(x)

function apply!(o::Momentum, x, Δ)
  η, ρ = o.eta, o.rho
  v = get_state!(o, x)::typeof(x)
  @. v = ρ * v - η * Δ
  @. Δ = -v
end

"""
    Nesterov(η, ρ)

Gradient descent with learning rate  `η` and Nesterov momentum `ρ`.

## Parameters
  - Learning Rate (η): Amount by which the gradients are dicsounted berfore updating the weights. Defaults to `0.001`.
  - Nesterov Momentum (ρ): Paramters controlling the amount of nesterov momentum to be applied. Defaults to `0.9`.

## Examples
```julia
opt = Nesterov() # uses defaults η = 0.001 and ρ = 0.9

opt = Nesterov(0.003, 0.95)
```
"""
mutable struct Nesterov <: AbstractOptimiser
  eta::Float64
  rho::Float64
  velocity::IdDict
end

Nesterov(η = 0.001, ρ = 0.9) = Nesterov(η, ρ, IdDict())

init_state(o::Nesterov, x) = zero(x)

function apply!(o::Nesterov, x, Δ)
  η, ρ = o.eta, o.rho
  v = get_state!(o, x)::typeof(x)
  d = @. ρ^2 * v - (1+ρ) * η * Δ
  @. v = ρ*v - η*Δ
  @. Δ = -d
end

"""
    RMSProp(η, ρ)

Implements the RMSProp algortihm. Often a good choice for recurrent networks. Paramters other than learning rate generally don't need tuning.

## Parameters
  - Learning Rate (η): Defaults to `0.001`.
  - Rho (ρ): Defaults to `0.9`.

## Examples
```julia
opt = RMSProp() # uses default η = 0.001 and ρ = 0.9

opt = RMSProp(0.002, 0.95)
```

## References
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
"""
mutable struct RMSProp <: AbstractOptimiser
  eta::Float64
  rho::Float64
  acc::IdDict
end

RMSProp(η = 0.001, ρ = 0.9) = RMSProp(η, ρ, IdDict())

init_state(o::RMSProp, x) = zero(x)

function apply!(o::RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = get_state!(o, x)::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + ϵ)
end

"""
    ADAM(η, β::Tuple)

Implements the ADAM optimiser.

## Paramters
  - Learning Rate (`η`): Defaults to `0.001`.
  - Beta (`β::Tuple`): The first element refers to β1 and the second to β2. Defaults to `(0.9, 0.999)`.

## Examples

```julia
opt = ADAM() # uses the default η = 0.001 and β = (0.9, 0.999)

opt = ADAM(0.001, (0.9, 0.8))
```
## References
[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.
"""
mutable struct ADAM <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, IdDict())

init_state(o::ADAM, x) = (zero(x), zero(x), o.beta)

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get_state!(o, x)
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end

"""
    RADAM(η, β::Tuple)

Implements the rectified ADAM optimizer.

## Parameters
  - Learning Rate (η): Defaults to `0.001`
  - Beta (β::Tuple): The first element refers to β1 and the second to β2. Defaults to `(0.9, 0.999)`.

## Examples

```julia
opt = RADAM() # uses the default η = 0.001 and β = (0.9, 0.999)

opt = RADAM(0.001, (0.9, 0.8))
```

## References
[RADAM](https://arxiv.org/pdf/1908.03265v1.pdf) optimiser (Rectified ADAM).
"""
mutable struct RADAM <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

RADAM(η = 0.001, β = (0.9, 0.999)) = RADAM(η, β, IdDict())

init_state(o::RADAM, x) = (zero(x), zero(x), o.beta, 1)

function apply!(o::RADAM, x, Δ)
  η, β = o.eta, o.beta
  ρ∞ = 2/(1-β[2])-1
  mt, vt, βp, t = get_state!(o, x)
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  ρ = ρ∞ - 2t*βp[2]/(1-βp[2])
  if ρ > 4
    r = sqrt((ρ-4)*(ρ-2)*ρ∞/((ρ∞-4)*(ρ∞-2)*ρ))
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η * r
  else
    @. Δ =  mt / (1 - βp[1]) * η
  end
  o.state[x] = (mt, vt, βp .* β, t+1)
  return Δ
end

"""
    AdaMax(η, β::Tuple)

Variant of ADAM based on ∞-norm.

## Parameters
  - Learning Rate (η): Defaults to `0.001`
  - Beta (β::Tuple): The first element refers to β1 and the second to β2. Defaults to `(0.9, 0.999)`.

## Examples
```julia
opt = AdaMax() # uses default η and β

opt = AdaMax(0.001, (0.9, 0.995))
```
## References
[AdaMax](https://arxiv.org/abs/1412.6980v9) optimiser.
"""
mutable struct AdaMax <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

AdaMax(η = 0.001, β = (0.9, 0.999)) = AdaMax(η, β, IdDict())

init_state(o::AdaMax, x) = (zero(x), zero(x), o.beta)

function apply!(o::AdaMax, x, Δ)
  η, β = o.eta, o.beta
  mt, ut, βp = get_state!(o, x)
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. ut = max(β[2] * ut, abs(Δ))
  @. Δ = (η/(1 - βp[1])) * mt/(ut + ϵ)
  o.state[x] = (mt, ut, βp .* β)
  return Δ
end

"""
    ADAGrad(η)

Implements AdaGrad. It has parameter specific learning rates based on how frequently it is updated.

## Parameters
  - Learning Rate (η): Defaults to `0.1`

## Examples
```julia
opt = ADAGrad() # uses default η = 0.1

opt = ADAGrad(0.001)
```

## References
[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.
Parameters don't need tuning.
"""
mutable struct ADAGrad <: AbstractOptimiser
  eta::Float64
  acc::IdDict
end

ADAGrad(η = 0.1) = ADAGrad(η, IdDict())

init_state(o::ADAGrad, x) = fill!(similar(x), ϵ)

function apply!(o::ADAGrad, x, Δ)
  η = o.eta
  acc = get_state!(o, x)::typeof(x)
  @. acc += Δ^2
  @. Δ *= η / (√acc + ϵ)
end

"""
    ADADelta(ρ)

Version of ADAGrad that adapts learning rate based on a window of past gradient updates. Parameters don't need tuning.

## Parameters
  - Rho (ρ): Factor by which gradient is decayed at each time step. Defaults to `0.9`.

## Examples
```julia
opt = ADADelta() # uses default ρ = 0.9
opt = ADADelta(0.89)
```

## References
[ADADelta](https://arxiv.org/abs/1212.5701) optimiser.
"""
mutable struct ADADelta <: AbstractOptimiser
  rho::Float64
  state::IdDict
end

ADADelta(ρ = 0.9) = ADADelta(ρ, IdDict())

init_state(o::ADADelta, x) = (zero(x), zero(x))

function apply!(o::ADADelta, x, Δ)
  ρ = o.rho
  acc, Δacc = get_state!(o, x)
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= √Δacc/ (√acc + ϵ)
  @. Δacc = ρ * Δacc + (1 - ρ) * Δ^2
  return Δ
end

"""
    AMSGrad(η, β::Tuple)

Implements AMSGrad version of the ADAM optimiser. Parameters don't need tuning.

## Parameters
  - Learning Rate (η): Defaults to `0.001`.
  - Beta (β::Tuple): The first element refers to β1 and the second to β2. Defaults to `(0.9, 0.999)`.

## Examples
```julia
opt = AMSGrad() # uses default η and β
opt = AMSGrad(0.001, (0.89, 0.995))
```

## References
[AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) optimiser.
"""
mutable struct AMSGrad <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::IdDict
end

AMSGrad(η = 0.001, β = (0.9, 0.999)) = AMSGrad(η, β, IdDict())

init_state(o::AMSGrad, x) = (fill!(similar(x), ϵ), fill!(similar(x), ϵ), fill!(similar(x), ϵ))

function apply!(o::AMSGrad, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, v̂t = get_state!(o, x)
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
  @. v̂t = max(v̂t, vt)
  @. Δ = η * mt / (√v̂t + ϵ)
end

"""
    NADAM(η, β::Tuple)

Nesterov variant of ADAM. Parameters don't need tuning.

## Parameters
  - Learning Rate (η): Defaults to `0.001`.
  - Beta (β::Tuple): The first element refers to β1 and the second to β2. Defaults to `(0.9, 0.999)`.

## Examples
```julia
opt = NADAM() # uses default η and β
opt = NADAM(0.002, (0.89, 0.995))
```

## References
[NADAM](http://cs229.stanford.edu/proj2015/054_report.pdf) optimiser.
"""
mutable struct NADAM <: AbstractOptimiser
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::IdDict
end

NADAM(η = 0.001, β = (0.9, 0.999)) = NADAM(η, β, IdDict())

init_state(o::NADAM, x) = (zero(x), zero(x), o.beta)

function apply!(o::NADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, (β1p, β2p) = get_state!(o, x)
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ = (β[1] * mt / (1 - β[1] * β1p) + (1 - β[1]) * Δ / (1 - β1p)) / (√(vt * β[2] / (1 - β2p)) + ϵ) * η
  o.state[x] = (mt, vt, (β1p * β[1], β2p * β[2]))
  return Δ
end


# Compose optimizers

"""
    Optimiser(a, b, c...)

Combine several optimisers into one; each optimiser produces a modified gradient
that will be fed into the next, and this is finally applied to the parameter as
usual.
"""
mutable struct Optimiser{T<:Tuple} <: AbstractOptimiser
  os::T
  Optimiser(os::Tuple) = new{typeof(os)}(os)
end

Optimiser(o...) = Optimiser(o)

@forward Optimiser.os Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!, Base.setindex!
@forward Optimiser.os Base.iterate

Base.getindex(c::Optimiser, i::AbstractArray) = Optimiser(c.os[i]...)

function apply!(o::Optimiser, x, Δ)
  for opt in o.os
    Δ = apply!(opt, x, Δ)
  end
  return Δ
end

"""
  InvDecay(γ)

Applies inverse time decay to an optimiser, i.e., the effective step size at iteration `n` is `eta / (1 + γ * n)` where `eta` is the initial step size. The wrapped optimiser's step size is not modified.
```

## Parameters
  - gamma (γ): Defaults to `0.001`

## Example
```julia
  Optimiser(InvDecay(..), Opt(..))
```
"""
mutable struct InvDecay  <: AbstractGradientModifier
  gamma::Float64
  state::IdDict
end

InvDecay(γ = 0.001) = InvDecay(γ, IdDict())

function apply!(o::InvDecay, x, Δ)
  γ = o.gamma
  n = get!(o.state, x, 1)
  Δ .*= 1 / (1 + γ * n)
  o.state[x] = n + 1
  return Δ
end

"""
  ExpDecay(eta, decay, decay_step, clip)

Discount the learning rate `eta` by a multiplicative factor `decay` every `decay_step` till a minimum of `clip`.

## Parameters
  - Learning Rate (eta): Defaults to `0.001`.
  - decay: Factor by which the learning rate is discounted. Defaults to `0.1`.
  - decay_step: Schedules decay operations by setting number of steps between two decay operations. Defaults to `1000`.
  - clip: Minimum value of learning rate. Defaults to `1e-4`.

## Example
To apply exponential decay to an optimiser:
```julia
  Optimiser(ExpDecay(..), Opt(..))

  opt = Optimiser(ExpDecay(), ADAM())
```
"""
mutable struct ExpDecay <: AbstractGradientModifier
  eta::Float64
  decay::Float64
  step::Int64
  clip::Float64
  current::IdDict
end

ExpDecay(opt = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4) = ExpDecay(opt, decay, decay_step, clip, IdDict())

function apply!(o::ExpDecay, x, Δ)
  η, s, decay = o.eta, o.step, o.decay
  n = o.current[x] = get(o.current, x, 0) + 1
  if o.current[x]%s == 0 && count(x -> x%s == 0, values(o.current)) == 1
    η = max(η * decay^(s / n), o.clip)
    o.eta = η
  end
  @. Δ *= η
end

"""
  WeightDecay(wd)

Decays the weight by `wd`

## Parameters
  - weight decay (wd): 0
"""
mutable struct WeightDecay <: AbstractGradientModifier
  wd::Real
end

WeightDecay() = WeightDecay(0)

function apply!(o::WeightDecay, x, Δ)
  wd = o.wd
  @. Δ += wd * x
end

"""
    ADAMW(η, β::Tuple, decay)

Variant of ADAM defined by fixing weight decay regularization.

## Parameters
  - Learning Rate (η): Defaults to `0.001`.
  - Beta (β::Tuple): The first element refers to β1 and the second to β2. Defaults to (0.9, 0.999).
  - decay: Decay applied to weights during optimisation. Defaults to 0.

## Examples
```julia
opt = ADAMW() # uses default η, β and decay
opt = ADAMW(0.001, (0.89, 0.995), 0.1)
```

## References
[ADAMW](https://arxiv.org/abs/1711.05101)
"""
const ADAMW = Optimiser{Tuple{ADAM, WeightDecay}}
ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) =
  Optimiser(ADAM(η, β), WeightDecay(decay))

init_state(o::ADAMW, x) = init_state(o[1], x)

