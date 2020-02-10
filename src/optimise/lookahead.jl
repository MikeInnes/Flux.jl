"""
    Lookahead(opt, α = 0.5, k = 6, state_handler = default_state_handler)

Implements the Lookahead optimiser.

## Parameters
  - Inner Optimiser
  - alpha (`α::Float64`): The slow weights (outer loop) step size. Defaults to `0.5`.
  - k : The inner-loop step. The outer loop will update per k inner-loop step.
  - state_handler: function for handling the inner optimiser states. Default do nothing. see also [`reset_state_handler!`](@ref), [`pullback_momentum_handler!`](@ref)

## Examples
```julia
inner_opt = ADAM() # create the inner optimiser

opt = Lookahead(inner_opt, α = 0.7, k = 10)
```

For using with custom Optimiser, overload the corresponding methods.

## References
[Lookahead](https://arxiv.org/pdf/1907.08610.pdf) optimiser.
"""
mutable struct Lookahead{O <: AbstractOptimiser, H <: Function} <: AbstractOptimiser
  opt::O
  state_handler::H
  alpha::Float64
  k::Int
  state::IdDict
end

GradientStyle(::Type{Lookahead}) = StatefulGradient()

"do nothing"
default_state_handler(o::Lookahead, x) = nothing

function Lookahead(opt::O, α = 0.5, k = 6, state_handler = default_state_handler) where {O <: AbstractOptimiser}
  require_state(state_handler) && !is_stateful(O) && error("Inner optimizer $opt does not have momentum")
  Lookahead{typeof(opt), typeof(state_handler)}(opt, state_handler, α, k, IdDict())
end

init_state(o::Lookahead, x) = (similar(x) .= x, 1)
get_states(o::Lookahead) = o.state

function apply!(o::Lookahead, x, Δ)
  slow_x, t = get_state!(o, x)
  apply!(o.opt, x, Δ)
  if t >= o.k
    t = 0
    α = o.alpha
    fast_x = x .- Δ
    @. slow_x = α * fast_x + (1 - α) * slow_x
    @. Δ = x - slow_x
    o.state_handler(o, x)
  end
  o.state[x] = (slow_x, t+1)
  return Δ
end

"return `true` if the handler need optimiser has momentum."
require_state(::Function) = true
require_state(::typeof(default_state_handler)) = false

@inline _new_state!(o::Union{Momentum, Nesterov, RMSProp}, x) = _reset!(get_state!(o, x))
@inline _new_state!(o::Union{ADAM, AdaMax, NADAM}, x) = (s = get_state!(o, x); (_reset!(s[1]), _reset!(s[2]), o.beta))
@inline _new_state!(o::RADAM, x) = (s = get_state!(o, x); (_reset!(s[1]), _reset!(s[2]), o.beta, 1))
@inline _new_state!(o::ADAGrad, x) = fill!(get_state!(o, x), ϵ)
@inline _new_state!(o::ADADelta, x) = _reset!.(get_state!(o, x))
@inline _new_state!(o::AMSGrad, x) = fill!.(get_state!(o, x), ϵ)

_reset!(x::AbstractArray{T}) where T = (x .= zero(T))

reset_state!(o, x) = (get_states(o)[x] = _new_state!(o, x); nothing)
reset_state!(o::Optimiser, x) = (map(Base.Fix2(reset_state!, x), filter(is_stateful, o.os)); nothing)

# state handlers

"Reset the inner optimiser momentums. Need to overload `reset_state!` for custom optimiser."
reset_state_handler!(o::Lookahead, x) = reset_state!(o.opt, x)

"return the momentum arrays in `Tuple`"
@inline momentum_buffer(o, x) = (s = get_state!(o, x); s isa Tuple ? s : (s,))
@inline momentum_buffer(o::Union{ADAM, RADAM, AdaMax, NADAM}, x) = get_state!(o, x)[1:2]
@inline function momentum_buffer(o::Optimiser, x)
  mapreduce(Base.Fix2(momentum_buffer, x), (init, x)->(init..., x...), filter(has_momentum, o); init=());
end

"pullback the inner momentum with outer momentum. Need to overload `momentum_buffer` for custom optimiser."
function pullback_momentum_handler!(o::Lookahead, x)
  opt, α = o.opt, o.alpha
  inter_mom = momentum_buffer(opt, x)
  outer_mom = map(inter_mom) do mom
    get!(o.state, mom, mom)
  end
  map(zip(inter_mom, outer_mom)) do (fast_mom, slow_mom)
    @. fast_mom = slow_mom = α * fast_mom + (1 - α) * slow_mom
  end
  return
end

