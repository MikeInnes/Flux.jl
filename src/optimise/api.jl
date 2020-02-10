abstract type GradientStyle end
struct StatefulGradient <: GradientStyle end
struct StatelessGradient <: GradientStyle end

Base.promote_rule(::Type{T}, ::Type{T}) where {T <: GradientStyle} = T
Base.promote_rule(::Type{StatefulGradient}, ::Type{<:GradientStyle}) = StatefulGradient
Base.promote_rule(::Type{<:GradientStyle}, ::Type{StatefulGradient}) = StatefulGradient

"""
    GradientStyle(o)
    GradientStyle(typeof(o))

`GradientStyle` specifies the gradient type for the optimiser `o`. When you 
define a new [`AbstractOptimiser`](@ref) type, you can specify the gradient type.
If the gradient is stateful (i.e. it has a moving average, momentum, or steps), 
set the trait for your optimiser to [`StatefulGradient()`](@ref) like this:

  Flux.GradientStyle(::Type{MyOptimiser}) = StatefulGradient()

or set to [`StatelessGradient()`](@ref) if it is stateless.

see also [`is_stateful`](@ref).
"""
GradientStyle(o::AbstractOptimiser) = GradientStyle(typeof(o))
GradientStyle(::Type{M}) where {M <: AbstractGradientModifier} = StatelessGradient()
if VERSION < v"1.1"
  GradientStyle(::Type{O}) where {O <: AbstractOptimiser} = IdDict in map(Base.Fix1(fieldtype, O), fieldnames(O)) ? StatefulGradient() : StatelessGradient()
else
  GradientStyle(::Type{O}) where {O <: AbstractOptimiser} = IdDict in fieldtypes(O) ? StatefulGradient() : StatelessGradient()
end
GradientStyle(::Type{Descent}) = StatelessGradient()
GradientStyle(::Type{WeightDecay}) = StatelessGradient()

for sT in :([Momentum, Nesterov, RMSProp, ADAM, RADAM, AdaMax, ADAGrad, ADADelta, AMSGrad, NADAM, ADAMW]).args
  @eval GradientStyle(::Type{$sT}) = StatefulGradient()
end

GradientStyle(o::Type{Optimiser{T}}) where T = mapreduce(GradientStyle, promote, T.parameters)

"""
    is_stateful(o)

return `true` if the optimiser is stateful.
"""
is_stateful(gs::StatelessGradient) = false
is_stateful(gs::StatefulGradient) = true
is_stateful(::Type{O}) where {O<:AbstractOptimiser}= is_stateful(GradientStyle(O))
is_stateful(o::AbstractOptimiser) = is_stateful(typeof(o))

"""
    get_state!(o, x)

return the state in `o` corresponding to `x`.
"""
get_state!(o::O, x) where {O <: AbstractOptimiser} = get_state!(GradientStyle(O), o, x)
get_state!(gs::StatelessGradient, o, x) = error("optimiser $o is stateless")
function get_state!(gs::StatefulGradient, o, x)
  state = get_states(o)
  if haskey(state, x)
    return state[x]
  else
    return get!(state, x, init_state(o, x))
  end
end

@inline get_states(o::Union{Momentum, Nesterov}) = o.velocity
@inline get_states(o::Union{RMSProp, ADAGrad}) = o.acc
@inline get_states(o::Union{ADAM, RADAM, AdaMax, ADADelta, AMSGrad, NADAM}) = o.state
@inline get_states(o::ADAMW) = o[1].state
