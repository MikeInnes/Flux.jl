using Base.Threads
import Flux
import ..Tracker: TrackedArray, back!

Base.copy!(d::Param,c::Param) = (copy!(d.x,c.x),copy!(d.Δ,c.Δ))
Base.copy!(d::Vector{Param},c::Vector{Param}) = foreach(s -> copy!(s[1],s[2]),zip(d,c))
addparams!(d::Param,c::Param) = (d.x .+=c.x,d.Δ .+=c.Δ)
addparams!(d::Vector{Param},c::Vector{Param}) = foreach(s -> addparams!(s[1],s[2]),zip(d,c))

Base.copy!(d::TrackedArray,c::TrackedArray) = (copy!(d.data,c.data),copy!(d.grad.x,c.grad.x))
Base.copy!(d::Param,c::TrackedArray) = (copy!(d.x,c.data),copy!(d.Δ,c.grad.x))

function Base.mean!(td::T,ps::T...) where {T<:Vector{Param}}
  l = length(ps)
  copy!(td,ps[1])
  foreach(s -> addparams!(td,s),ps[2:end])
  foreach(s -> (s.x ./=l,s.Δ ./=l),td)
end

function _back!(model,loss,ds)
  l = loss(model,ds)
  if isinf(l.data[])
    error("inf in the model");
  end
  if isnan(l.data[])
    error("nan in the model");
  end

  back!(l)
  l.data[]
end

function back!(pars,models,parss,dss,loss)
  foreach(s -> copy!(s,pars),parss)
  l = zeros(nthreads())
  @threads for i in 1:length(dss)
    l[threadid()] += _back!(models[threadid()],loss,dss[i])
  end
  mean!(pars,parss...)
  l
end

getpars(m) = Vector{Param}(Flux.params(m))


"""
    train_threaded(model,loss,data,opt;cb = () -> ())

multi threaded version of training. It works by creating @nthreads deepcopies of model and distributes
the calculation of gradients to individual models. Gradient is then collected and copied back to model, on which
the optimization is performed.

Note that:
  * loss accepts two parameters, first is the model and second is the dataset
  * iterator data has to provide array of datasets, which is then distributed to individual cores.

Example of use see below.

```julia
julia> using Flux
julia> using Base.Threads
julia> using Iterators
julia> makedataset() = (hcat(randn(4,50),randn(4,50).+3),vcat(ones(50),2ones(50)))
julia> model = Chain(Dense(4,2),softmax)
julia> loss(f,ds) = Flux.crossentropy(f(ds[1]),Flux.onehotbatch(ds[2],1:2))

julia> data = Iterators.repeated([makedataset() for i in 1:nthreads()],100)
julia> opt = Flux.Optimise.ADAM(params(model))
julia> Flux.train_threaded(model,loss,data,opt;cb = () -> ())
```
"""
function train_threaded(model,loss,data,opt;cb = () -> ())
  models = [deepcopy(model) for i in 1:nthreads()];
  parss = getpars.(models);
  pars = getpars(model)

  for ds in data
    l = mean(back!(pars,models,parss,ds,loss))
    opt()
    cb() == :stop && break
  end
end
