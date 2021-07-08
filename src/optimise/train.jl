using ProgressLogging: @progress
import Zygote: Params, gradient

"""
    update!(x, x̄)

Update the array `x` according to `x .-= x̄`.
"""
function update!(x::AbstractArray, x̄)
  x .-= x̄
end

"""
    update!(opt, p, g)
    update!(opt, ps::Params, gs)

Perform an update step of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).

As a result, the parameters are mutated and the optimizer's internal state may change.
The gradient could be mutated as well.
"""
function update!(opt, x, x̄)
  x̄r = ArrayInterface.restructure(x, x̄) # address some cases where Zygote's
                                          # output are not mutable, see #1510 
  x .-= apply!(opt, x, x̄r)
end

function update!(opt, xs::Params, gs)
  for x in xs
    isnothing(gs[x]) && continue
    update!(opt, x, gs[x])
  end
end

# Callback niceties
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)

struct SkipException <: Exception end

"""
    skip()

Call `Flux.skip()` in a callback to indicate when a callback condition is met.
This will trigger the train loop to skip the current data point and not update with the calculated gradient.

# Examples
```julia
cb = function ()
  loss() > 1e7 && Flux.skip()
end
```
"""
function skip()
  throw(SkipException())
end


struct StopException <: Exception end

"""
    stop()

Call `Flux.stop()` in a callback to indicate when a callback condition is met.
This will trigger the train loop to stop and exit.

# Examples
```julia
cb = function ()
  accuracy() > 0.9 && Flux.stop()
end
```
"""
function stop()
  throw(StopException())
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

"""
    train!(loss, params, data, opt; cb)

For each datapoint `d` in `data`, compute the gradient of  `loss` with
respect to `params` through backpropagation and call the optimizer `opt`.

If `d` is a tuple of arguments to `loss` call `loss(d...)`, else call `loss(d)`.

A callback is given with the keyword argument `cb`. For example, this will print
"training" every 10 seconds (using [`Flux.throttle`](@ref)):
    train!(loss, params, data, opt, cb = throttle(() -> println("training"), 10))

The callback can call [`Flux.stop`](@ref) to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, ps, data, opt; cb = () -> ())
  ps = Params(ps)
  cb = runall(cb)
  @progress for d in data
    try
      gs = gradient(ps) do
        loss(batchmemaybe(d)...)
      end
      update!(opt, ps, gs)
      cb()
    catch ex
      if ex isa StopException
        break
      elseif ex isa SkipException
        continue
      else
        rethrow(ex)
      end
    end
  end
end

"""
    @epochs N body

Run `body` `N` times. Mainly useful for quickly doing multiple epochs of
training in a REPL.

# Examples
```jldoctest
julia> Flux.@epochs 2 println("hello")
[ Info: Epoch 1
hello
[ Info: Epoch 2
hello
```
"""
macro epochs(n, ex)
  :(@progress for i = 1:$(esc(n))
      @info "Epoch $i"
      $(esc(ex))
    end)
end
