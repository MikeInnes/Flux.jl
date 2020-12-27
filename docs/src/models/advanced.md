# Advanced Model Building and Customisation

Here we will try and describe usage of some more advanced features that Flux provides to give more control over model building.

## Customising Parameter Collection for a Model

Taking reference from our example `Affine` layer from the [basics](basics.md#Building-Layers-1).

By default all the fields in the `Affine` type are collected as its parameters, however, in some cases it may be desired to hold other metadata in our "layers" that may not be needed for training, and are hence supposed to be ignored while the parameters are collected. With Flux, it is possible to mark the fields of our layers that are trainable in two ways.

The first way of achieving this is through overloading the `trainable` function.

```julia-repl
julia> @functor Affine

julia> a = Affine(rand(3,3), rand(3))
Affine{Array{Float64,2},Array{Float64,1}}([0.66722 0.774872 0.249809; 0.843321 0.403843 0.429232; 0.683525 0.662455 0.065297], [0.42394, 0.0170927, 0.544955])

julia> Flux.params(a) # default behavior
Params([[0.66722 0.774872 0.249809; 0.843321 0.403843 0.429232; 0.683525 0.662455 0.065297], [0.42394, 0.0170927, 0.544955]])

julia> Flux.trainable(a::Affine) = (a.W,)

julia> Flux.params(a)
Params([[0.66722 0.774872 0.249809; 0.843321 0.403843 0.429232; 0.683525 0.662455 0.065297]])
```

Only the fields returned by `trainable` will be collected as trainable parameters of the layer when calling `Flux.params`.

Another way of achieving this is through the `@functor` macro directly. Here, we can mark the fields we are interested in by grouping them in the second argument:

```julia
Flux.@functor Affine (W,)
```

However, doing this requires the `struct` to have a corresponding constructor that accepts those parameters.

## Freezing Layer Parameters

When it is desired to not include all the model parameters (for e.g. transfer learning), we can simply not pass in those layers into our call to `params`.

Consider a simple multi-layer perceptron model where we want to avoid optimising the first two `Dense` layers. We can obtain
this using the slicing features `Chain` provides:

```julia
m = Chain(
      Dense(784, 64, relu),
      Dense(64, 64, relu),
      Dense(32, 10)
    )

ps = Flux.params(m[3:end])
```

The `Zygote.Params` object `ps` now holds a reference to only the parameters of the layers passed to it.

During training, the gradients will only be computed for (and applied to) the last `Dense` layer, therefore only that would have its parameters changed.

`Flux.params` also takes multiple inputs to make it easy to collect parameters from heterogenous models with a single call. A simple demonstration would be if we wanted to omit optimising the second `Dense` layer in the previous example. It would look something like this:

```julia
Flux.params(m[1], m[3:end])
```

Sometimes, a more fine-tuned control is needed. 
We can freeze a specific parameter of a specific layer which already entered a `Params` object `ps`, 
by simply deleting it from `ps`:

```julia
ps = params(m)
delete!(ps, m[2].b) 
```

## Custom multiple input or output layer, and the custom parallel layer

Sometimes a model needs to receive several separate inputs at once, or several separate outputs are required. Sometimes parallel separate paths within the deeper structures of the model are advantageous.

With FluxML, custom layers can be implemented that allow multiple inputs, outputs or even internal parallel structures. The custom layers that can be used for this purpose are explained below, namely Join, Split and Parallel. All layers should work with gpu acceleration. The examples have not been tested or optimized for performance.

These layers are also already implemented as standard layers in Flux, and can be accessed via `Flux.Join(..)`, `Flux.Split(..)`, or `Flux.Parallel(..)`.

### The custom multiple input layer: the custom join layer

By using the following layer, your model can receive multiple inputs through a single tuple. `CustomJoin(a,b)` receives a tuple with two entries ~ `([], [])` and returns a concatenated vector of the outputs of each path. The length of the tuple is arbitrary in the following code, so the input tuple can contain 2, 3, or more arrays. The defined number of paths (separated by a comma: `CustomJoin(p1,p2,...)`) and the length of the input tuple must be equal. It is recommended to use simple formats for the output, e.g. layers with a vector as output.

```julia
using Flux
using CUDA

# custom join layer
struct CustomJoin
  fs
end

function CustomJoin(fs...)
  CustomJoin(fs)
end

function (w::CustomJoin)(t::Tuple)
  vcat([w.fs[i](t[i]) for i in 1:length(w.fs)]...)
end

Flux.@functor CustomJoin

# test
model = Chain(
  CustomJoin(
    Chain(
      Dense(1, 5),
      Dense(5, 1)
    ),
    Dense(1, 2),
    Dense(1, 1),
  ),
  Dense(4, 1)
) |> gpu

tuple_input = cu(rand(1), rand(1), rand(1))

model(tuple_input)
# returns a single float vector with one value
```

### The custom multiple output layer: the custom split layer

By using the following layer, your model can return multiple outputs as tuples. `CustomSplit(a,b)` receives a single array but returns a tuple with two or more entries ~ `([], [])`. The length of the tuple is arbitrary and depends on the number of implemented paths. It is recommended to use simple formats for the output, e.g. dense layer output.

```julia
using Flux
using CUDA

custom split layer
struct CustomSplit
  fs
end

function CustomSplit(fs...)
  CustomSplit(fs)
end

function (w::CustomSplit)(x::AbstractArray)
  tuple([w.fs[i](x) for i in 1:length(w.fs)])
end

Flux.@functor CustomSplit

# test
model = Chain(
  Dense(1, 1),
  CustomSplit(
    Dense(1, 1),
    Dense(1, 1),
    Dense(1, 1)
  )
) |> gpu

model(cu(rand(1))) 
# returns a tuple with three float vectors, each with one value
```

A custom loss function for the multiple outputs may look like this:

```
using Statistics
function loss(x, y)
  # rms over all the mse
  sqrt(mean([Flux.mse(modelSplit(x)[i], y[i]) for i in 1:length(y)].^2.))
end
```

### The custom multiple paths internal layer: the parallel layer

By using the following layer, your model can calculate several separate layers within your model separately. CustomParallel(a,b)` receives a single input vector and returns a merged vector of the outputs of each path. However, multiple forward paths are created within the parallel layer. It is recommended to use simple formats for the output, e.g. the output of the dense layer.

```julia
using Flux
using CUDA

# custom parallel layer
struct CustomParallel
  fs
end

function CustomParallel(fs...)
  CustomParallel(fs)
end

function (w::CustomParallel)(x::AbstractArray)
  vcat([w.fs[i](x) for i in 1:length(w.fs)]...)
end

Flux.@functor CustomParallel

# test
model = Chain(
  Dense(1, 1),
  CustomParallel(
    Dense(1, 1),
    Dense(1, 3),
    Chain(
      Dense(1, 5),
      Dense(5, 2),
    )
  ),
  Dense(6, 1)
) |> gpu

model(cu((rand(1))))
# returns a single float vector with one value
```








