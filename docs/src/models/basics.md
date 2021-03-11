# Model-Building Basics

## Taking Gradients

Flux's core feature is taking gradients of Julia code. The [`gradient`](@ref) function takes another Julia function `f` and a set of arguments, and returns the gradient with respect to each argument. (It's a good idea to try pasting these examples in the Julia terminal.)

```jldoctest basics
julia> using Flux

julia> f(x) = 3x^2 + 2x + 1;

julia> df(x) = gradient(f, x)[1]; # df/dx = 6x + 2

julia> df(2)
14

julia> d2f(x) = gradient(df, x)[1]; # d²f/dx² = 6

julia> d2f(2)
6
```

When a function has many parameters, we can get gradients of each one at the same time by passing each one as an argument:

```jldoctest basics
julia> f(x, y) = sum((x .- y).^2);
```
For this `f`, we can get the gradient with respect to both `x` and `y` in a single call:
```jldoctest basics
julia> gradient(f, [2, 1], [2, 0])
([0, 2], [0, -2])
```
where `x = [2, 1]` and `y = [2, 0]`. The above syntax is the same as: `df(x,y) = gradient(f,x,y); df([2,1], [2,0])`

But machine learning models can have *hundreds* of parameters! To handle this, Flux lets you work with collections of parameters, via `params`. You can get the gradient of all parameters used in a program without explicitly passing them in.

```jldoctest basics
julia> x = [2, 1];

julia> y = [2, 0];

julia> gs = gradient(params(x, y)) do
         f(x, y)
       end
Grads(...)

julia> gs[x]
2-element Array{Int64,1}:
 0
 2

julia> gs[y]
2-element Array{Int64,1}:
  0
 -2
```

Here, `gradient` takes a zero-argument function; no arguments are necessary because the `params` tell it what to differentiate.

This will come in really handy when dealing with big, complicated models. For now, though, let's start with something simple.

## Simple Models

Consider a simple linear regression, which tries to predict an output array `y` from an input `x`.

```julia
W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3
```

To improve the prediction we can take the gradients of the loss with respect to `W` and `b` and perform gradient descent.

```julia
using Flux

gs = gradient(() -> loss(x, y), params(W, b))
```

Now that we have gradients, we can pull them out and update `W` to train the model.

```julia
W̄ = gs[W]

W .-= 0.1 .* W̄

loss(x, y) # ~ 2.5
```

The loss has decreased a little, meaning that our prediction `x` is closer to the target `y`. If we have some data we can already try [training the model](../training/training.md).

All deep learning in Flux, however complex, is a simple generalisation of this example. Of course, models can *look* very different – they might have millions of parameters or complex control flow. Let's see how Flux handles more complex models.

## Building Layers

It's common to create more complex models than the linear regression above. For example, we might want to have two linear layers with a nonlinearity like [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) (`σ`) in between them. In the above style we could write this as:

```julia
using Flux

W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1 * x .+ b1

W2 = rand(2, 3)
b2 = rand(2)
layer2(x) = W2 * x .+ b2

model(x) = layer2(σ.(layer1(x)))

model(rand(5)) # => 2-element vector
```

This works but is fairly unwieldy, with a lot of repetition – especially as we add more layers. One way to factor this out is to create a function that returns linear layers.

```julia
function linear(in, out)
  W = randn(out, in)
  b = randn(out)
  x -> W * x .+ b
end

linear1 = linear(5, 3) # we can access linear1.W etc
linear2 = linear(3, 2)

model(x) = linear2(σ.(linear1(x)))

model(rand(5)) # => 2-element vector
```

Another (equivalent) way is to create a struct that explicitly represents the affine layer.

```julia
struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) =
  Affine(randn(out, in), randn(out))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10, 5)

a(rand(10)) # => 5-element vector
```

Congratulations! You just built the `Dense` layer that comes with Flux. Flux has many interesting layers available, but they're all things you could have built yourself very easily.

(There is one small difference with `Dense` – for convenience it also takes an activation function, like `Dense(10, 5, σ)`.)

## Stacking It Up

It's pretty common to write models that look something like:

```julia
layer1 = Dense(10, 5, σ)
# ...
model(x) = layer3(layer2(layer1(x)))
```

For long chains, it might be a bit more intuitive to have a list of layers, like this:

```julia
using Flux

layers = [Dense(10, 5, σ), Dense(5, 2), softmax]

model(x) = foldl((x, m) -> m(x), layers, init = x)

model(rand(10)) # => 2-element vector
```

Handily, this is also provided for in Flux:

```julia
model2 = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)

model2(rand(10)) # => 2-element vector
```

This quickly starts to look like a high-level deep learning library; yet you can see how it falls out of simple abstractions, and we lose none of the power of Julia code.

A nice property of this approach is that because "models" are just functions (possibly with trainable parameters), you can also see this as simple function composition.

```julia
m = Dense(5, 2) ∘ Dense(10, 5, σ)

m(rand(10))
```

Likewise, `Chain` will happily work with any Julia function.

```julia
m = Chain(x -> x^2, x -> x+1)

m(5) # => 26
```

## Layer helpers

Flux provides a set of helpers for custom layers, which you can enable by calling

```julia
Flux.@functor Affine
```

This enables a useful extra set of functionality for our `Affine` layer, such as [collecting its parameters](../training/optimisers.md) or [moving it to the GPU](../gpu.md).

For some more helpful tricks, including parameter freezing, please checkout the [advanced usage guide](advanced.md).
