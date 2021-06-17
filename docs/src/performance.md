# Performance Tips

All the usual [Julia performance tips apply](https://docs.julialang.org/en/v1/manual/performance-tips/).
As always [profiling your code](https://docs.julialang.org/en/v1/manual/profile/#Profiling-1) is generally a useful way of finding bottlenecks.
Below follow some Flux specific tips/reminders.

## Don't write loss functions that use a non-constant globally declared model.
This is a special case of one of the most important [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-global-variables-1).
Non-constant global variables are slow.
We repeat it here as it is a common mistake.

This advice is appliable also to writing callbacks, and more generally to all Julia code.

### Don't write:
```julia
data = ...
m = Chain(Dense(784, 32, σ), Dense(32, 10), softmax)
loss(x, y) = Flux.mse(m(x), y)

Flux.train!(loss, Flux.params(m), data, Descent(0.1))
```
In this bad code, the model `m` is a non-constant global.
It is being used inside the function `loss`, which is one of the most performance critical parts of this code.
It will be slow, as the compiler can't rely on `m` always being the same type -- it is a mutable global, it could change at any time.

### Correct alternatives:
#### Mark the model `const`
```julia
data = ...
const m = Chain(Dense(784, 32, σ), Dense(32, 10), softmax)
loss(x, y) = Flux.mse(m(x), y)

Flux.train!(loss, Flux.params(m), data, Descent(0.1))
```
Similarly anything else that is a non-constant global that is used in functions should also be made constant

#### Put everything in a main function:
For more flexibility, you could even make this take `m` as a argument -- it doesn't matter of `m` was originally declared as a non-const global once it has been passed in as a argument because it then becomes a local variable.
```julia
function main(data)
    m = Chain(Dense(784, 32, σ), Dense(32, 10), softmax)
    loss(x, y) = Flux.mse(m(x), y)

    Flux.train!(loss, Flux.params(m), data, Descent(0.1))
end
```

#### Make the loss function actually close over `m`.
Closures can be very useful.

```julia
data = ...
m = Chain(Dense(784, 32, σ), Dense(32, 10), softmax)
get_loss_function(mdl) = (x, y) -> Flux.mse(mdl(x), y)

Flux.train!(get_loss_function(m), Flux.params(m), data, Descent(0.1))
```
This example is particularly applicable to callbacks.

## Don't use more precision than you need

Flux works great with all kinds of number types.
But often you do not need to be working with say `Float64` (let alone `BigFloat`).
Switching to `Float32` can give you a significant speed up,
not because the operations are faster, but because the memory usage is halved.
Which means allocations occur much faster.
And you use less memory.

## Preserve inputs' types

Not only should your activation and loss functions be [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/#Write-%22type-stable%22-functions-1),
they should also preserve the type of their inputs.

A very artificial example using an activation function like

```julia
my_tanh(x) = Float64(tanh(x))
```

will result in performance on `Float32` input orders of magnitude slower than the normal `tanh` would,
because it results in having to use slow mixed type multiplication in the dense layers.
Similar situations can occur in the loss function during backpropagation.

Which means if you change your data say from `Float64` to `Float32` (which should give a speedup: see above),
you will see a large slow-down.

This can occur sneakily, because you can cause type-promotion by interacting with a numeric literals.
E.g. the following will have run into the same problem as above:

```julia
leaky_tanh(x) = 0.01*x + tanh(x)
```

While one could change the activation function (e.g. to use `0.01f0*x`), the idiomatic (and safe way)  to avoid type casts whenever inputs changes is to use `oftype`:

```julia
leaky_tanh(x) = oftype(x/1, 0.01)*x + tanh(x)
```

## Evaluate batches as Matrices of features

While it can sometimes be tempting to process your observations (feature vectors) one at a time
e.g.

```julia
function loss_total(xs::AbstractVector{<:Vector}, ys::AbstractVector{<:Vector})
    sum(zip(xs, ys)) do (x, y_target)
        y_pred = model(x)  # evaluate the model
        return loss(y_pred, y_target)
    end
end
```

It is much faster to concatenate them into a matrix,
as this will hit BLAS matrix-matrix multiplication, which is much faster than the equivalent sequence of matrix-vector multiplications.
The improvement is enough that it is worthwhile allocating new memory to store them contiguously.

```julia
x_batch = reduce(hcat, xs)
y_batch = reduce(hcat, ys)
...
function loss_total(x_batch::Matrix, y_batch::Matrix)
    y_preds = model(x_batch)
    sum(loss.(y_preds, y_batch))
end
```

When doing this kind of concatenation use `reduce(hcat, xs)` rather than `hcat(xs...)`.
This will avoid the splatting penalty, and will hit the optimised `reduce` method.
