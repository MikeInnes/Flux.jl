# Recurrent Models

## Recurrent Cells

In the simple feedforward case, our model `m` is a simple function from various inputs `xᵢ` to predictions `yᵢ`. (For example, each `x` might be an MNIST digit and each `y` a digit label.) Each prediction is completely independent of any others, and using the same `x` will always produce the same `y`.

```julia
y₁ = f(x₁)
y₂ = f(x₂)
y₃ = f(x₃)
# ...
```

Recurrent networks introduce a *hidden state* that gets carried over each time we run the model. The model now takes the old `h` as an input, and produces a new `h` as output, each time we run it.

```julia
h = # ... initial state ...
h, y₁ = f(h, x₁)
h, y₂ = f(h, x₂)
h, y₃ = f(h, x₃)
# ...
```

Information stored in `h` is preserved for the next prediction, allowing it to function as a kind of memory. This also means that the prediction made for a given `x` depends on all the inputs previously fed into the model.

(This might be important if, for example, each `x` represents one word of a sentence; the model's interpretation of the word "bank" should change if the previous input was "river" rather than "investment".)

Flux's RNN support closely follows this mathematical perspective. The most basic RNN is as close as possible to a standard `Dense` layer, and the output is also the hidden state.

```julia
Wxh = randn(5, 10)
Whh = randn(5, 5)
b   = randn(5)

function rnn(h, x)
  h = tanh.(Wxh * x .+ Whh * h .+ b)
  return h, h
end

x = rand(10) # dummy data
h = rand(5)  # initial hidden state

h, y = rnn(h, x)
```

If you run the last line a few times, you'll notice the output `y` changing slightly even though the input `x` is the same.

We sometimes refer to functions like `rnn` above, which explicitly manage state, as recurrent *cells*. There are various recurrent cells available, which are documented in the [layer reference](layers.md). The hand-written example above can be replaced with:

```julia
using Flux

rnn2 = Flux.RNNCell(10, 5)

x = rand(10) # dummy data
h = rand(5)  # initial hidden state

h, y = rnn2(h, x)
```

## Stateful Models

For the most part, we don't want to manage hidden states ourselves, but to treat our models as being stateful. Flux provides the `Recur` wrapper to do this.

```julia
x = rand(10)
h = rand(5)

m = Flux.Recur(rnn, h)

y = m(x)
```

The `Recur` wrapper stores the state between runs in the `m.state` field.

If you use the `RNN(10, 5)` constructor – as opposed to `RNNCell` – you'll see that it's simply a wrapped cell.

```julia
julia> RNN(10, 5)
Recur(RNNCell(10, 5, tanh))
```

## Sequences

Often we want to work with sequences of inputs, rather than individual `x`s.

```julia
seq = [rand(10) for i = 1:10]
```

With `Recur`, applying our model to each element of a sequence is trivial:

```julia
m.(seq) # returns a list of 5-element vectors
```

This works even when we've chain recurrent layers into a larger model.

```julia
m = Chain(LSTM(10, 15), Dense(15, 5))
m.(seq)
```

Finally, we can reset the hidden state of the cell back to its initial value using `reset!(m)`. We can also get the hidden states of a model using `Flux.states(m)` or load states back using `Flux.loadstates!`. 