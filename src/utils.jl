# Arrays
"""
    nfan(n_out, n_in=1) -> Tuple
    nfan(dims...)
    nfan(dims::Tuple)

For a layer characterized by dimensions `dims`, return a tuple `(fan_in, fan_out)`, where `fan_in`
is the number of input neurons connected to an output one, and `fan_out` is the number of output neurons
connected to an input one.

This function is mainly used by weight initializers, e.g., [`kaiming_normal`](@ref Flux.kaiming_normal).

# Examples

```jldoctest
julia> layer = Dense(10, 20)
Dense(10, 20)

julia> Flux.nfan(size(layer.W))
(10, 20)

julia> layer = Conv((3, 3), 2=>10)
Conv((3, 3), 2=>10)

julia> Flux.nfan(size(layer.weight))
(18, 90)
```
"""
nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) # In case of convolution kernels

ofeltype(x, y) = convert(float(eltype(x)), y)
epseltype(x) = eps(float(eltype(x)))

"""
    glorot_uniform([rng=GLOBAL_RNG], dims...)

Return an `Array` of size `dims` containing random variables taken from a uniform
distribution in the interval ``[-x, x]``, where `x = sqrt(6 / (fan_in + fan_out))`.

This method is described in [1] and also known as Xavier initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.glorot_uniform(2, 3)
2×3 Array{Float32,2}:
 0.601094  -0.57414   -0.814925
 0.900868   0.805994   0.057514
```

# See also

* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." _Proceedings of the thirteenth international conference on artificial intelligence and statistics_. 2010.
"""
glorot_uniform(rng::AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(nfan(dims...)))
glorot_uniform(dims...) = glorot_uniform(Random.GLOBAL_RNG, dims...)
glorot_uniform(rng::AbstractRNG) = (dims...) -> glorot_uniform(rng, dims...)

"""
    glorot_normal([rng=GLOBAL_RNG], dims...)

Return an `Array` of size `dims` containing random variables taken from a normal
distribution with mean 0 and standard deviation `sqrt(2 / (fan_in + fan_out))`.

This method is described in [1] and also known as Xavier initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.glorot_normal(3, 2)
3×2 Array{Float32,2}:
  0.429505  -0.0852891
  0.523935   0.371009
 -0.223261   0.188052
```

# See also

* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." _Proceedings of the thirteenth international conference on artificial intelligence and statistics_. 2010.
"""
glorot_normal(rng::AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0 / sum(nfan(dims...)))
glorot_normal(dims...) = glorot_normal(Random.GLOBAL_RNG, dims...)
glorot_normal(rng::AbstractRNG) = (dims...) -> glorot_normal(rng, dims...)

"""
    kaiming_uniform([rng=GLOBAL_RNG], dims...; gain = √2)

Return an `Array` of size `dims` containing random variables taken from a uniform distribution in the
interval `[-x, x]`, where `x = gain * sqrt(3/fan_in)`.

This method is described in [1] and also known as He initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.kaiming_uniform(3, 2)
3×2 Array{Float32,2}:
  0.950413   1.27439
  1.4244    -1.28851
 -0.907795   0.0909376
```

# See also

* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, dims...; gain = √2)
  bound = Float32(√3 * gain / sqrt(first(nfan(dims...)))) # fan_in
  return (rand(rng, Float32, dims...) .- 0.5f0) .* 2bound
end

kaiming_uniform(dims...; kwargs...) = kaiming_uniform(Random.GLOBAL_RNG, dims...; kwargs...)
kaiming_uniform(rng::AbstractRNG; kwargs...) = (dims...; kwargs...) -> kaiming_uniform(rng, dims...; kwargs...)

"""
    kaiming_normal([rng=GLOBAL_RNG], dims...; gain = √2)

Return an `Array` of size `dims` containing random variables taken from a normal
distribution with mean 0 and standard deviation `gain * sqrt(fan_in)`.

This method is described in [1] and also known as He initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.kaiming_normal(3, 2)
3×2 Array{Float32,2}:
  0.679107  -0.134854
  0.828413   0.586617
 -0.353007   0.297336
```

# See also

* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, dims...; gain = √2f0)
  std = Float32(gain / sqrt(first(nfan(dims...)))) # fan_in
  return randn(rng, Float32, dims...) .* std
end

kaiming_normal(dims...; kwargs...) = kaiming_normal(Random.GLOBAL_RNG, dims...; kwargs...)
kaiming_normal(rng::AbstractRNG; kwargs...) = (dims...; kwargs...) -> kaiming_normal(rng, dims...; kwargs...)

ones(T::Type, dims...) = Base.ones(T, dims...)
zeros(T::Type, dims...) = Base.zeros(T, dims...)

ones(dims...) = Base.ones(Float32, dims...)
zeros(dims...) = Base.zeros(Float32, dims...)

"""
    unsqueeze(xs, dim)

Return `xs` reshaped into an `Array` one dimensionality higher than `xs`,
where `dim` indicates in which dimension `xs` is extended.

# Examples
```jldoctest
julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> Flux.unsqueeze(xs, 1)
1×3 Array{Array{Int64,1},2}:
 [1, 2]  [3, 4]  [5, 6]

julia> Flux.unsqueeze([1 2; 3 4], 2)
2×1×2 Array{Int64,3}:
[:, :, 1] =
 1
 3

[:, :, 2] =
 2
 4
```
"""
unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))

"""
    stack(xs, dim)

Concatenate the given `Array` of `Array`s `xs` into a single `Array` along the
given dimension `dim`.

# Examples
```jldoctest
julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> Flux.stack(xs, 1)
3×2 Array{Int64,2}:
 1  2
 3  4
 5  6

julia> cat(xs, dims=1)
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]
```
"""
stack(xs, dim) = cat(unsqueeze.(xs, dim)..., dims=dim)

"""
    unstack(xs, dim)

Unroll the given `xs` into an `Array` of `Array`s along the given dimension `dim`.

# Examples
```jldoctest
julia> Flux.unstack([1 3 5 7; 2 4 6 8], 2)
4-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
```
"""
unstack(xs, dim) = [copy(selectdim(xs, dim, i)) for i in 1:size(xs, dim)]

"""
    chunk(xs, n)

Split `xs` into `n` parts.

# Examples
```jldoctest
julia> Flux.chunk(1:10, 3)
3-element Array{UnitRange{Int64},1}:
 1:4
 5:8
 9:10

julia> Flux.chunk(collect(1:10), 3)
3-element Array{SubArray{Int64,1,Array{Int64,1},Tuple{UnitRange{Int64}},true},1}:
 [1, 2, 3, 4]
 [5, 6, 7, 8]
 [9, 10]
```
"""
chunk(xs, n) = collect(Iterators.partition(xs, ceil(Int, length(xs)/n)))

batchindex(xs, i) = (reverse(Base.tail(reverse(axes(xs))))..., i)

"""
    frequencies(xs)

Count the number of times that each element of `xs` appears.

# Examples
```jldoctest
julia> Flux.frequencies(['a','b','b'])
Dict{Char,Int64} with 2 entries:
  'a' => 1
  'b' => 2
```
"""
function frequencies(xs)
  fs = Dict{eltype(xs),Int}()
  for x in xs
    fs[x] = get(fs, x, 0) + 1
  end
  return fs
end

head(x::Tuple) = reverse(Base.tail(reverse(x)))

squeezebatch(x) = reshape(x, head(size(x)))

"""
    batch(xs)

Batch the arrays in `xs` into a single array.

# Examples
```jldoctest
julia> Flux.batch([[1,2,3],[4,5,6]])
3×2 Array{Int64,2}:
 1  4
 2  5
 3  6
```
"""
function batch(xs)
  data = first(xs) isa AbstractArray ?
    similar(first(xs), size(first(xs))..., length(xs)) :
    Vector{eltype(xs)}(undef, length(xs))
  for (i, x) in enumerate(xs)
    data[batchindex(data, i)...] = x
  end
  return data
end

"""
Return the given sequence padded with `p` up to a maximum length of `n`.

# Examples
```jldoctest
julia> rpad([1, 2], 4, 0)
4-element Array{Int64,1}:
 1
 2
 0
 0

julia> rpad([1, 2, 3], 2, 0)
3-element Array{Int64,1}:
 1
 2
 3
```
"""
Base.rpad(v::AbstractVector, n::Integer, p) = [v; fill(p, max(n - length(v), 0))]

"""
    batchseq(seqs, pad)

Take a list of `N` sequences, and turn them into a single sequence where each
item is a batch of `N`. Short sequences will be padded by `pad`.

# Examples
```jldoctest
julia> Flux.batchseq([[1, 2, 3], [4, 5]], 0)
3-element Array{Array{Int64,1},1}:
 [1, 4]
 [2, 5]
 [3, 0]
```
"""
function batchseq(xs, pad = nothing, n = maximum(length(x) for x in xs))
  xs_ = [rpad(x, n, pad) for x in xs]
  [batch([xs_[j][i] for j = 1:length(xs_)]) for i = 1:n]
end

# Flattening models to weight vectors, and back

function _restructure(m, xs)
  i = 0
  fmap(m) do x
    x isa AbstractArray || return x
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end

@adjoint function _restructure(m, xs)
  _restructure(m, xs), dm -> (nothing,destructure(dm)[1])
end

"""
    destructure(m)

Flatten a model's parameters into a single weight vector.

    julia> m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)

    julia> θ, re = destructure(m);

    julia> θ
    67-element Array{Float32,1}:
    -0.1407104
    ...

The second return value `re` allows you to reconstruct the original network after making
modifications to the weight vector (for example, with a hypernetwork).

    julia> re(θ .* 2)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
"""
function destructure(m)
  xs = Zygote.Buffer([])
  fmap(m) do x
    x isa AbstractArray && push!(xs, x)
    return x
  end
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p)
end

# Other

"""
    throttle(f, timeout; leading=true, trailing=false)

Return a function that when invoked, will only be triggered at most once
during `timeout` seconds.

Normally, the throttled function will run as much as it can, without ever
going more than once per `wait` duration; but if you'd like to disable the
execution on the leading edge, pass `leading=false`. To enable execution on
the trailing edge, pass `trailing=true`.
"""
function throttle(f, timeout; leading=true, trailing=false)
  cooldown = true
  later = nothing
  result = nothing

  function throttled(args...; kwargs...)
    yield()

    if cooldown
      if leading
        result = f(args...; kwargs...)
      else
        later = () -> f(args...; kwargs...)
      end

      cooldown = false
      @async try
        while (sleep(timeout); later != nothing)
          later()
          later = nothing
        end
      finally
        cooldown = true
      end
    elseif trailing
      later = () -> (result = f(args...; kwargs...))
    end

    return result
  end
end

"""
    @jit ...

The `@jit` annotation can be applied to any code, and the code will be compiled
for performance.

    @jit f(x) = @jit(x) + @jit(x)

Note that compilation happens regardless of the `@jit` macro, so it should only
be used for aesthetic purposes, or by recovering Python users.
"""
macro jit(ex)
  esc(ex)
end
