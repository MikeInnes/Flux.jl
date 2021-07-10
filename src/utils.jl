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
julia> layer = Dense(10, 20);

julia> Flux.nfan(size(layer.W))
(10, 20)

julia> layer = Conv((3, 3), 2=>10);

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
2×3 Matrix{Float32}:
 0.601094  -0.57414   -0.814925
 0.900868   0.805994   0.057514
```

# See also

* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
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
3×2 Matrix{Float32}:
  0.429505  -0.0852891
  0.523935   0.371009
 -0.223261   0.188052
```

# See also

* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
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
3×2 Matrix{Float32}:
  0.950413   1.27439
  1.4244    -1.28851
 -0.907795   0.0909376
```

# See also

* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_uniform(rng::AbstractRNG, dims...; gain = √2)
  bound = Float32(√3 * gain / sqrt(first(nfan(dims...)))) # fan_in
  return (rand(rng, Float32, dims...) .- 0.5f0) .* 2bound
end

kaiming_uniform(dims...; kwargs...) = kaiming_uniform(Random.GLOBAL_RNG, dims...; kwargs...)
kaiming_uniform(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> kaiming_uniform(rng, dims...; init_kwargs..., kwargs...)

"""
    kaiming_normal([rng=GLOBAL_RNG], dims...; gain = √2)

Return an `Array` of size `dims` containing random variables taken from a normal
distribution with mean 0 and standard deviation `gain * sqrt(fan_in)`.

This method is described in [1] and also known as He initialization.

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.kaiming_normal(3, 2)
3×2 Matrix{Float32}:
  0.679107  -0.134854
  0.828413   0.586617
 -0.353007   0.297336
```

# See also

* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)
* calculation of `fan_in` and `fan_out`: [`nfan`](@ref Flux.nfan)

# References

[1] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." _Proceedings of the IEEE international conference on computer vision_. 2015.
"""
function kaiming_normal(rng::AbstractRNG, dims...; gain = √2f0)
  std = Float32(gain / sqrt(first(nfan(dims...)))) # fan_in
  return randn(rng, Float32, dims...) .* std
end

kaiming_normal(dims...; kwargs...) = kaiming_normal(Random.GLOBAL_RNG, dims...; kwargs...)
kaiming_normal(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> kaiming_normal(rng, dims...; init_kwargs..., kwargs...)

"""
    orthogonal([rng=GLOBAL_RNG], dims...; gain = 1)

Return an `Array` of size `dims` which is a (semi) orthogonal matrix, as described in [1].

The input must have at least 2 dimensions.
For `length(dims) > 2`, a `prod(dims[1:(end - 1)])` by `dims[end]` orthogonal matrix
is computed before reshaping it to the original dimensions.

# Examples
```jldoctest; setup = :(using LinearAlgebra)
julia> W = Flux.orthogonal(5, 7);

julia> summary(W)
"5×7 Matrix{Float32}"

julia> W * W' ≈ I(5)
true

julia> W2 = Flux.orthogonal(7, 5);

julia> W2 * W2' ≈ I(7)
false

julia> W2' * W2 ≈ I(5)
true

julia> W3 = Flux.orthogonal(3, 3, 2, 4);

julia> transpose(reshape(W3, :, 4)) * reshape(W3, :, 4) ≈ I(4)
true
```

# See also
* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)
* sparse initialization: [`sparse_init`](@ref Flux.sparse_init)

# References
[1] Saxe, McClelland, Ganguli. "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks", ICLR 2014, https://arxiv.org/abs/1312.6120

"""
function orthogonal(rng::AbstractRNG, rows::Integer, cols::Integer; gain = 1)
  mat = rows > cols ? randn(rng, Float32, rows, cols) : randn(rng, Float32, cols, rows)

  Q, R = LinearAlgebra.qr(mat)
  Q = Array(Q) * sign.(LinearAlgebra.Diagonal(R))
  if rows < cols
    Q = transpose(Q)
  end

  return gain * Q
end

function orthogonal(rng::AbstractRNG, d1::Integer, ds::Integer...; kwargs...)
  dims = (d1, ds...)
  rows = prod(dims[1:end-1])
  cols = dims[end]
  return reshape(orthogonal(rng, rows, cols; kwargs...), dims)
end

orthogonal(dims::Integer...; kwargs...) = orthogonal(Random.GLOBAL_RNG, dims...; kwargs...)
orthogonal(rng::AbstractRNG; init_kwargs...) = (dims::Integer...; kwargs...) -> orthogonal(rng, dims...; init_kwargs..., kwargs...)

"""
    sparse_init([rng=GLOBAL_RNG], dims...; sparsity, std = 0.01)

Return an `Array` of size `dims` where each column contains a fixed fraction of
zero elements given by `sparsity`. Non-zero elements are normally distributed
with a mean of zero and standard deviation `std`.

This method is described in [1].

# Examples
```jldoctest; setup = :(using Random; Random.seed!(0))
julia> Flux.sparse_init(3, 2, sparsity=0.1)
3×2 Matrix{Float32}:
  0.00828413  0.0
 -0.00353007  0.00297336
  0.0         0.00586617
```

# See also

* kaiming initialization using normal distribution: [`kaiming_normal`](@ref Flux.kaiming_normal)
* kaiming initialization using uniform distribution: [`kaiming_uniform`](@ref Flux.kaiming_uniform)
* glorot initialization using normal distribution: [`glorot_normal`](@ref Flux.glorot_normal)
* glorot initialization using uniform distribution: [`glorot_uniform`](@ref Flux.glorot_uniform)

# References

[1] Martens, J, "Deep learning via Hessian-free optimization" _Proceedings of the 27th International Conference on International Conference on Machine Learning_. 2010.
"""
function sparse_init(rng::AbstractRNG, dims...; sparsity, std = 0.01)
  if length(dims) != 2
    throw(ArgumentError("Only 2-dimensional outputs are supported for sparse initialization."))
  end
  rows, cols = dims
  prop_zero = min(1.0, sparsity)
  num_zeros = ceil(Integer, prop_zero * rows)
  sparse_array = randn(rng, Float32, dims...) .* Float32(std)
  sparse_array[1:num_zeros, :] .= 0f0
  return mapslices(shuffle, sparse_array, dims=1)
end

sparse_init(dims...; kwargs...) = sparse_init(Random.GLOBAL_RNG, dims...; kwargs...)
sparse_init(rng::AbstractRNG; init_kwargs...) = (dims...; kwargs...) -> sparse_init(rng, dims...; init_kwargs..., kwargs...)

"""
    identity_init([rng=GLOBAL_RNG], dims...; gain=1, shift=0)

Return an `Array` of size `dims` which yields an identity mapping when used as parameters in 
most Flux layers. Use `gain` to scale the identity by a constant.

Often useful in the context of transfer learning, i.e when one wants to add more capacity to
a model but start from the same mapping.

Use `shift` (integer or tuple) to apply circular shift to the output.
Equivalent to `Base.circshift(identity(dims...), shift)`.

Some caveats: Not all layers will be identity mapping when used with this init. Exceptions
include recurrent layers, `DepthwiseConv` and normalization layers.

Also note that layers must have `input_size == output_size` for identity mapping to be 
possible. When this is not the case, extra dimensions of the array are padded with zeros.

For convolutional layers, in addition to the above, the kernel sizes must also be odd and 
padding must be applied so that output feature maps have the same size as input feature maps,
e.g by using [`SamePad`](@ref).

Has the following behaviour
*  1D: A `Vector` of `zeros` (useful for an identity bias)
*  2D: An identity matrix (useful for an identity matrix multiplication)
*  More than 2D: A dense block array of center tap spatial filters (useful for an identity convolution)

```jldoctest
julia> Flux.identity_init(3,3)
3×3 Matrix{Float32}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0

julia> Flux.identity_init(3,5)
3×5 Matrix{Float32}:
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0

julia> Flux.identity_init(3,3,2,2)
3×3×2×2 Array{Float32,4}:
[:, :, 1, 1] =
 0.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.0

[:, :, 2, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 1, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2, 2] =
 0.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.0
```
"""
# Assume bias
identity_init(cols; gain=1, shift=0) = zeros(Float32, cols)

# Assume matrix multiplication
identity_init(rows, cols; gain=1, shift=0) = circshift(Matrix{Float32}(I * gain, rows,cols), shift)

# Assume convolution
function identity_init(dims...; gain=1, shift=0)
  nin, nout = dims[end-1], dims[end]
  centers = map(d -> cld(d, 2), dims[1:end-2])
  weights = zeros(Float32, dims)
  for i in 1:min(nin,nout)
    weights[centers..., i, i] = gain
  end
  return circshift(weights, shift)
end

identity_init(::AbstractRNG, dims...; kwargs...) = identity_init(dims...; kwargs...)
identity_init(; init_kwargs...) = identity_init(Random.GLOBAL_RNG; init_kwargs...)
identity_init(rng::AbstractRNG; init_kwargs...) = (args...;kwargs...) -> identity_init(rng, args...; init_kwargs..., kwargs...)


ones(T::Type, dims...) = Base.ones(T, dims...)
zeros(T::Type, dims...) = Base.zeros(T, dims...)

ones(dims...) = Base.ones(Float32, dims...)
zeros(dims...) = Base.zeros(Float32, dims...)

"""
    create_bias(weights, bias, length)

Return a bias parameter for a layer, based on the value given
to the constructor's keyword `bias=bias`.

* `bias == true` creates a zero vector, of the same type as weights.
* `bias == false` returns `Zeros()`, a special struct which exists only to encode the absence of bias.
* `bias::AbstractArray` uses the array provided, provided it has the correct size and eltype. If the type is wrong, it will be converted.
"""
function create_bias(weights::AbstractArray, bias::Bool, dims::Integer...)
  bias ? fill!(similar(weights, dims...), 0) : Zeros()
end
function create_bias(weights::AbstractArray, bias::AbstractArray, dims::Integer...)
  size(bias) == dims || throw(DimensionMismatch("expected bias of size $(dims), got size $(size(bias))"))
  bias
end

"""
    unsqueeze(xs, dim)

Return `xs` reshaped into an array one dimensionality higher than `xs`,
where `dim` indicates in which dimension `xs` is extended.

See also [`flatten`](@ref), [`stack`](@ref).

# Examples
```jldoctest
julia> Flux.unsqueeze([1 2; 3 4], 2)
2×1×2 Array{Int64, 3}:
[:, :, 1] =
 1
 3

[:, :, 2] =
 2
 4

julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> Flux.unsqueeze(xs, 1)
1×3 Matrix{Vector{Int64}}:
 [1, 2]  [3, 4]  [5, 6]
```
"""
unsqueeze(xs::AbstractArray, dim::Integer) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))

"""
    unsqueeze(dim)

Returns a function which, acting on an array, inserts a dimension of size 1 at `dim`.

# Examples
```jldoctest
julia> rand(21, 22, 23) |> Flux.unsqueeze(2) |> size
(21, 1, 22, 23)

julia> m = Chain(Flux.unsqueeze(3), Flux.unsqueeze(4), Conv((3,3), 1=>7, pad=SamePad()));

julia> rand(Float32, 10, 10) |> m |> size
(10, 10, 7, 1)
```
"""
unsqueeze(dim::Integer) = Base.Fix2(unsqueeze, dim)

Base.show_function(io::IO, u::Base.Fix2{typeof(unsqueeze)}, ::Bool) = print(io, "unsqueeze(", u.x, ")")

"""
    stack(xs, dim)

Concatenate the given `Array` of `Array`s `xs` into a single `Array` along the
given dimension `dim`.

# Examples
```jldoctest
julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> Flux.stack(xs, 1)
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6

julia> cat(xs, dims=1)
3-element Vector{Vector{Int64}}:
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
4-element Vector{Vector{Int64}}:
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
3-element Vector{UnitRange{Int64}}:
 1:4
 5:8
 9:10

julia> Flux.chunk(collect(1:10), 3)
3-element Vector{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}:
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
Dict{Char, Int64} with 2 entries:
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
3×2 Matrix{Int64}:
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
4-element Vector{Int64}:
 1
 2
 0
 0

julia> rpad([1, 2, 3], 2, 0)
3-element Vector{Int64}:
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
3-element Vector{Vector{Int64}}:
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
  m̄ = fmap(m) do x
    x isa AbstractArray || return x
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
  length(xs) == i || @warn "Expected $(i) params, got $(length(xs))"
  return m̄
end

@adjoint function _restructure(m, xs)
  m̄, numel = _restructure(m, xs), length(xs)
  function _restructure_pullback(dm)
    xs′ = destructure(dm)[1]
    numel == length(xs′) || @warn "Expected $(numel) params, got $(length(xs′))"
    return (nothing, xs′)
  end
  return m̄, _restructure_pullback
end

"""
    destructure(m)

Flatten a model's parameters into a single weight vector.

    julia> m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)

    julia> θ, re = destructure(m);

    julia> θ
    67-element Vector{Float32}:
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
    modules(m)

Return an iterator over non-leaf objects
that can be reached by recursing `m` over
the children given by [`functor`](@ref).

Useful for applying a function (e.g. a regularizer)
over specific modules or subsets of the parameters
(e.g. the weights but not the biases).

# Examples

```jldoctest
julia> m1 = Chain(Dense(28^2, 64), BatchNorm(64, relu));

julia> m2 = Chain(m1, Dense(64, 10))
Chain(
  Chain(
    Dense(784 => 64),                   # 50_240 parameters
    BatchNorm(64, relu),                # 128 parameters, plus 128
  ),
  Dense(64 => 10),                      # 650 parameters
)         # Total: 6 trainable arrays, 51_018 parameters,
          # plus 2 non-trainable, 128 parameters, summarysize 200.312 KiB.

julia> Flux.modules(m2)
5-element Vector{Any}:
 Chain(Chain(Dense(784 => 64), BatchNorm(64, relu)), Dense(64 => 10))  # 51_018 parameters, plus 128 non-trainable
 Chain(Dense(784 => 64), BatchNorm(64, relu))  # 50_368 parameters, plus 128 non-trainable
 Dense(784 => 64)    # 50_240 parameters
 BatchNorm(64, relu)  # 128 parameters, plus 128 non-trainable
 Dense(64 => 10)     # 650 parameters

julia> L2(m) = sum(sum(abs2, l.weight) for l in Flux.modules(m) if l isa Dense)
L2 (generic function with 1 method)
```
"""
modules(m) = [x for x in Functors.fcollect(m) if !isleaflike(x)]

@nograd modules

isleaflike(x) = Functors.isleaf(x)
isleaflike(::Tuple{Vararg{<:Number}}) = true
isleaflike(::Tuple{Vararg{<:AbstractArray{<:Number}}}) = true

"""
    patience(predicate, wait)

Return a function that internally counts by one when
`predicate(...) == true`, otherwise the count is reset to zero.
If the count is greater than or equal to `wait`,
the function returns `true`, otherwise it returns `false`.

# Examples
```jldoctest
julia> loss() = rand();

julia> trigger = Flux.patience(() -> loss() < 1, 3);


julia> Flux.@epochs 10 begin
         trigger() && break
       end
[ Info: Epoch 1
[ Info: Epoch 2
[ Info: Epoch 3
```
"""
function patience(predicate, wait)
  let count = 0
    function on_trigger(args...; kwargs...)
      count = predicate(args...; kwargs...) ? count + 1 : 0

      return count >= wait
    end
  end
end

"""
    early_stopping(f, delay; distance = -, init_score = 0, min_dist = 0)

Return a function that internally counts by one when
`distance(best_score, f(...)) <= min_dist`, where
`best_score` is the last seen best value of `f(...)`.
If the count is greater than or equal to `delay`,
the function returns `true`, otherwise it returns `false`.
The count is reset when `distance(best_score, f(...)) > min_dist`.

# Examples
```jldoctest
julia> loss = let l = 0
         () -> l += 1
       end; # pseudo loss function that returns increasing values

julia> es = Flux.early_stopping(loss, 3);


julia> Flux.@epochs 10 begin
         es() && break
       end
[ Info: Epoch 1
[ Info: Epoch 2
[ Info: Epoch 3
```
"""
function early_stopping(f, delay; distance = -, init_score = 0, min_dist = 0)
  trigger = let best_score = init_score
    (args...; kwargs...) -> begin
      score = f(args...; kwargs...)
      Δ = distance(best_score, score)
      best_score = Δ < 0 ? best_score : score

      return Δ < min_dist
    end
  end

  return patience(trigger, delay)
end

"""
    plateau(f, width; distance = -, init_score = 0, min_dist = 1f-6)

Return a function that internally counts by one when
`abs(distance(last_score, f(...))) <= min_dist`, where
`last_score` holds the last value of `f(...)`.
If the count is greater than or equal to `width`,
the function returns `true`, otherwise it returns `false`.
The count is reset when `abs(distance(last_score, f(...))) > min_dist`.

# Examples
```jldoctest
julia> f = let v = 10
         () -> v = v / abs(v) - v
       end; # -9, 8, -7, 6, ...

julia> trigger = Flux.plateau(f, 3; init_score=10, min_dist=18);


julia> Flux.@epochs 10 begin
         trigger() && break
       end
[ Info: Epoch 1
[ Info: Epoch 2
[ Info: Epoch 3
[ Info: Epoch 4
```
"""
function plateau(f, width; distance = -, init_score = 0, min_dist = 1f-6)
  is_plateau = let last_score = init_score
    (args...; kwargs...) -> begin
      score = f(args...; kwargs...)
      Δ = abs(distance(last_score, score))
      last_score = score

      return Δ < min_dist
    end
  end

  return patience(is_plateau, width)
end
