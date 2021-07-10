"""
    flatten(x::AbstractArray)

Reshape arbitrarly-shaped input into a matrix-shaped output,
preserving the size of the last dimension.

See also [`unsqueeze`](@ref).

# Examples
```jldoctest
julia> rand(3,4,5) |> Flux.flatten |> size
(12, 5)

julia> xs = rand(Float32, 10,10,3,7);

julia> m = Chain(Conv((3,3), 3 => 4, pad=1), Flux.flatten, Dense(400 => 33));

julia> xs |> m[1] |> size
(10, 10, 4, 7)

julia> xs |> m |> size
(33, 7)
```
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end

"""
    normalise(x; dims=ndims(x), ϵ=1e-5)

Normalise `x` to mean 0 and standard deviation 1 across the dimension(s) given by `dims`.
Per default, `dims` is the last dimension. 
`ϵ` is a small additive factor added to the denominator for numerical stability.
"""
function normalise(x::AbstractArray; dims=ndims(x), ϵ=ofeltype(x, 1e-5))
  μ = mean(x, dims=dims)
    #   σ = std(x, dims=dims, mean=μ, corrected=false) # use this when Zygote#478 gets merged
  σ = std(x, dims=dims, corrected=false)
  return (x .- μ) ./ (σ .+ ϵ)
end
