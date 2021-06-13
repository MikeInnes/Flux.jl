using Flux
using Flux.Data
using Test
using Random, Statistics, LinearAlgebra
using IterTools: ncycle

Random.seed!(0)

@testset "Utils" begin
  include("utils.jl")
end

@testset "Onehot" begin
  include("onehot.jl")
end

@testset "Optimise" begin
  include("optimise.jl")
end

@testset "Data" begin
  include("data.jl")
end

@testset "Losses" begin
  include("losses.jl")
  include("ctc.jl")
  if Flux.default_gpu_converter[] !== identity
    include("ctc-gpu.jl")
  end
end

@testset "Layers" begin
  include("layers/basic.jl")
  include("layers/normalisation.jl")
  include("layers/stateless.jl")
  include("layers/recurrent.jl")
  include("layers/conv.jl")
  include("layers/upsample.jl")
end

@testset "outputsize" begin
  using Flux: outputsize
  include("outputsize.jl")
end

@static if VERSION == v"1.5"
  using Documenter
  @testset "Docs" begin
    DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
    doctest(Flux)
  end
end
