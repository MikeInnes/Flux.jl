using Flux

# Run the tests
testdir = joinpath(dirname(dirname(@__DIR__)), "test")
include("$testdir/runtests.jl")
