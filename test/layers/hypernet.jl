using Flux
using Flux: destructure,restructure
using Test


@testset "Correct Dimensions for HyperDense" begin
    m = Chain(Dense(4,5),Dense(5,2))
    ps, re = destructure(m)
    hyper = Chain(Dense(1,10,tanh),HyperDense(10,m))
    @test length(destructure(hyper([1]))[1]) == length(ps)
    hyper = Chain(Dense(1,10,tanh),HyperDense(10,m,tanh))
    @test length(destructure(hyper([1]))[1]) == length(ps)
end

