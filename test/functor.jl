@testset "use params in gradient context" begin
    m = Chain(Dense(3,2), Dense(2,2))
    ps = Flux.params(m)
    gs = gradient(() -> sum(sum(p) for p in Flux.params(m)), ps)
    for p in ps
        @test gs[p] ≈ ones(size(p))
    end    

    w1, w2 =  rand(2), rand(2)
    ps = Flux.params(w1, w2)
    gs = gradient(() -> sum(sum(p) for p in Flux.params(w1, w2)), ps)
    for p in ps
        @test gs[p] ≈ ones(size(p))
    end

    # BROKEN TESTS
    m = Chain(Dense(3,2), Dense(2,2))
    @test_broken gradient(m -> sum(params(m)[1]), m) != (nothing, )
    @test_broken gradient(m -> sum(params(m)[1]), m) != (nothing, )

    gs = gradient(() -> sum(params(m)[1]), params(m))
    @test_broken gs[params(m)[1]] !== nothing
end
