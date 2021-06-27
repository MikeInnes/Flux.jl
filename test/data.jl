using Random

@testset "DataLoader" begin
    X = reshape([1:10;], (2, 5))
    Y = [1:5;]

    d = DataLoader((X,), batchsize=2)
    @inferred first(d)
    batches = collect(d)
    # @test eltype(batches) == eltype(d) == typeof(X)
    @test length(batches) == 3
    @test batches[1] == X[:,1:2]
    @test batches[2] == X[:,3:4]
    @test batches[3] == X[:,5:5]

    d = DataLoader((X,), batchsize=2, partial=false)
    @inferred first(d)
    batches = collect(d)
    # @test eltype(batches) == eltype(d) == typeof(X)
    @test length(batches) == 2
    @test batches[1] == X[:,1:2]
    @test batches[2] == X[:,3:4]

    d = DataLoader((X,), batchsize=2, partial=false)
    @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == Tuple{typeof(X)}
    @test length(batches) == 2
    @test batches[1] == (X[:,1:2],)
    @test batches[2] == (X[:,3:4],)

    d = DataLoader((X, Y), batchsize=2)
    @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == Tuple{typeof(X), typeof(Y)}
    @test length(batches) == 3
    @test length(batches[1]) == 2
    @test length(batches[2]) == 2
    @test length(batches[3]) == 2
    @test batches[1][1] == X[:,1:2]
    @test batches[1][2] == Y[1:2]
    @test batches[2][1] == X[:,3:4]
    @test batches[2][2] == Y[3:4]
    @test batches[3][1] == X[:,5:5]
    @test batches[3][2] == Y[5:5]

    # test with NamedTuple
    # d = DataLoader((x=X, y=Y), batchsize=2)
    # @inferred first(d)
    # batches = collect(d)
    # @test eltype(batches) == eltype(d) == NamedTuple{(:x, :y), Tuple{typeof(X), typeof(Y)}}
    # @test length(batches) == 3
    # @test length(batches[1]) == 2
    # @test length(batches[2]) == 2
    # @test length(batches[3]) == 2
    # @test batches[1][1] == batches[1].x == X[:,1:2]
    # @test batches[1][2] == batches[1].y == Y[1:2]
    # @test batches[2][1] == batches[2].x == X[:,3:4]
    # @test batches[2][2] == batches[2].y == Y[3:4]
    # @test batches[3][1] == batches[3].x == X[:,5:5]
    # @test batches[3][2] == batches[3].y == Y[5:5]

    # test interaction with `train!`
    θ = ones(2)
    X = zeros(2, 10)
    loss(x) = sum((x .- θ).^2)
    d  = DataLoader(X)
    Flux.train!(loss, [θ], ncycle(d, 10), Descent(0.1))
    @test norm(θ) < 1e-4

    # test interaction with `train!`
    θ = zeros(2)
    X = ones(2, 10)
    Y = fill(2, 10)
    loss(x, y) = sum((y - x'*θ).^2)
    d  = DataLoader((X, Y))
    Flux.train!(loss, [θ], ncycle(d, 10), Descent(0.1))
    @test norm(θ .- 1) < 1e-10

    # specify the rng
    d = map(identity, DataLoader(X, batchsize=2; shuffle = x -> Random.shuffle!(Random.default_rng(), x)))
end
