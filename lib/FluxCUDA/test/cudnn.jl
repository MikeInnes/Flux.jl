@testset "CUDNN BatchNorm" begin
    @testset "4D Input" begin
        x = rand(Float32, 2, 2, 3, 4)
        m = BatchNorm(3)
        gx = gpu(x)
        gm = gpu(m)

        y, back = pullback((m, x) -> m(x), m, x)
        gy, gback = pullback((m, x) -> m(x), gm, gx)

        @test cpu(gy) ≈ y

        Δ = randn(Float32, size(y))
        dm, dx = back(Δ)
        gdm, gdx = gback(gpu(Δ))

        @test dm[].γ ≈ cpu(gdm[].γ)
        @test dm[].β ≈ cpu(gdm[].β)
        @test dx ≈ cpu(gdx)
    end

    @testset "2D Input" begin
        x = rand(Float32, 3, 4)
        m = BatchNorm(3)
        gx = gpu(x)
        gm = gpu(m)

        y, back = pullback((m, x) -> m(x), m, x)
        gy, gback = pullback((m, x) -> m(x), gm, gx)

        @test cpu(gy) ≈ y

        Δ = randn(Float32, size(y))
        dm, dx = back(Δ)
        gdm, gdx = gback(gpu(Δ))

        @test dm[].γ ≈ cpu(gdm[].γ)
        @test dm[].β ≈ cpu(gdm[].β)
        @test dx ≈ cpu(gdx)
    end
end
