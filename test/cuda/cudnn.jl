using Flux, CuArrays, Test
using Flux: pullback

@testset "CUDNN BatchNorm" begin
    @testset "4D Input" begin
        x = Float64.(collect(reshape(1:12, 2, 2, 3, 1)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y, back = pullback((m, x) -> m(x), m, x)
        cy, cback = pullback((m, x) -> m(x), cm, cx)

        @test cpu(cy) ≈ y

        Δ = randn(size(y))
        dm, dx = back(Δ)
        cdm, cdx = cback(gpu(Δ))

        @test dm[].γ ≈ cpu(cdm[].γ)
        @test dm[].β ≈ cpu(cdm[].β)
        @test dx ≈ cpu(cdx)
    end

    @testset "2D Input" begin
        x = Float64.(collect(reshape(1:12, 3, 4)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y, back = pullback((m, x) -> m(x), m, x)
        cy, cback = pullback((m, x) -> m(x), cm, cx)

        @test cpu(cy) ≈ y

        Δ = randn(size(y))
        dm, dx = back(Δ)
        cdm, cdx = cback(gpu(Δ))

        @test dm[].γ ≈ cpu(cdm[].γ)
        @test dm[].β ≈ cpu(cdm[].β)
        @test dx ≈ cpu(cdx)
    end
end

@testset "CUDNN same padding $layer" for layer in (Conv, ConvTranspose, CrossCor, MeanPool, MaxPool)
  for k in ((1, 1), (2, 1), (3, 1), (4, 5), (6, 7, 8))
    data = ones(Float32, (k .+ 8)..., 1, 1) |> gpu

    if layer in (MeanPool, MaxPool)
      l = layer(k, pad = "same") |> gpu
      @test size(l(data))[1:end-2] == ceil.(Int, size(data)[1:end-2] ./ k)
    else
      l = layer(k, 1 => 1, pad = "same") |> gpu
      @test size(l(data)) == size(data)
  
      l = layer(k, 1 => 1, pad = "same", dilation = max(1, k .÷ 2)) |> gpu
      @test size(l(data)) == size(data)
  
      stride = 3
      l = layer(k, 1 => 1, pad = "same", stride = stride) |> gpu
      if layer == ConvTranspose
        @test size(l(data))[1:end-2] == stride .* size(data)[1:end-2] .- stride .+ 1
      else
        @test size(l(data))[1:end-2] == ceil.(Int, size(data)[1:end-2] ./ stride)
      end
    end
  end
end
