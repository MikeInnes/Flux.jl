using Flux:onecold
using Test

@testset "onecold" begin
  a = [1, 2, 5, 3.]
  A = [1 20 5; 2 7 6; 3 9 10; 2 1 14]
  labels = ['A', 'B', 'C', 'D']

  @test onecold(a) == 3
  @test onecold(A) == [3, 1, 4]
  @test onecold(a, labels) == 'C'
  @test onecold(A, labels) == ['C', 'A', 'D']

  data = [:b, :a, :c]
  labels = [:a, :b, :c]
  hot = Flux.onehotbatch(data, labels)
  cold = onecold(hot, labels)

  @test cold == data
end

@testset "onehotbatch indexing" begin
  y = Flux.onehotbatch(ones(3), 1:10)
  @test y[:,1] isa Flux.OneHotVector
  @test y[:,:] isa Flux.OneHotMatrix
end

@testset "abstractmatrix onehotvector multiplication" begin
  A = [1 3 5; 2 4 6; 3 6 9]
  b1 = Flux.OneHotVector(1, 3)
  b2 = Flux.OneHotVector(3, 5)

  @test A*b1 == A[:,1]
  @test_throws DimensionMismatch A*b2
end

@testset "OneHotArray" begin
  using Flux: OneHotArray, OneHotVector, OneHotMatrix, OneHotLike
  
  ov = OneHotVector(rand(1:10), 10)
  ov2 = OneHotVector(rand(1:11), 11)
  om = OneHotMatrix(rand(1:10, 5), 10)
  om2 = OneHotMatrix(rand(1:11, 5), 11)
  oa = OneHotArray(rand(1:10, 5, 5), 10)

  # sizes
  @testset "Base.size" begin
    @test size(ov) == (10,)
    @test size(om) == (10, 5)
    @test size(oa) == (10, 5, 5)
  end

  @testset "Indexing" begin
    # vector indexing
    @test ov[3] == (ov.indices == 3)
    @test ov[:] == ov
    
    # matrix indexing
    @test om[3, 3] == (om.indices[3] == 3)
    @test om[:, 3] == OneHotVector(om.indices[3], 10)
    @test om[3, :] == (om.indices .== 3)
    @test om[:, :] == om

    # array indexing
    @test oa[3, 3, 3] == (oa.indices[3, 3] == 3)
    @test oa[:, 3, 3] == OneHotVector(oa.indices[3, 3], 10)
    @test oa[3, :, 3] == (oa.indices[:, 3] .== 3)
    @test oa[3, :, :] == (oa.indices .== 3)
    @test oa[:, 3, :] == OneHotMatrix(oa.indices[3, :], 10)
    @test oa[:, :, :] == oa

    # cartesian indexing
    @test oa[CartesianIndex(3, 3, 3)] == oa[3, 3, 3]
  end

  @testset "Concatenating" begin
    # vector cat
    @test hcat(ov, ov) == OneHotMatrix(vcat(ov.indices, ov.indices), 10)
    @test hcat(ov, ov) isa OneHotMatrix
    @test vcat(ov, ov) == vcat(collect(ov), collect(ov))
    @test cat(ov, ov; dims = 3) == OneHotArray(cat(ov.indices, ov.indices; dims = 2), 10)

    # matrix cat
    @test hcat(om, om) == OneHotMatrix(vcat(om.indices, om.indices), 10)
    @test hcat(om, om) isa OneHotMatrix
    @test vcat(om, om) == vcat(collect(om), collect(om))
    @test cat(om, om; dims = 3) == OneHotArray(cat(om.indices, om.indices; dims = 2), 10)

    # array cat
    @test cat(oa, oa; dims = 3) == OneHotArray(cat(oa.indices, oa.indices; dims = 2), 10)
    @test cat(oa, oa; dims = 3) isa OneHotArray
    @test cat(oa, oa; dims = 1) == cat(collect(oa), collect(oa); dims = 1)

    # proper error handling of inconsistent sizes
    @test_throws DimensionMismatch hcat(ov, ov2)
    @test_throws DimensionMismatch hcat(om, om2)
  end

  @testset "Base.reshape" begin
    # reshape test
    @test reshape(oa, 10, 25) isa OneHotLike
    @test reshape(oa, 10, :) isa OneHotLike
    @test reshape(oa, :, 25) isa OneHotLike
    @test reshape(oa, 50, :) isa OneHotLike
    @test reshape(oa, 5, 10, 5) isa OneHotLike
    @test reshape(oa, (10, 25)) isa OneHotLike

    @testset "w/ cat" begin
      r = reshape(oa, 10, :)
      @test hcat(r, r) isa OneHotArray
      @test vcat(r, r) isa Array{Bool}
    end

    @testset "w/ argmax" begin
      r = reshape(oa, 10, :)
      @test argmax(r) == argmax(OneHotMatrix(reshape(oa.indices, :), 10))
      @test Flux.onecold(r) == collect(reshape(oa.indices, :))
    end
  end

  @testset "Base.argmax" begin
    # argmax test
    @test argmax(ov) == argmax(convert(Array{Bool}, ov))
    @test argmax(om) == argmax(convert(Array{Bool}, om))
    @test argmax(om; dims = 1) == argmax(convert(Array{Bool}, om); dims = 1)
    @test argmax(om; dims = 2) == argmax(convert(Array{Bool}, om); dims = 2)
    @test argmax(oa; dims = 1) == argmax(convert(Array{Bool}, oa); dims = 1)
    @test argmax(oa; dims = 3) == argmax(convert(Array{Bool}, oa); dims = 3)
  end
end