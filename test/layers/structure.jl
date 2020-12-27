using Test, Random
import Flux

@testset "structure" begin
  
  @testset "Join" begin
    @testset "model_creation" begin
      @test_nowarn Chain(Join(Dense(10,6),Dense(10,5)), Dense(11,1))
      #@test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
    end
    @testset "model_training" begin
      model = Chain(
        Join(
          Chain(
            Dense(1, 5),
            Dense(5, 1)
          ),
          Dense(1, 2),
          Dense(1, 1),
        ),
        Dense(4, 1)
      )
      opt = Descent(0.1)
      loss(x, y) = Flux.mse(model(x), y)
      ps = Flux.params(model)
      x = map( x -> ( rand(1),rand(1),rand(1) ), 1:10 )
      y = map( x -> ( rand(1) ), 1:10)
      dat = zip(x, y)
      before = loss(x[1], y[1])
      map(x -> Flux.train!(loss, ps, dat, opt), 1:3)
      after = loss(x[1], y[1])
      @test after < before # loss should decrease with training
    end
  end
  
  @testset "Split" begin
    @testset "model_creation" begin
      @test_nowarn Chain(Dense(1, 1),Split(Dense(1, 1),Dense(1, 1),Dense(1, 1)))
      #@test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
    end
    @testset "model_training" begin
      model = Chain(
        Dense(1, 1),
        Split(
          Dense(1, 1),
          Dense(1, 1),
          Dense(1, 1)
        )
      )
      opt = Descent(0.1)
      using Statistics
      function loss(x, y)
        sqrt(mean([Flux.mse(model(x)[i], y[i]) for i in 1:length(y)].^2.)) # rms over each mse
      end
      ps = Flux.params(model)
      x = map( x -> ( rand(1) ), 1:10)
      y = map( x -> ( rand(1),rand(1),rand(1) ), 1:10 )
      dat = zip(x, y)
      before = loss(x[1], y[1])
      map(x -> Flux.train!(loss, ps, dat, opt), 1:3)
      after = loss(x[1], y[1])
      @test after < before # loss should decrease with training
    end
  end
  
  @testset "Parallel" begin
    @testset "model_creation" begin
      @test_nowarn Chain(Dense(1, 1),Parallel(Dense(1, 1),Dense(1, 3),Chain(Dense(1, 5),Dense(5, 2),)),Dense(6, 1))
      #@test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
    end
    @testset "model_training" begin
      model = Chain(
        Dense(1, 1),
        Parallel(
          Dense(1, 1),
          Dense(1, 3),
          Chain(
            Dense(1, 5),
            Dense(5, 2),
          )
        ),
        Dense(6, 1)
      ) 
      opt = Descent(0.1)
      loss(x, y) = Flux.mse(model(x), y)
      ps = Flux.params(model)
      x = map( x -> ( rand(1) ), 1:10)
      y = map( x -> ( rand(1) ), 1:10 )
      dat = zip(x, y)
      before = loss(x[1], y[1])
      map(x -> Flux.train!(loss, ps, dat, opt), 1:3)
      after = loss(x[1], y[1])
      @test after < before # loss should decrease with training
    end
  end
  
  @testset "Nop" begin
    @testset "model_creation" begin
      @test_nowarn Chain(Dense(1, 10),Dense(10, 1),Flux.Split(Nop,Nop))
    end
    @testset "nop_test_in_model" begin
      model = Chain(Dense(1, 10),Dense(10, 1),Flux.Split(Nop,Nop))
      result = model(rand(1))
      @test result[1] == result[2]
    end
  end
  
end
