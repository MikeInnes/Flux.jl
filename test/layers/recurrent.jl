using Flux, Test
using Flux: states, loadstates!

@testset "States" begin
  m1 = LSTM(10, 10)
  m2 = LSTM(10, 10)
  @test states(m1) == [m1.state...]
  loadstates!(m2, states(m1))
  @test states(m2) == states(m1)
  m1 = GRU(10, 10)
  m2 = GRU(10, 10)
  @test states(m1) == [m1.state]
  loadstates!(m2, states(m1))
  @test states(m2) == states(m1)
end
