using Test
using Flux
using Flux: onehot, onehotbatch

alphabet = map(c -> convert(Char, c), 0:255)
data = onehotbatch("This are forty-two onehotbatch characters.", alphabet)

@testset "LSTM" begin
    m = Chain(
        LSTM(256,10),
        LSTM(10,256))
    
    @test size(m(data)) == (256, 42)
end

@testset "peephole LSTM" begin
    m = Chain(
        PLSTM(256,10),
        PLSTM(10,256))

    @test size(m(data)) == (256, 42)
end

@testset "fully connected LSTM" begin
    m = Chain(
        FCLSTM(256,10),
        FCLSTM(10,256))

    @test size(m(data)) == (256, 42)
end
