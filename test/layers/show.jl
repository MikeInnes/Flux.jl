
@testset "layer printing" begin # 2-arg show, defined with layes

  @test repr(Dense(2,3)) == "Dense(2 => 3)"
  @test repr(Chain(Dense(2,3))) == "Chain(Dense(2 => 3))"

end
@testset "nested model printing" begin # 3-arg show, defined in show.jl

  # Dense -- has parameter count, but not when inside a matrix:

  toplevel_dense = repr("text/plain", Dense(2,3))
  @test occursin("Dense(2 => 3)", toplevel_dense)
  @test occursin("# 9 parameters", toplevel_dense)

  @test Meta.isexpr(Meta.parse(toplevel_dense), :call)  # comment is ignored

  vector_dense = repr("text/plain", [Dense(2,3), Dense(2,3)])
  @test occursin("Dense(2 => 3)", vector_dense)
  @test occursin("# 9 parameters", vector_dense)

  matrix_dense = repr("text/plain", fill(Dense(2,3), 3, 3))
  @test occursin("Dense(2 => 3)", matrix_dense)
  @test !occursin("# 9 parameters", matrix_dense)

  tuple_dense = repr("text/plain", tuple(Dense(2,3)))
  @test occursin("Dense(2 => 3)", tuple_dense)
  @test !occursin("# 9 parameters", tuple_dense)

  # Chain -- gets split over lines at top level only

  toplevel_chain = repr("text/plain", Chain(Dense(2,3)))
  @test occursin("Chain(\n  Dense(2 => 3)", toplevel_chain)
  @test occursin("# 9 parameters", toplevel_chain)
  @test !occursin("# Total:", toplevel_chain)

  vector_chain = repr("text/plain", [Chain(Dense(2,3)), Chain(Dense(2,3))])
  @test occursin("Chain(Dense(2 => 3))", vector_chain)
  @test occursin("# 9 parameters", vector_chain)
  @test !occursin("# Total:", vector_chain)

  matrix_chain = repr("text/plain", fill(Chain(Dense(2,3)), 3,3))
  @test occursin("Chain(Dense(2 => 3))", matrix_chain)
  @test !occursin("# 9 parameters", matrix_chain)
  @test !occursin("# Total:", matrix_chain)

  # ... and only long enough chains get a total at the end:

  longchain = Chain(Dense(2 => 3), Dense(3 => 4), Dense(4 => 5), softmax)

  toplevel_longchain = repr("text/plain", longchain)
  @test occursin("Chain(\n  Dense(2 => 3)", toplevel_longchain)
  @test occursin("# 9 parameters", toplevel_longchain)
  @test occursin("# Total: 6 arrays, 50 parameters", toplevel_longchain)

  vector_longchain = repr("text/plain", [longchain, longchain]) # pretty ugly in reality
  @test occursin("Chain(Dense(2 => 3)", vector_longchain)
  @test occursin("# 50 parameters", vector_longchain)
  @test !occursin("# 9 parameters", vector_longchain)
  @test !occursin("# Total:", vector_longchain)

  matrix_longchain = repr("text/plain", fill(longchain, 3,3))
  @test occursin("Chain(Dense(2 => 3)", matrix_longchain)
  @test !occursin("# 9 parameters", matrix_longchain)
  @test !occursin("# Total:", matrix_longchain)

  @test Meta.isexpr(Meta.parse(toplevel_longchain), :call)  # comments are ignored
  @test Meta.parse(toplevel_longchain).args[1] == :Chain

end
