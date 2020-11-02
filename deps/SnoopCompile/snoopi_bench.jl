using SnoopCompile

botconfig = BotConfig(
  "Flux";
  os = ["linux", "windows", "macos"],
  version = [v"1.4.1", v"1.3.1"],
  exhaustive = false,
)


println("Benchmarking the inference time of `using Flux`")
snoopi_bench(
  botconfig,
  :(using Flux),
)


println("Benchmarking the inference time of the tests")
snoopi_bench(
  botconfig,
  "$(@__DIR__)/example_script.jl",
)
