function check_grad(g_gpu, g_cpu, atol, rtol)
  @show g_gpu g_cpu
  @test false
end
check_grad(g_gpu::Base.RefValue, g_cpu::Base.RefValue, atol, rtol) = 
    check_grad(g_gpu[], g_cpu[], atol, rtol)
check_grad(g_gpu::Nothing, g_cpu::Nothing, atol, rtol) = @test true
check_grad(g_gpu::Float32, g_cpu::Float32, atol, rtol) = @test g_cpu ≈ g_gpu   rtol=rtol atol=atol 
check_grad(g_gpu::CuArray{Float32}, g_cpu::Array{Float32}, atol, rtol) = 
  @test g_cpu ≈ collect(g_gpu)   rtol=rtol atol=atol

function check_grad(g_gpu::Tuple, g_cpu::Tuple, atol, rtol)
  for (v1, v2) in zip(g_gpu, g_cpu)  
    check_grad(v1, v2, atol, rtol)
  end
end

function check_grad(g_gpu::NamedTuple, g_cpu::NamedTuple, atol, rtol)
  for ((k1,v1), (k2,v2)) in zip(pairs(g_gpu), pairs(g_cpu))  
    @test k1 == k2
    # @show k2 v2
    check_grad(v1, v2, atol, rtol)
  end
end

function gpu_autodiff_test(f_cpu, xs_cpu::Array{Float32}...; 
        test_equal=true, rtol=1e-4, atol=1e-4)

  check_type(x) = false
  check_type(x::Float32) = true
  check_type(x::CuArray{Float32}) = true
  check_type(x::Array{Float32}) = true

  ### GRADIENT WITH RESPECT TO INPUT #####
  # y_cpu, back_cpu = pullback((f, x...) -> f(x...), f_cpu, xs_cpu...)
  y_cpu, back_cpu = pullback((x...) -> f_cpu(x...), xs_cpu...)
  @test check_type(y_cpu)
  Δ_cpu = size(y_cpu) == () ? randn(Float32) : randn(Float32, size(y_cpu))
  gs_cpu = back_cpu(Δ_cpu)

  f_gpu = f_cpu |> gpu
  xs_gpu = gpu.(xs_cpu)
  Δ_gpu = Δ_cpu |> gpu 
  # y_gpu, back_gpu = pullback((f, x...) -> f(x...), f_gpu, xs_gpu...)
  y_gpu, back_gpu = pullback((x...) -> f_gpu(x...), xs_gpu...)
  @test check_type(y_gpu)
  gs_gpu = back_gpu(Δ_gpu)

  if test_equal 
    @test collect(y_cpu) ≈ collect(y_gpu)   rtol=rtol atol=atol
    for (g_gpu, g_cpu) in zip(gs_gpu, gs_cpu)
      check_grad(g_gpu, g_cpu, atol, rtol)
    end
  end

  ### GRADIENT WITH RESPECT TO f #####
  ps_cpu = Flux.params(f_cpu)
  y_cpu, back_cpu = pullback(() -> f_cpu(xs_cpu...), ps_cpu)
  gs_cpu = back_cpu(Δ_cpu)

  ps_gpu = Flux.params(f_gpu)
  y_gpu, back_gpu = pullback(() -> f_gpu(xs_gpu...), ps_gpu)
  gs_gpu = back_gpu(Δ_gpu)
  
  if test_equal 
    @test collect(y_cpu) ≈ collect(y_gpu)   rtol=rtol atol=atol
    @assert length(ps_gpu) == length(ps_cpu)
    for (p_gpu, p_cpu) in zip(ps_gpu, ps_cpu)
      check_grad(gs_gpu[p_gpu], gs_cpu[p_cpu], atol, rtol)
    end
  end
end
