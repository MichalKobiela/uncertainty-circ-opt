include("setup.jl")
#Tracl progress
callback = function (state, l)
    global loss_array[iter] = l
    global iter = iter + 1

    display(l)
end
global loss_array = zeros(20005)
global iter = 1



result = []
Random.seed!(0)
lk = ReentrantLock()
adtype = Optimization.AutoZygote()
posterior_sample = particles_arr[1,:]
optf = Optimization.OptimizationFunction((x, p) -> loss(posterior_sample,x))
optprob = Optimization.OptimizationProblem(optf, design, lb = [0.1, 0.1], ub = [1000.0, 1000.0])

using Main.Threads
@time @threads for i=1:5
    Random.seed!(i+100)
    println(i)
    postetrior_sample = particles_arr[i,:]
    optf = Optimization.OptimizationFunction((x, p) -> loss(postetrior_sample,x))
    optprob = Optimization.OptimizationProblem(optf, design, lb = [0.1, 0.1], ub = [1000.0, 1000.0])
    
    result_ode_1 = Optimization.solve(optprob, BBO_separable_nes()    ,
        maxiters = 100)
    p_opt = result_ode_1.u
    result_ode_2 = Optimization.solve(optprob, BBO_separable_nes()    ,
        maxiters = 100)

    if loss(postetrior_sample, p_opt) < loss(postetrior_sample, result_ode_2.u)
        p_opt = result_ode_2.u
    end
    
    result_ode_3 = Optimization.solve(optprob, BBO_separable_nes()    ,
        maxiters = 100)

    if loss(postetrior_sample, p_opt) < loss(postetrior_sample, result_ode_3.u)
        p_opt = result_ode_3.u
    end
    
    lock(lk) do
        push!(result, p_opt)
    end
end

result_arr = hcat(result...)
CSV.write(string(@__DIR__)*"/thompson.csv",  Tables.table(result_arr'), writeheader=false)
