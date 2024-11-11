using RecursiveArrayTools, Distributions, Plots, StatsPlots
using Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using Zygote, Turing, StatsPlots, Random
using LinearAlgebra, Serialization, CSV, Tables, JLD2
include(string(@__DIR__)*"/../Models/rpa_ode.jl")

Random.seed!(0); 

# Use prior samples instaed of posterior samples
prior_samples_arr = rand(100,4)

# We reparemtrize Thompson samples to ensure they are within the desired range
sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))

upper_bound = 1000
num_optimized_vars = 5

function reparm(x)
    return sigmoid.(x) .* ones(num_optimized_vars) * upper_bound
end

function loss(p,q)
    input_change = rand()*9+1
    prob = ODEProblem(rpa_ode_opt!, u0, tspan)
    q_reparameterized =  reparm(q) # ensure parameters are within the range
    sol = solve(prob, Euler(), p = vcat(p,q_reparameterized,input_change), dt=0.1, saveat = [40,90])[1,:]
    return mean(x -> norm(x)^2, sol .- 10)
end

prob = ODEProblem(rpa_ode_opt!, u0, tspan)

optimization_repeats = 1

thompson = zeros(100,5, optimization_repeats)
adtype = Optimization.AutoZygote()

num_iter = 10000
step_size = 0.001

Random.seed!(0); # Ensure reproducibility

using Base.Threads

lk = ReentrantLock()

loss_trace = zeros(num_iter, 100)


println("Start sampling")
@time @threads for i=1:100
    try
        for j = 1:optimization_repeats
            res = randn(5)
            for k=1:num_iter
                res = res .- step_size*Zygote.gradient(q -> loss(prior_samples_arr[i,:],q),res)[1]
                loss_trace[k,i] = loss(prior_samples_arr[i,:],res)
            end
            lock(lk) do
                thompson[i,:,j] = res
            end
        end
        println("Done "*string(i))
    catch
        println("Instability detected. Go to next sample.")
        continue
    end
end


thompson_reparm = sigmoid.(thompson) .* reshape(ones(num_optimized_vars) * upper_bound, 1, 5, 1) 

f = open(string(@__DIR__)*"/thompson_reparm_prior.jls", "w")
serialize(f, thompson_reparm)
close(f)
