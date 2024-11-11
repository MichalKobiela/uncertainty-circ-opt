using RecursiveArrayTools, Distributions, Plots, StatsPlots
using Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using Zygote, Turing, StatsPlots, Random
using LinearAlgebra, Serialization, CSV, Tables, JLD2
include(string(@__DIR__)*"/../Model/rpa_ode.jl")

Random.seed!(0); 

# Open the the posterior data
f = open(string(@__DIR__)*"/posterior_chains.jls", "r")
chain = deserialize(f)
close(f)
posterior_samples = sample(chain[[:β_RA, :β_BA, :β_AB, :β_BB]], 100; replace=false)
samples = Array(posterior_samples)

# We reparemtrize Thompson samples to ensure they are within the desired range
sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))

upper_bound = 1000
num_optimized_vars = 5

function reparm(x)
    return sigmoid.(x) .* ones(num_optimized_vars) * upper_bound
end

# Strategy to avoid local minima, see docs.sciml.ai/SciMLSensitivity/stable/tutorials/training_tips/local_minima/
function loss_warm_up(p,q)
    input_change = rand()*9+1
    prob = ODEProblem(rpa_ode_opt!, u0, tspan)
    q_reparameterized =  reparm(q) # ensure parameters are within the range
    sol = solve(prob, Euler(), p = vcat(p,q_reparameterized,input_change), dt=0.1, saveat = [40])[1,:]
    return mean(x -> norm(x)^2, sol .- 10)
end

# Actual loss function
function loss(p,q)
    input_change = rand()*9+1
    prob = ODEProblem(rpa_ode_opt!, u0, tspan)
    q_reparameterized =  reparm(q) # ensure parameters are within the range
    sol = solve(prob, Euler(), p = vcat(p,q_reparameterized,input_change), dt=0.1, saveat = [40,90])[1,:]
    return mean(x -> norm(x)^2, sol .- 10)
end

prob = ODEProblem(rpa_ode_opt!, u0, tspan)

optimization_repeats = 10

thompson = zeros(100,5)
adtype = Optimization.AutoZygote()

num_iter = 10000
num_iter_worm_up = 1000
step_size = 0.005

Random.seed!(0); # Ensure reproducibility

using Base.Threads

lk = ReentrantLock()

posterior_samples_arr = Array(posterior_samples)

loss_trace_worm_up = zeros(optimization_repeats,num_iter_worm_up)

loss_trace = zeros(optimization_repeats,num_iter)

println("Start sampling")

function thompson_sample()
    @time @threads for i=1:100
        best_res = randn(5)
        for j = 1:optimization_repeats
            if j == 1 # Use the best result from the previous optimization as the initial guess
                if i == 1
                    res = best_res
                else
                    res = thompson[i-1,:]
                end
            else
                res = randn(5)
            end
            for k=1:num_iter_worm_up
                res = res .- step_size*Zygote.gradient(q -> loss_warm_up(posterior_samples_arr[i,:],q),res)[1]
                loss_trace_worm_up[j,k] = loss_warm_up(posterior_samples_arr[i,:],res)
            end
            for k=1:num_iter
                res = res .- step_size*Zygote.gradient(q -> loss(posterior_samples_arr[i,:],q),res)[1]
                loss_trace[j,k] = loss(posterior_samples_arr[i,:],res)
            end

            if loss(posterior_samples_arr[i,:],best_res) > loss(posterior_samples_arr[i,:],res)
                best_res = res
            end
        end

        lock(lk) do
            thompson[i,:] = best_res
        end

        println("Done "*string(i))
    end
    return thompson
end

thompson = thompson_sample()

thompson_reparm = sigmoid.(thompson) .* reshape(ones(num_optimized_vars) * upper_bound, 1, 5, 1) 

f = open(string(@__DIR__)*"/thompson_samples.jls", "w")
serialize(f, thompson_reparm)
close(f)
