using StochasticDiffEq
using Statistics
using FFTW
using BenchmarkTools
using StatsBase
using Random
using DataFrames
include( "../Model/repressilator_model.jl")

function rectangular_wave(length, period, tau)
    num_of_osc = length÷period
    valley = period-tau
    single_osc = vcat( zeros(valley), ones(tau))
    return repeat(single_osc, num_of_osc+1)[1:length]
  end
target = 500 .* rectangular_wave(1000,250,250÷3)

t = [1:1000;]



function objectiveFFT_matching(signal, target)
    FFT_sig = fft(signal)
    FFT_target = fft(target)
    F_sig = abs.(FFT_sig)[2:length(signal)÷2]
    F_target = abs.(FFT_target)[2:length(signal)÷2] 
    return sqrt(mean((F_sig.-F_target).^2))
end


loss(p,unc) = objectiveFFT_matching(solve(prob,EM(),p=vcat([unc[1],unc[1],unc[1],unc[2],unc[2],unc[2]],p),dt=0.1)[2,1001:2000], target)/1000
function loss_iter(p,unc, n = 100)
    losses = zeros(n)
        for i in 1:n
                losses[i] = loss(p,unc)
        end
    return sum(losses)/n
end

using BayesianOptimization, GaussianProcesses, Distributions, Distributed
sample = [0.5,2] # Ground Truth

amplitudes = [250, 500, 1000]
results = []
for i in 1:3
    amp = amplitudes[i]
    target = amp .* rectangular_wave(1000,250,250÷3)
    f_l(x) = loss_iter(x,sample)     
    model = ElasticGPE(2,                           
                    mean = MeanConst(0.),         
                    kernel = SEArd([0., 0.], 5.),
                    logNoise = 0.,
                    capacity = 3000)            
    set_priors!(model.mean, [Normal(1, 2)])

    modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-4, 3],     
                                    kernbounds = [[-1, -1, 0], [4, 4, 10]], 
                                    maxeval = 40)
    opt = BOpt(f_l,
            model,
            UpperConfidenceBound(),    
            modeloptimizer,                        
            [100.0, 0.01], [10000.0, 10.0],         
            repetitions = 3,                 
            maxiterations = 1000,          
            sense = Min, 
            acquisitionoptions = (method = :LD_LBFGS,
                                    restarts = 10,   
                                    maxtime = 0.1, 
                                    maxeval = 1000), 
                verbosity = Progress)

    result = boptimize!(opt)
    push!(results, result)
end

results[1][2]
Random.seed!(1234)
sol_res_1 = solve(prob, EM(), p = vcat(ground_truth_unc, results[1][2]), dt=0.1)[2,1000:2000]
sol_res_2 = solve(prob, EM(), p = vcat(ground_truth_unc, results[2][2]), dt=0.1)[2,1000:2000]
sol_res_3 = solve(prob, EM(), p = vcat(ground_truth_unc, results[3][2]), dt=0.1)[2,1000:2000]

using Plots
plot(sol_res_1, label = "result for target 1")
plot!(sol_res_2, label = "result for target 2")
plot!(sol_res_3, label = "result for target 3")

plot(250*rectangular_wave(1000,250,250÷3), label = "target 1")
plot!(sol_res_1, label = "result for target 1")


plot(500*rectangular_wave(1000,250,250÷3), label = "target 2")
plot!(sol_res_2, label = "result for target 2")


plot(1000*rectangular_wave(1000,250,250÷3), label = "target 3")
plot!(sol_res_3, label = "result for target 3")