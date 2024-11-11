using StochasticDiffEq
using Statistics
using Plots
using FFTW
using BenchmarkTools
using StatsBase
using Random
using DataFrames
include("../Model/repressilator_model.jl")

init_cond = zeros(3)
# prob_det = ODEProblem(f,init_cond,(0.0, 2000.0),p) 
prob = SDEProblem(f, g,init_cond, (0.0, 200.0),p,noise_rate_prototype = zeros(3, 6)) 

sol = solve(prob, EM(), dt = 0.1)
#sol = solve(prob_det,Euler(), dt = 0.1)

function rectangular_wave(length, period, tau)
    num_of_osc = length÷period
    valley = period-tau
    single_osc = vcat( zeros(valley), ones(tau))
    return repeat(single_osc, num_of_osc+1)[1:length]
  end
target = 250 .* rectangular_wave(1000,200,200÷5)


function objectiveFFT_matching(signal, target)
    FFT_sig = fft(signal)
    FFT_target = fft(target)
    F_sig = abs.(FFT_sig)[2:length(signal)÷2]
    F_target = abs.(FFT_target)[2:length(signal)÷2] 
    return sqrt(mean((F_sig.-F_target).^2))
end

tdata = randn(1000) .* 0.04 .+ 2;

num_iter = 1

t = [1:1000;]

data_x = zeros(num_iter,2)
data_y =  zeros(num_iter,1000)

ground_truth_unc = [0.5, 0.5, 0.5, 2.0, 2.0, 2.0]

Random.seed!(3)
for i = 1:num_iter
    params = rand(2) .* [1000,10]
    data_x[i,:] = params
    traj = solve(prob,EM(),p=vcat(ground_truth_unc, params),dt=0.1)[2,1001:2000]
    data_y[i,:] =  traj
end

function loss_correction(p_in)
    p_unc = [p_in[1],p_in[1],p_in[1],p_in[2],p_in[2],p_in[2]]
    result = 0
    for i = 1:size(data_x)[1]
            result = result + objectiveFFT_matching(solve(prob,EM(),p=vcat(p_unc,data_x[i,:]),dt=0.1)[2,1001:2000], data_y[i,:])/1000
    end
    return result/size(data_x)[1]
end

using KissABC
prior = Product(fill(Uniform(0, 10), 2));

Random.seed!(0)
@time ressmc_1 = smc(prior, loss_correction, nparticles=100, alpha = 0.95, parallel = true, M=10)
@time ressmc_2 = smc(prior, loss_correction, nparticles=100, alpha = 0.95, parallel = true, M=10)
@time ressmc_3 = smc(prior, loss_correction, nparticles=100, alpha = 0.95, parallel = true, M=10)

scatter(ressmc_1.P[1].particles,ressmc_1.P[2].particles, label = "SMC first run")
scatter!(ressmc_2.P[1].particles,ressmc_2.P[2].particles, label = "SMC second run")
scatter!(ressmc_3.P[1].particles,ressmc_3.P[2].particles, label = "SMC third run", title = "Oscillator Posterior",xlabel = "Degradation rate",ylabel = "n")


scatter!([0.5],[2.0]; label = "Ground truth", markersize = 7)

result_1 = hcat(ressmc_1.P[1].particles, ressmc_1.P[2].particles)
result_2 = hcat(ressmc_2.P[1].particles, ressmc_2.P[2].particles)
result_3 = hcat(ressmc_3.P[1].particles, ressmc_3.P[2].particles)

result = vcat(result_1, result_2, result_3)

using CSV, Tables

CSV.write(string(@__DIR__)*"/posterior_samples.csv",  DataFrame(result, :auto), )

CSV.write(string(@__DIR__)*"/data_y.csv",  DataFrame(data_y, :auto), )
CSV.write(string(@__DIR__)*"/data_x.csv",  DataFrame(data_x, :auto), )