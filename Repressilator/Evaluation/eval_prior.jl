using Serialization
using StochasticDiffEq
using Statistics
using FFTW
using BenchmarkTools
using StatsBase
using Random
using DataFrames
using CSV, DataFrames
using Base.Threads

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

file = open(string(@__DIR__)*"/designs_prior.jls", "r")
res = deserialize(file)
close(file)

designs = res[:,1,:]


file = open(string(@__DIR__)*"/samples_prior.jls", "r")
samples = deserialize(file)*10
close(file)


evaluation = zeros(10000,100)

lk = ReentrantLock()

Random.seed!(0)
@time @threads for i=1:100
    for j = 1:100
        for k = 1:100
            result = loss(designs[j,:],samples[i,:])
            lock(lk) do
                evaluation[(i-1)*100+k,j] = result
            end 
        end
    end
end 
CSV.write(string(@__DIR__)*"/evaluation_prior.csv",  Tables.table(evaluation), writeheader=false)

row_means = mean(evaluation, dims=1)

argmin(row_means) # best prior opt sample