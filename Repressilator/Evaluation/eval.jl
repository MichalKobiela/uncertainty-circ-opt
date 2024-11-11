using Serialization
using StochasticDiffEq
using Statistics
using FFTW
using BenchmarkTools
using StatsBase
using Random
using DataFrames
using CSV
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

file = open(string(@__DIR__)*"/designs.jls", "r")
res = deserialize(file)
close(file)

designs = res[:,1,:]

df = CSV.read(string(@__DIR__)*"/posterior_samples.csv", DataFrame)
samples = Matrix(df)

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
CSV.write(string(@__DIR__)*"/evaluation.csv",  Tables.table(evaluation), writeheader=false)



row_means = median(evaluation, dims=1)
row_quantiles = mapslices(x -> quantile(x, 0.75), evaluation, dims=1)

indices = findall((row_means .< 5.8) .& (row_quantiles .< 6.5))
row_indices = [i[2] for i in indices]

clust_designs = designs[row_indices,:]

centroid = mean(clust_designs, dims = 1)[:]


CSV.write(string(@__DIR__)*"/centroid.csv",  Tables.table(centroid), writeheader=false)
CSV.write(string(@__DIR__)*"/cluster_indices.csv",  Tables.table(row_indices), writeheader=false)
CSV.write(string(@__DIR__)*"/thompson_samples.csv",  Tables.table(designs), writeheader=false)

