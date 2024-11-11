using Serialization
using StochasticDiffEq
using Statistics
using FFTW
using BenchmarkTools
using StatsBase
using Random
using DataFrames
using DSP
using CSV
using Base.Threads

include( "../Model/repressilator_model.jl")

function smooth(row::AbstractVector{T}, window_size::Int = 3) where T
    kernel = ones(T, window_size) / window_size
    padded_row = vcat(repeat([row[1]], floor(Int, window_size / 2)), row, repeat([row[end]], floor(Int, window_size / 2)))
    smoothed_row = conv(padded_row, kernel)
    for i = 1:10
        smoothed_row = conv(smoothed_row, kernel)
    end
    return smoothed_row
end

function first_peak_index(row::AbstractVector{T})::Int where T
    n = length(row)
    for i in 1:n
        if (i > 1 && i < n && row[i] > row[i - 1] && row[i] > row[i + 1])
            return i
        end
    end
    return n + 1
end

function sort_by_first_peak(array::Array{T, 2}, window_size::Int = 3) where T
    smoothed_array = [smooth(row, window_size) for row in eachrow(array)]
    peak_indices = [first_peak_index(row) for row in smoothed_array]
    rows_with_indices = collect(zip(peak_indices, eachrow(array)))
    sorted_rows_with_indices = sort(rows_with_indices, by = x -> x[1])
    sorted_array = hcat([row for (index, row) in sorted_rows_with_indices]...)

    return sorted_array
end

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

df = CSV.read(string(@__DIR__)*"/centroid.csv", DataFrame, header = false)
centroid = Matrix(df)

df = CSV.read(string(@__DIR__)*"/posterior_samples.csv", DataFrame)
samples = Matrix(df)

file = open(string(@__DIR__)*"/samples_prior.jls", "r")
samples_prior = deserialize(file)*10
close(file)

centroid_heatmap = zeros(100,1001)

Random.seed!(0)
for i=1:100
    sol = solve(prob, EM(), p = vcat(ground_truth_unc,centroid), dt=0.1)[2,1000:2000]
    centroid_heatmap[i,:] = sol
end

CSV.write(string(@__DIR__)*"/heatmap_centroid_ale.csv",  Tables.table(sort_by_first_peak(centroid_heatmap,5)), writeheader=false)


centroid_heatmap_epi = zeros(100,1001)

Random.seed!(0)
for i=1:100
    sam_idx = rand(1:100)
    sol = solve(prob, EM(), p = vcat(samples[sam_idx,1],samples[sam_idx,1],samples[sam_idx,1],samples[sam_idx,2],samples[sam_idx,2],samples[sam_idx,2],centroid), dt=0.1)[2,1000:2000]
    centroid_heatmap_epi[i,:] = sol
end

CSV.write(string(@__DIR__)*"/heatmap_centroid_epi.csv",  Tables.table(sort_by_first_peak(centroid_heatmap_epi,5)), writeheader=false)

file = open(string(@__DIR__)*"/designs_prior.jls", "r")
res = deserialize(file)
close(file)

designs_prior = res[:,1,:]

good_prior = designs_prior[16,:]

prior_heatmap = zeros(100,1001)

Random.seed!(0)
for i=1:100
    sol = solve(prob, EM(), p = vcat(ground_truth_unc,good_prior), dt=0.1)[2,1000:2000]
    prior_heatmap[i,:] = sol
end

CSV.write(string(@__DIR__)*"/heatmap_prior_ale.csv",  Tables.table(sort_by_first_peak(prior_heatmap,5)), writeheader=false)

prior_heatmap_epi = zeros(100,1001)
for i=1:100
    sam_idx = rand(1:100)
    sol = solve(prob, EM(), p = vcat(samples_prior[sam_idx,1],samples_prior[sam_idx,1],samples_prior[sam_idx,1],samples_prior[sam_idx,2],samples_prior[sam_idx,2],samples_prior[sam_idx,2],good_prior), dt=0.1)[2,1000:2000]
    prior_heatmap_epi[i,:] = sol
end

CSV.write(string(@__DIR__)*"/heatmap_prior_epi.csv",  Tables.table(sort_by_first_peak(prior_heatmap_epi,5)), writeheader=false)

eval_centroid = zeros(10000)
for i=1:100
        for k = 1:100
            result = loss(centroid,samples[i,:])
            eval_centroid[(i-1)*100+k] = result
        end
end 

CSV.write(string(@__DIR__)*"/centroid_violin.csv",  Tables.table(eval_centroid), writeheader=false)


