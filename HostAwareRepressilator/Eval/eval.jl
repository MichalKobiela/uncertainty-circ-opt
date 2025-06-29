include("../Design/setup.jl")
using CSV, Tables
using Statistics
using StatsBase

result_arr = Array(CSV.read(string(@__DIR__)*"/thompson.csv", DataFrame, header = false))

evaluate = zeros(length(particles_arr[:,1]), length(result_arr[:,1]))
@time for i = 1:length(result_arr[:,1])
    for j = 1:length(particles_arr[:,1])
        evaluate[j,i] = loss(particles_arr[j,:],result_arr[i,:])
    end
end

using CSV, Tables
CSV.write(string(@__DIR__)*"/eval.csv",  Tables.table(evaluate), writeheader=false)
# read the evaluate file
# evaluate = Array(CSV.read(string(@__DIR__)*"/eval.csv", DataFrame, header = false))

means = median(evaluate, dims = 1)
quantiles = mapslices(row -> quantile(row, 0.75), evaluate, dims=1)

selected = []
for i=1:length(means)
    if means[i] < 0.01 && quantiles[i] < 0.04
        push!(selected, i)
    end
end

result_selected = result_arr[selected,:]

scatter(result_arr[:,1], result_arr[:,2], xlabel = "kf", ylabel = "th", legend = false, title = "Thompson samples")
scatter!(result_selected[:,1], result_selected[:,2], color = "red")

CSV.write(string(@__DIR__)*"/selected.csv",  Tables.table(selected), writeheader=false)
CSV.write(string(@__DIR__)*"/medians.csv",  Tables.table(means), writeheader=false)
CSV.write(string(@__DIR__)*"/quantiles.csv",  Tables.table(quantiles), writeheader=false)

