using RecursiveArrayTools, Distributions, Plots, StatsPlots
using Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using Zygote
using Turing
using LinearAlgebra
using Serialization
using CSV, Tables
include(string(@__DIR__)*"/../Model/rpa_ode.jl")

using Random
Random.seed!(0);

init_cond = [1.0,1.0]
p = [0.1,0.001,0.01,0.001] # ground truth
rpa_prob = prob
sol = solve(rpa_prob, Euler(),dt =0.01, p = p)
CSV.write(string(@__DIR__)*"/sol_true.csv",  Tables.table(sol.u), writeheader=false)


t = collect(range(1, stop = 90, length = 30))
randomized = VectorOfArray([(sol(t[i])[1] + 1*randn()) for i in 1:length(t)])
data = convert(Array, randomized)
CSV.write(string(@__DIR__)*"/data_true.csv",  Tables.table(data), writeheader=false)



@model function fit(data::AbstractVector, prob)

    σ ~ InverseGamma(2, 3)
    β_RA ~ truncated(Distributions.Uniform(0.0, 1.0), lower = 0.0)
    β_BA ~ truncated(Distributions.Uniform(0.0, 1.0), lower = 0.0)
    β_AB  ~ truncated(Distributions.Uniform(0.0, 1.0), lower = 0.0)
    β_BB ~ truncated(Distributions.Uniform(0.0, 1.0), lower = 0.0)

    p = [β_RA, β_BA, β_AB, β_BB]
    predicted = solve(rpa_prob, Euler(); dt = 0.01, p=p, saveat=t, save_idxs=1)

    data ~ MvNormal(predicted.u, σ^2 * I)

    return nothing
end

model2 = fit(data, prob)

@time chain = sample(model2, NUTS(0.65), MCMCThreads(), 1000, 3; progress=false)

f = open(string(@__DIR__)*"/posterior_chains.jls", "w")
serialize(f, chain)
close(f)

