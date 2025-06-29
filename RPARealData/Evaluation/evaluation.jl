include(string(@__DIR__)*"/../Model/model.jl")
using CSV, DataFrames
using Tables
using Plots
using Optimization, OptimizationPolyalgorithms, SciMLSensitivity, Zygote
using CairoMakie
using PairPlots
using Random
using Turing
using LinearAlgebra
using Serialization
using StatsPlots

Random.seed!(0)

# Read the data
df = CSV.read(string(@__DIR__)*"/../Inference/time_points.csv", DataFrame)
time = Matrix(df)
df = CSV.read(string(@__DIR__)*"/../Inference/data.csv", DataFrame)
data = Matrix(df)
background_fluorescence = 17.6
data = data .- background_fluorescence

post = Array(CSV.read(string(@__DIR__)*"/../Inference/posterior_samples.csv", DataFrame))
thompson_samples = Array(CSV.read(string(@__DIR__)*"/../Design/thompson_samples.csv", DataFrame, header=false))

lk = ReentrantLock()
using Main.Threads
function median_quantile_loss(k)
    residues = []
    @sync @threads for i in 1:length(post[:,1])

        posterior_sample = vcat(post[i,2:4],2.75e3, post[i,5:16],4006.9,post[i,17:17])

        posterior_sample[7] *= k

        p_opt = posterior_sample

        y0 = [24.0, 350.0]  # Initial values for y1 and y2

        y0_og = y0

        cuma = 2 * 1e-6

        prob1 = ODEProblem(odes_warm_up!, y0, tspan, p_opt)

        # Solve the ODE
        sol = solve(prob1, Rosenbrock23())

        y0 =  sol[:,end]

        cuma = 300 * 1e-6
        predicted = solve(prob, Rosenbrock23(); p= vcat(p_opt,cuma), saveat=time, u0=y0)[1,:] .+ background_fluorescence
        lock(lk) do
            res = ((predicted[end] - 50).^2 + (y0[1] + background_fluorescence - 50).^2)/2
            residues = push!(residues, res)
        end
    end
    return median(residues), quantile(residues, 0.75), std(residues) 
end

@time evaluation = median_quantile_loss.(thompson_samples)


CSV.write(string(@__DIR__)*"/evaluation.csv", DataFrame(Tables.table(evaluation)), writeheader = false)


med_res = [x[1] for x in evaluation]

quant_res = [x[2] for x in evaluation]

std_res = [x[3] for x in evaluation]

Plots.scatter(med_res, quant_res, xlabel = "Median Residue", ylabel = "Quantile Residue", title = "Evaluation of Thompson Samples", label = "Evaluation", color = "blue")

indices = findall(x -> x[1] < 344  && x[2] < 387, evaluation)
indices_arr = map(x -> x[1], indices)
good_values = thompson_samples[indices]
centroid = mean(good_values)

Plots.scatter!(med_res[indices_arr], quant_res[indices_arr], xlabel = "Median Residue", ylabel = "Quantile Residue", title = "Evaluation of Thompson Samples", label = "Good Values", color = "red")

CSV.write(string(@__DIR__)*"/indices.csv", DataFrame(Tables.table(indices_arr)), writeheader = false)
CSV.write(string(@__DIR__)*"/medians.csv", DataFrame(Tables.table(med_res)), writeheader = false)
CSV.write(string(@__DIR__)*"/quantiles.csv", DataFrame(Tables.table(quant_res)), writeheader = false)