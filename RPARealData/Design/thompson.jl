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

# read the posterior samples to array
post_samples = CSV.read(string(@__DIR__)*"/../Inference/posterior_samples.csv", DataFrame)
post = Array(post_samples)

function loss(posterior_sample, k)
    
    posterior_sample_copy = copy(posterior_sample)

    posterior_sample_copy[7] *= k

    p_opt = posterior_sample_copy

    y0 = [24.0, 350.0]  # Initial values for y1 and y2

    y0_og = y0

    cuma = 2 * 1e-6

    prob1 = ODEProblem(odes_warm_up!, y0, tspan, p_opt)

    # Solve the ODE
    sol = solve(prob1, Rosenbrock23())

    y0 =  sol[:,end]

    cuma = 300 * 1e-6
    predicted = solve(prob, Rosenbrock23(); p= vcat(p_opt,cuma), saveat=time, u0=y0)[1,:] .+ background_fluorescence

    res = ((predicted[end] - 50).^2 + (y0[1] - 50).^2)/2

    return res
end

thompson_samples = []
@time for i in 1:length(post[:,1])
    posterior_sample = vcat(post[i,2:4],2.75e3, post[i,5:16],4006.9,post[i,17:17])
    #optimize loss
    values = collect(LinRange(0.01, 3, 100))
    loss_post = x -> loss(posterior_sample, x)
    losses = loss_post.(values)
    min_loss_index = argmin(losses)
    push!(thompson_samples, values[min_loss_index])
end

thompson_samples
CSV.write(string(@__DIR__)*"/thompson_samples.csv", DataFrame(Tables.table(thompson_samples)), writeheader = false)