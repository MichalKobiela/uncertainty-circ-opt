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


using StatsPlots

Plots.plot()

traj_20 = []
traj_100 = []
traj_300 = []
traj_1000 = []

for i in 1:length(post[:,1])

    posterior_sample = vcat(post[i,2:4],2.75e3, post[i,5:16],4006.9,post[i,17:17])

    p_opt = posterior_sample

    y0 = [24.0, 350.0]  # Initial values for y1 and y2

    y0_og = y0

    cuma = 2 * 1e-6

    prob1 = ODEProblem(odes_warm_up!, y0, tspan, p_opt)

    # Solve the ODE
    sol = solve(prob1, Rosenbrock23())

    y0 =  sol[:,end]

    cuma = 20 * 1e-6
    predicted = solve(prob, Rosenbrock23(); p= vcat(p_opt,cuma), saveat=time, u0=y0)[1,:]
    push!(traj_20, predicted)
    Plots.plot!(time,predicted, label = "model input = 20", color = "blue")


    cuma = 100 * 1e-6
    predicted = solve(prob, Rosenbrock23(); p= vcat(p_opt,cuma), saveat=time, u0=y0)[1,:]
    push!(traj_100, predicted)
    Plots.plot!(time,predicted, label = "model input = 100", color = "red")

    cuma = 300 * 1e-6
    predicted = solve(prob, Rosenbrock23(); p= vcat(p_opt,cuma), saveat=time, u0=y0)[1,:]
    push!(traj_300, predicted)
    Plots.plot!(time,predicted, label = "model input = 300", color = "orange")


    cuma = 1000 * 1e-6
    predicted = solve(prob, Rosenbrock23(); p= vcat(p_opt,cuma), saveat=time, u0=y0)[1,:]
    push!(traj_1000, predicted)
    Plots.plot!(time,predicted, label = "model input = 1000", color = "green")
end

traj_20_arr = hcat(traj_20...)
traj_100_arr = hcat(traj_100...)
traj_300_arr = hcat(traj_300...)
traj_1000_arr = hcat(traj_1000...)
# save
CSV.write(string(@__DIR__)*"/posterior_predictive_traj_20.csv", DataFrame(Tables.table(traj_20_arr')), header = false)
CSV.write(string(@__DIR__)*"/posterior_predictive_traj_100.csv", DataFrame(Tables.table(traj_100_arr')), header = false)
CSV.write(string(@__DIR__)*"/posterior_predictive_traj_300.csv", DataFrame(Tables.table(traj_300_arr')), header = false)
CSV.write(string(@__DIR__)*"/posterior_predictive_traj_1000.csv", DataFrame(Tables.table(traj_1000_arr')), header = false)





Plots.scatter!(time,data[:,2], label = "data input = 20", color = "blue")
Plots.scatter!(time,data[:,5], label = "data input = 100", color = "red")
Plots.scatter!(time,data[:,end], label = "data input = 1000", color = "green")
Plots.plot!(title = "Posterior predictive", legend = false)


traj = []
for i in 1:length(post[:,1])

    posterior_sample = vcat(post[i,2:4],2.75e3, post[i,5:16],4006.9,post[i,17:17])

    posterior_sample[7] *= 0.21

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
    push!(traj, predicted)
    Plots.plot!(time,predicted, label = "model input = 20", color = "red")
end

traj_arr = hcat(traj...)
CSV.write(string(@__DIR__)*"/posterior_predictive_traj_0.21.csv", DataFrame(Tables.table(traj_arr')), header = false)

traj = []
for i in 1:length(post[:,1])

    posterior_sample = vcat(post[i,2:4],2.75e3, post[i,5:16],4006.9,post[i,17:17])

    posterior_sample[7] *= 0.17

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
    push!(traj, predicted)
    Plots.plot!(time,predicted, label = "model input = 20", color = "red")
end

traj_arr = hcat(traj...)
CSV.write(string(@__DIR__)*"/posterior_predictive_traj_0.17.csv", DataFrame(Tables.table(traj_arr')), header = false)

#prior predictive
Plots.plot()
Plots.plot!(ylims = (0, 200))

for i in 1:500

    # Define truncated uniform distributions
    alfa1_sample = rand(Truncated(Uniform(0.0, 2000.0), 0.0, Inf))
    kx1_sample = rand(Truncated(Uniform(0.0, 3.0e-8), 0.0, Inf))
    nx1_sample = rand(Truncated(Uniform(1.0, 5.0), 0.0, Inf))
    kcymRtot_sample = rand(Truncated(Uniform(0.0, 2.70e4), 0.0, Inf))
    beta1_sample = rand(Truncated(Uniform(0.0, 200.0), 0.0, Inf))
    alfa2_sample = rand(Truncated(Uniform(0.0, 250.0), 0.0, Inf))
    kx2_sample = rand(Truncated(Uniform(0.0, 1000.0), 0.0, Inf))
    nx2_sample = rand(Truncated(Uniform(1.0, 10.0), 0.0, Inf))
    beta2_sample = rand(Truncated(Uniform(0.0, 50.0), 0.0, Inf))
    alfa4_sample = rand(Truncated(Uniform(0.0, 1.0e9), 0.0, Inf))
    kr_sample = rand(Truncated(Uniform(0.0, 100.0), 0.0, Inf))
    nr_sample = rand(Truncated(Uniform(1.0, 10.0), 0.0, Inf))
    beta4_sample = rand(Truncated(Uniform(0.0, 200.0), 0.0, Inf))
    r1_sample = rand(Truncated(Uniform(0.0, 1000.0), 0.0, Inf))
    r2_sample = rand(Truncated(Uniform(0.0, 100.0), 0.0, Inf))
    alfa3_sample = rand(Truncated(Uniform(0.0, 1000.0), 0.0, Inf))
    beta3_sample = rand(Truncated(Uniform(0.0, 10.0), 0.0, Inf))

    # Fixed parameter
    kx3_sample = 4006.9

    # Store in an array
    params_sample = [alfa1_sample, kx1_sample, nx1_sample, kcymRtot_sample, beta1_sample, 
                    alfa2_sample, kx2_sample, nx2_sample, beta2_sample, alfa4_sample, 
                    kr_sample, nr_sample, beta4_sample, r1_sample, r2_sample, 
                    alfa3_sample, kx3_sample, beta3_sample]

    p_opt = params_sample

    y0 = [24.0, 350.0]  # Initial values for y1 and y2

    y0_og = y0

    cuma = 2 * 1e-6

    prob1 = ODEProblem(odes_warm_up!, y0, tspan, p_opt)

    # Solve the ODE
    sol = solve(prob1, Rosenbrock23())

    y0 =  sol[:,end]

    cuma = 300 * 1e-6
    predicted = solve(prob, Rosenbrock23(); p= vcat(p_opt,cuma), saveat=time, u0=y0)[1,:] .+ background_fluorescence
    Plots.plot!(time,predicted, label = "model input = 20", color = "blue")
end

Plots.plot!(legend=false)

Plots.plot!(title = "Prior predicitive for input = 300", legend = false)
