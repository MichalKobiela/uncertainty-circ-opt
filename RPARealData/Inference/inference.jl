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
df = CSV.read(string(@__DIR__)*"/time_points.csv", DataFrame)
time = Matrix(df)
df = CSV.read(string(@__DIR__)*"/data.csv", DataFrame)
data = Matrix(df)
background_fluorescence = 17.6
data = data .- background_fluorescence


# Plot the
Plots.scatter(time, data[:,2], xlabel = "Time", ylabel = "Concentration", title = "Data", label="Data input = 20", color = "blue")
Plots.scatter!(time, data[:,5], xlabel = "Time", ylabel = "Concentration", title = "Data", label="Data input = 100", color = "green")
Plots.scatter!(time, data[:,end], xlabel = "Time", ylabel = "Concentration", title = "Data", label="Data input = 1000", color = "red")
Plots.plot!(;title = "Parameters fitted with gradient descent")

adtype = Optimization.AutoZygote()


@model function fit(data::AbstractVector, prob)

    σ ~ InverseGamma(2, 3)


    alfa1 ~ truncated(Distributions.Uniform(0.0, 2000.0), lower = 0.0)
    kx1 ~ truncated(Distributions.Uniform(0, 3.0e-8), lower = 0.0)
    nx1 ~ truncated(Distributions.Uniform(1.0, 5.0), lower = 0.0)
    kcymRtot = 2.75e3
    beta1  ~ truncated(Distributions.Uniform(0, 200.0), lower = 0.0)
    alfa2 ~ truncated(Distributions.Uniform(0.0, 250.0), lower = 0.0)
    kx2 ~ truncated(Distributions.Uniform(0.0, 10000), lower = 0.0)
    nx2 ~ truncated(Distributions.Uniform(1.0, 10.0), lower = 0.0)
    beta2 ~ truncated(Distributions.Uniform(0, 100.0), lower = 0.0)
    alfa4 ~ truncated(Distributions.Uniform(0, 1.0e13), lower = 0.0)
    kr ~ truncated(Distributions.Uniform(0.0, 100.0), lower = 0.0)
    nr ~ truncated(Distributions.Uniform(1.0, 100.0), lower = 0.0)
    beta4 ~ truncated(Distributions.Uniform(0,5000), lower = 0.0)
    r1 ~ truncated(Distributions.Uniform(0.0,1000.0), lower = 0.0)
    r2 ~ truncated(Distributions.Uniform(0.0, 1000.0), lower = 0.0)
    alfa3 ~ truncated(Distributions.Uniform(0.0, 10000.0), lower = 0.0)
    beta3 ~ truncated(Distributions.Uniform(0.0, 5000.0), lower = 0.0)
    kx3 = 4006.9
    params = [alfa1, kx1, nx1, kcymRtot, beta1, alfa2, kx2, nx2, beta2, alfa4, kr, nr, beta4, r1, r2, alfa3, kx3, beta3]
    
    y0_init = [24.0, 350.0]  # Initial values for y1 and y2

    # Solve the ODE
    try
        sol = solve(prob1, Rosenbrock23(), p=params, u0=y0_init, dtmin=1e-12)
        y0 = sol[:, end]
    
        input = 20 / 1e6
        predicted = solve(prob, Rosenbrock23(); p=vcat(params, input), saveat=time, u0=y0, dtmin=1e-12)[1,:]
    
        input = 100 / 1e6
        predicted_100 = solve(prob, Rosenbrock23(); p=vcat(params, input), saveat=time, u0=y0, dtmin=1e-12)[1,:]
    
        input = 1000 / 1e6
        predicted_1000 = solve(prob, Rosenbrock23(); p=vcat(params, input), saveat=time, u0=y0, dtmin=1e-12)[1,:]
    
        data ~ MvNormal(vcat(predicted, predicted_100, predicted_1000), σ^2 * I)
    
    catch e
        print(e)
            Turing.@addlogprob! -1e10  # reject bad bad samples
    end

    return nothing
end

model2 = fit(vcat(data[:,2], data[:,5], data[:,end]), prob)

# Initilize parameters using results from the RPA paper
init_params = Dict(
    :σ => 3.0,
    :alfa1 => 83.4743,
    :kx1 => 1.28e-8,
    :nx1 => 2.34,
    :beta1 => 11.9586,
    :alfa2 => 391.1627,
    :kx2 => 36.4063,
    :nx2 => 1.3,
    :beta2 => 3.9e-4,
    :alfa4 => 8.7519e6,
    :kr => 0.51,
    :nr => 3.2,
    :beta4 => 7.1347,
    :r1 => 89.0635,
    :r2 => 7.0188,
    :alfa3 => 17.7437,
    :beta3 => 0.6644
    # kx3 is fixed, so it's not included in the initial parameters
)

init_params_arr = [3.0,83.4743, 1.28e-8, 2.34, 2.75e3, 11.9586, 391.1627, 36.4063, 1.3, 3.9e-4, 8.7519e6, 0.51, 3.2, 7.1347, 89.0635, 7.0188, 17.7437, 4006.9, 0.6644]

# We choose random seeds that run in reasonable time and appear to mix well
# Because the inference precedure is not completely robust, we run three independent chains
# We report the results of all of them combined in the main text
# and each of them separetely in the supplementary material
Random.seed!(4)
@time chain_1 = sample(model2,  NUTS(0.65,init_ϵ = 0.001), MCMCSerial(), 3000, 1, init_params = init_params)

f = open(string(@__DIR__)*"/posterior_samples_large_range_1_c.jls", "w")
serialize(f, chain_1)
close(f)

Random.seed!(6)
@time chain_2 = sample(model2,  NUTS(0.65,init_ϵ = 0.001), MCMCSerial(), 3000, 1, init_params = init_params)

using Serialization
f = open(string(@__DIR__)*"/posterior_samples_large_range_2_c.jls", "w")
serialize(f, chain_2)
close(f)

Random.seed!(0)
@time chain_3 = sample(model2,  NUTS(0.65,init_ϵ = 0.001), MCMCSerial(), 3000, 1, init_params = init_params)


f = open(string(@__DIR__)*"/posterior_samples_large_range_3_c.jls", "w")
serialize(f, chain_3)
close(f)

param_names = [:σ, :alfa1, :kx1, :nx1, :kcymRtot, :beta1, :alfa2, :kx2, :nx2, :beta2, :alfa4, :kr, :nr, :beta4, :r1, :r2, :alfa3, :kx3, :beta3]

#open the chains

f = open(string(@__DIR__)*"/posterior_samples_large_range_1_c.jls", "r")
chain_1 = deserialize(f)
close(f)

f = open(string(@__DIR__)*"/posterior_samples_large_range_2_c.jls", "r")
chain_2 = deserialize(f)
close(f)

f = open(string(@__DIR__)*"/posterior_samples_large_range_3_c.jls", "r")
chain_3 = deserialize(f)
close(f)

StatsPlots.plot(chain_1)

using Serialization
f = open(string(@__DIR__)*"/posterior_samples_6_c.jls", "w")
serialize(f, chain_1)
close(f)

using Serialization
f = open(string(@__DIR__)*"/posterior_samples_2_big_ranges.jls", "w")
serialize(f, chain_2)
close(f)

f = open(string(@__DIR__)*"/posterior_samples_2_big_ranges.jls", "r")
chain_1 = deserialize(f)
close(f)

using Serialization
f = open(string(@__DIR__)*"/posterior_chains.jls", "w")
serialize(f, chain)
close(f)
using StatsPlots
plot(chain_1)

StatsPlots.plot(chain_1)

# Increased burnout for better mixing
StatsPlots.plot(chain_1[1000:2000])
StatsPlots.plot!(chain_2[1000:2000])
StatsPlots.plot!(chain_3[1000:2000])

# save the plot
savefig(string(@__DIR__)*"/posterior.pdf")

posterior_samples = sample(chain_3, 1000; replace=false)
post1 = Array(chain_1[1000:2000])
post2 = Array(chain_2[1000:2000])
post3 = Array(chain_3[1000:2000])
post = vcat(post1, post2, post3)


# pair plot
post_df = DataFrame(post,  [:σ, :alfa1, :kx1, :nx1, :beta1, :alfa2, :kx2, :nx2, :beta2, :alfa4, :kr, :nr, :beta4, :r1, :r2, :alfa3, :beta3])
fig = pairplot(post_df)

# Save the pair plot
save(string(@__DIR__)*"/pair_plot.pdf", fig)


CSV.write(string(@__DIR__)*"/posterior_samples.csv",  post_df)