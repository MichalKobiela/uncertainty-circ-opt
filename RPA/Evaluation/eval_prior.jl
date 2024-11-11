using LinearAlgebra
using Serialization
using Random
using Turing
using Plots
using CSV, Tables
include(string(@__DIR__)*"/../Model/rpa_ode.jl")


f = open(string(@__DIR__)*"/prior_samples.jls", "r")
opt_samples = deserialize(f)
close(f)

uncertainty_samples = rand(100,4)

prob = ODEProblem(rpa_ode_opt!, u0, tspan)
function loss(p,q)
    input_change = 10
    sol = solve(prob, Euler(), p = vcat(p,q,input_change), dt=0.1, saveat = [40,90])[1,:]
    return mean(x -> norm(x)^2, sol .- 10)
end

evaluation = zeros(100,100)
for i=1:100
    for j = 1:100
            result = loss(uncertainty_samples[i,:],opt_samples[j,:])
            evaluation[i,j] = result
    end
end 

CSV.write(string(@__DIR__)*"/evaluation_prior.csv",  Tables.table(evaluation), writeheader=false)