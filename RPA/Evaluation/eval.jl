using LinearAlgebra
using Serialization
using Statistics
using Turing
using Plots
using CSV, Tables
include(string(@__DIR__)*"/../Model/rpa_ode.jl")

f = open(string(@__DIR__)*"/thompson_samples.jls", "r")
thompson_reparm = deserialize(f)
close(f)

f = open(string(@__DIR__)*"/posterior_samples.jls", "r")
uncertainty_samples = deserialize(f)
close(f)



prob = ODEProblem(rpa_ode_opt!, u0, tspan)
function loss(p,q)
    input_change = 10
    sol = solve(prob, Euler(), p = vcat(p,q,input_change), dt=0.1, saveat = [40,90])[1,:]
    return mean(x -> norm(x)^2, sol .- 10)
end

evaluation = zeros(99,100)
for i=1:99
    for j = 1:100
            result = loss(uncertainty_samples[i,:],thompson_reparm[j,:])
            evaluation[i,j] = result
    end
end 


cluster_indices = findall(x -> median(x)<0.5 && quantile(x,0.75)<0.75, eachcol(evaluation))

cluster_samples = thompson_reparm[cluster_indices,:]

centroid = mean(cluster_samples,dims=1)

CSV.write(string(@__DIR__)*"/evaluation.csv",  Tables.table(evaluation), writeheader=false)

CSV.write(string(@__DIR__)*"/cluster_indices.csv",  Tables.table(cluster_indices), writeheader=false)

CSV.write(string(@__DIR__)*"/centroid.csv",  Tables.table(centroid), writeheader=false)

CSV.write(string(@__DIR__)*"/thompson_samples.csv",  Tables.table(thompson_reparm[:,:,1]), writeheader=false)

CSV.write(string(@__DIR__)*"/cluster_thompson_samples.csv",  Tables.table(cluster_samples), writeheader=false)