include(string(@__DIR__)*"/eval.jl")

exemplar_designs = [34,64,31]

for i = 1:3
    sols = zeros(100,1001)
    plot(ylim=(0, 15))
    for j = 1:100
        prob = ODEProblem(rpa_ode_opt!, u0, tspan)
        sol = solve(prob, Euler(), p = vcat(uncertainty_samples[j,:],thompson_reparm[exemplar_designs[i],:],10), dt=0.1)[1,:]
        sols[j,:] = sol
        plot!(sol, color= "Grey", label = "")
    end
    CSV.write(string(@__DIR__)*"/exemplar_design$(i).csv",  Tables.table(sols), writeheader=false)
end

sols = zeros(100,1001)
plot(ylim=(0, 15))
for i = 1:100
    prob = ODEProblem(rpa_ode_opt!, u0, tspan)
    sol = solve(prob, Euler(), p = vcat(uncertainty_samples[i,:],vec(centroid),10), dt=0.1)[1,:]
    sols[i,:] = sol
    plot!(sol, color= "Grey", label = "")
end
CSV.write(string(@__DIR__)*"/centroid_design.csv",  Tables.table(sols), writeheader=false)

p = [0.1,0.001,0.01,0.001] # ground truth
sol = solve(prob, Euler(), p = vcat(p,vec(centroid),10), dt=0.1)[1,:]
CSV.write(string(@__DIR__)*"/ground_truth_centroid.csv",  Tables.table(sol), writeheader=false)