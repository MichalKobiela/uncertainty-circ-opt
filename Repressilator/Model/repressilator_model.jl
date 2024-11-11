using StochasticDiffEq
using Random


Random.seed!(0)

k_transcription = 100.0
k_degradation_A = 0.5
k_degradation_B = 0.5
k_degradation_C = 0.5
n_A = 2.1
n_B = 2.0
n_C = 1.9
K_A = 1.0
K_B = 1.0
K_C = 1.0



p = [0.5, 0.5, 0.5, 2.1, 2.0, 1.9, 100.0, 1.0]

ground_truth_unc = [0.5, 0.5, 0.5, 2.0, 2.0, 2.0]

function f(du, u, p, t)
    A = u[1]
    B = u[2] 
    C = u[3]

    k_degradation_A = p[1]
    k_degradation_B = p[2]
    k_degradation_C = p[3]
    n_A = p[4]
    n_B = p[5]
    n_C = p[6]
    k_transcription = p[7]
    K = p[8]
    
    du[1] = k_transcription / (1 + abs(C / K)^n_C) - k_degradation_A*A
    du[2] = k_transcription / (1 + abs(A / K)^n_A) - k_degradation_B*B
    du[3] = k_transcription / (1 + abs(B / K)^n_B) - k_degradation_C*C
end


function g(du, u, p, t)
    A = u[1]
    B = u[2] 
    C = u[3]

    k_degradation_A = p[1]
    k_degradation_B = p[2]
    k_degradation_C = p[3]
    n_A = p[4]
    n_B = p[5]
    n_C = p[6]
    k_transcription = p[7]
    K = p[8]

    du[1,1] = sqrt(abs(k_transcription / (1 + abs(C / K)^n_C)))
    du[1,2] = sqrt(abs(k_degradation_A*A))
    du[1,3] = 0 
    du[1,4] = 0
    du[1,5] = 0
    du[1,6] = 0

    du[2,1] = 0
    du[2,2] = 0
    du[2,3] = sqrt(abs(k_transcription / (1 + abs(A / K)^n_A))) 
    du[2,4] = sqrt(abs(k_degradation_B*B))
    du[2,5] = 0
    du[2,6] = 0

    du[3,1] = 0
    du[3,2] = 0
    du[3,3] = 0
    du[3,4] = 0
    du[3,5] = sqrt(abs(k_transcription / (1 + abs(B / K)^n_B)))
    du[3,6] = sqrt(abs(k_degradation_C*C))
end

init_cond = zeros(3)
prob_det = ODEProblem(f,init_cond,(0.0, 2000.0),p) 
prob = SDEProblem(f, g,init_cond, (0.0, 200.0),p,noise_rate_prototype = zeros(3, 6)) 

sol = solve(prob, EM(), dt = 0.1, p = vcat(ground_truth_unc, [435.80181004585216, 0.5218630366540828]))[2,1000:2000]