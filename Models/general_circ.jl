using DifferentialEquations

alpha = ones(3)*1000
gamma = ones(3)*1
K_pos = ones(3,3)
K_neg = ones(3,3)
pos = [0 0 0 
       0 0 0
       0 0 0] # matrix of positive conections
neg = [0 0 1
       1 0 0
       0 1 0] # matrix of negative conections

p = [alpha, gamma, K_pos, K_neg, pos, neg]

function circ_ode!(du, u, p, t)

    alpha = p[1]
    gamma = p[2]
    K_pos = p[3]
    K_neg = p[4]
    pos = p[5]
    neg = p[6]    

    pos_act = (1 ./ (1 .+ (K_pos ./ u)'.^2))
    pos_act[pos .== 0] .= 1
    pos_act_vec = prod(pos_act, dims = 2)

    neg_act = (1 ./ (1 .+ (u ./ K_neg)'.^2))
    neg_act[neg .== 0] .= 1
    neg_act_vec = prod(neg_act, dims = 2)  

    du .= (alpha .* pos_act_vec .* neg_act_vec) .- (gamma .* u)
end

u0 = [0.1,0.2,0.1]
tspan = (0.0, 100.0)
prob = ODEProblem(circ_ode!, u0, tspan)

sol = solve(prob, p=p)
plot(sol.t, sol[1,:])