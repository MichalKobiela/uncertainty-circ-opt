using OrdinaryDiffEq
using Plots

alpha_1 = 100
alpha_2 = 100
beta_RA = 0.0
beta_AB = 0.0
beta_BA = 0.0
beta_BB = 0.0
gamma_A = 1
gamma_B = 1
n_RA = 1
n_BA = 1
n_AB = 1
n_BB = 1
K_IR = 1 
K_TF = 1
K_BA = 1
K_AB = 1
K_BB = 1

p=[beta_RA
beta_AB
beta_BA 
beta_BB ]

function rpa_ode!(du, u, p, t)
    beta_RA = p[1]
    beta_AB = p[2]
    beta_BA = p[3]
    beta_BB = p[4]

    input = t<50 ? 1 : 10
    
    A = u[1]
    B = u[2]

    du[1] = alpha_1*(1/(1+(K_TF/(1+(input/K_IR)))^n_RA) + beta_RA)* (1/((K_BA/B)^n_BA+1) + beta_BA )   - gamma_A*A
    du[2] = alpha_2*(1/(1+(A/K_AB))^n_AB + beta_AB) * (1/((K_BB/B)^n_BB+1) + beta_BB)  - gamma_B*B
end


u0 = [1.0; 1.0]
tspan = (0.0, 100.0)
prob = ODEProblem(rpa_ode!, u0, tspan)

sol = solve(prob,Rosenbrock23(), p=p)

function rpa_ode_opt!(du, u, p, t)
    beta_RA = p[1]
    beta_AB = p[2]
    beta_BA = p[3]
    beta_BB = p[4]
    alpha_1 = p[5]
    alpha_2 = p[6]
    K_BA = p[7]
    K_AB = p[8]
    K_BB = p[9]

    input = t<50 ? 1 : p[10]
    
    A = u[1]
    B = u[2]

    du[1] = alpha_1*(1/(1+(K_TF/(1+(input/K_IR)))^n_RA) + beta_RA)* (1/((K_BA/B)^n_BA+1) + beta_BA )   - gamma_A*A
    du[2] = alpha_2*(1/(1+(A/K_AB))^n_AB + beta_AB) * (1/((K_BB/B)^n_BB+1) + beta_BB)  - gamma_B*B
end