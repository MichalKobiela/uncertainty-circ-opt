using DifferentialEquations
using Plots
using DataFrames
using Random

tr =  2500 # transcription rate
tl = 1 # translation rate
d_rna = 3 # degradation
d_p = 3/5 # degradation

k=1

function repressilator_ode(du, u, p, t)
    x_rna = u[1]
    x_p = u[2] 
    y_rna = u[3] 
    y_p = u[4]
    z_rna = u[5]
    z_p = u[6]

    tr = 100
    tl = 1
    d_rna = p[1]
    d_p = d_rna/4
    K = 3
    h = p[2]

    beta_1 = 0.1
    beta_2 = 0.15
    beta_3 = 0.1


    du[1] = tr/(1 + abs((z_p/K))^h) + beta_1 - d_rna*x_rna # mRNA_X
    du[2] = tl*x_rna - d_p*x_p # X
    du[3] = tr/(1 + abs((x_p/K))^h) + beta_2 - d_rna*y_rna # mRNA_Y
    du[4] = tl*y_rna - d_p*y_p # Y
    du[5] = tr/(1 + abs((y_p/K))^h) + + beta_3 - d_rna*z_rna # mRNA_Z
    du[6] = tl*z_rna - d_p*z_p # Z
end

init_cond = zeros(6)
p = [1.0,2.0]
osc_prob = ODEProblem(repressilator_ode,init_cond,(0.0, 1000.0),p)

sol = solve(osc_prob, Euler(), dt = 1)

plot(sol[2,800:1000])