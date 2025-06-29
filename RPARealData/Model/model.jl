using OrdinaryDiffEq
using Plots
using PairPlots

function odes_warm_up!(du, y, p, t)
    y = max.(y, 0)
    # Parameters
    alfa1, kx1, nx1, kcymRtot, beta1, alfa2, kx2, nx2, beta2, alfa4, kr, nr, beta4,
    r1, r2, alfa3, kx3, beta3 = p
    cuma = 2 * 1e-6
    # ODE equations
    du[1] = 0.02 * (alfa1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta1) *
            (alfa2 * (y[2]^nx2) / (kx2^nx2 + y[2]^nx2) + beta2) - 0.02 * r1 * y[1]

    du[2] = 0.02 * (alfa3 * (y[2]^nx2) / (kx3^nx2 + y[2]^nx2) + beta3) *
            (alfa4 / (1 + (y[1] / kr)^nr) + beta4) - 0.1 * r2 * y[2]
end


# Define the ODE system
function odes!(du, y, p, t)
    y = max.(y, 0)
    # Parameters
    alfa1, kx1, nx1, kcymRtot, beta1, alfa2, kx2, nx2, beta2, alfa4, kr, nr, beta4,
    r1, r2, alfa3, kx3, beta3, cuma = p

    # ODE equations
    du[1] = 0.02 * (alfa1 / (1 + (kcymRtot / (1 + cuma / kx1))^nx1) + beta1) *
            (alfa2 * (y[2]^nx2) / (kx2^nx2 + y[2]^nx2) + beta2) - 0.02 * r1 * y[1]

    du[2] = 0.02 * (alfa3 * (y[2]^nx2) / (kx3^nx2 + y[2]^nx2) + beta3) *
            (alfa4 / (1 + (y[1] / kr)^nr) + beta4) - 0.1 * r2 * y[2]
end



# Initial conditions


# Time span
tspan = (0.0, 10.0)  # From t = 0 to t = 50


# Parameters (from provided values in orgininal RPA work)
params_og = [
    88.0,        # alfa1
    1.28e-8,     # kx1
    2.34,        # nx1
    2.75e3,      # kcymRtot
    6.5,         # beta1
    149.0,       # alfa2
    119.0,       # kx2
    1.3,         # nx2
    3.9e-4,      # beta2
    2.2e7,       # alfa4
    0.51,        # kr
    3.2,         # nr
    3.4,         # beta4
    45.0,        # r1
    15.2,        # r2
    18.4,        # alfa3
    4006.9,      # kx3
    3.32         # beta3
]

# Optimized Parameters (from provided values in orgininal RPA work)
params = [
    83.4743,        # alfa1
    1.28e-8,     # kx1
    2.34,        # nx1
    2.60e3,      # kcymRtot
    11.9586,         # beta1
    391.1627,       # alfa2
    36.4063,       # kx2
    1.3,         # nx2
    3.9e-4,      # beta2
    8.7519e6,       # alfa4
    0.51,        # kr
    3.2,         # nr
    7.1347,         # beta4
    89.0635,        # r1
    7.0188,        # r2
    17.7437,        # alfa3
    4006.9,      # kx3
    0.6644         # beta3
]

# Define the problem
y0_warm_up = [24.0, 350.0]  # Initial values for y1 and y2

# Check the ODE system

# Worm-up with initial values
cuma = 2 * 1e-6

prob1 = ODEProblem(odes_warm_up!, y0_warm_up, tspan, params)

# Solve the ODE
sol = solve(prob1, Tsit5(), u0 = y0_warm_up)

Plots.plot(sol)

y0 =  sol[:,end]

cuma = 20 * 1e-6

prob = ODEProblem(odes!, y0, tspan, vcat(params, cuma))

sol = solve(prob, Tsit5(), u0 = y0)

# Plot the solution
Plots.plot(sol, xlabel="Time", vars = (1),ylabel="Concentration", lw=2, title="Gene Expression Dynamics", color = "blue", label = "model input = 20")

cuma = 1000 * 1e-6

prob = ODEProblem(odes!, y0, tspan, vcat(params, cuma))

sol = solve(prob, Tsit5(), u0 = y0)

Plots.plot(sol)