using Pkg
Pkg.activate("./HostAwareRepressilator")

include("../model/reprissilator_host.jl")

using Distributions, StatsBase
using ProgressBars

infer = [2.0, log(2) / 4]
design = [1.0,100.0]
function solve_rep(infer, design)
    b = 0
    dm = 0.1
    kb = 1
    ku = 1.0
    f = cl * k_cm
    rates = [b, dm, kb, ku, f]

    # Define initial conditions
    rmr_0 = 0
    em_0 = 0
    rmp_0 = 0
    rmq_0 = 0
    rmt_0 = 0
    et_0 = 0
    rmm_0 = 0
    zmm_0 = 0
    zmr_0 = 0
    zmp_0 = 0
    zmq_0 = 0
    zmt_0 = 0
    mt_0 = 0
    mm_0 = 0
    q_0 = 0
    p_0 = 0
    si_0 = 0
    mq_0 = 0
    mp_0 = 0
    mr_0 = 0
    r_0 = 10.0
    a_0 = 1000.0
    init = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, et_0, rmm_0, zmm_0, zmr_0, zmp_0, zmq_0, zmt_0, mt_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]

    # Define parameters
    parameters = [thetar, k_cm, nr, gmax, cl, nume0, s0, vm, Km, numr0, nx, kq, Kp, vt, nump0, numq0, Kt, nq, aatot, ns, thetax]
    t0 = 0.0
    tf = 1e7  # Final time for simulation
    tspan = (t0, tf)
    
    # Define solver options (adjust if necessary)
    t = [t0, tf]
    prob = ODEProblem(ribonew_mc2_odes!, init, tspan, [rates; parameters])
    sol = solve(prob, Rosenbrock23())

    # Access the solution
    y = sol

    rmr = y[1, :]
    em = y[2, :]
    rmp = y[3, :]
    rmq = y[4, :]
    rmt = y[5, :]
    et = y[6, :]
    rmm = y[7, :]
    zmm = y[8, :]
    zmr = y[9, :]
    zmp = y[10, :]
    zmq = y[11, :]
    zmt = y[12, :]
    mt = y[13, :]
    mm = y[14, :]
    q = y[15, :]
    p = y[16, :]
    si = y[17, :]
    mq = y[18, :]
    mp = y[19, :]
    mr = y[20, :]
    r = y[21, :]
    a = y[22, :]


    rmr_0 = rmr[end]
    em_0 = em[end]
    rmp_0 = rmp[end]
    rmq_0 = rmq[end]
    rmt_0 = rmt[end]
    et_0 = et[end]
    rmm_0 = rmm[end]
    zmm_0 = zmm[end]
    zmr_0 = zmr[end]
    zmp_0 = zmp[end]
    zmq_0 = zmq[end]
    zmt_0 = zmt[end]
    mt_0 = mt[end]
    mm_0 = mm[end]
    q_0 = q[end]
    p_0 = p[end]
    si_0 = si[end]
    mq_0 = mq[end]
    mp_0 = mp[end]
    mr_0 = mr[end]
    r_0 = r[end]
    a_0 = a[end]

    # Randomize initial conditions for the GFP species
    meanGFP = em[end]
    meanmGFP = mm[end]
    meanrmGFP = rmm[end]

    mg_0 = meanmGFP + 0.3 * meanmGFP * randn()
    rmg_0 = meanrmGFP + 0.3 * meanrmGFP * randn()
    g_0 = meanGFP + 0.3 * meanGFP * randn()

    # Define the initial conditions array
    init = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, rmg_0, et_0, rmm_0, zmm_0, zmr_0, zmp_0, zmq_0, zmt_0, mt_0, mg_0, g_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]

    # INDUCTION PARAMETER
    numg0 = 25

    # Redefine the parameter vector for the GFP model
    parameters = [cl, nume0, vm, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, gmax, thetax, Km, Kp, Kt, numg0, kq, numr0, nump0]

    # Define rate constants
    b = 0
    dm = 0.1
    kb = 1
    ku = 1.0
    f = cl * k_cm
    dmg = log(2) / 2
    dg = log(2) / 4
    rates = [b, dm, kb, ku, f, dmg, dg]

    t0 = 0.0
    tf = 1e7  # Final time for simulation
    tspan = (t0, tf)
    
    
    # Define the ODE problem
    prob = ODEProblem(ribonew_mc2_gfp_odes!, init, tspan, [rates; parameters])
    
    # Solve using the Rosenbrock method
    sol = solve(prob, Rosenbrock23())
    
    # Extract the solution
    y = sol
    
    # Extract the individual variables from the solution
    rmr = y[1, :]
    em = y[2, :]
    rmp = y[3, :]
    rmq = y[4, :]
    rmt = y[5, :]
    rmg = y[6, :]
    et = y[7, :]
    rmm = y[8, :]
    zmm = y[9, :]
    zmr = y[10, :]
    zmp = y[11, :]
    zmq = y[12, :]
    zmt = y[13, :]
    mt = y[14, :]
    mg = y[15, :]
    g = y[16, :]
    mm = y[17, :]
    q = y[18, :]
    p = y[19, :]
    si = y[20, :]
    mq = y[21, :]
    mp = y[22, :]
    mr = y[23, :]
    r = y[24, :]
    a = y[25, :]

    rmr_0 = rmr[end]
    em_0 = em[end]
    rmp_0 = rmp[end]
    rmq_0 = rmq[end]
    rmt_0 = rmt[end]
    et_0 = et[end]
    rmm_0 = rmm[end]
    zmm_0 = zmm[end]
    zmr_0 = zmr[end]
    zmp_0 = zmp[end]
    zmq_0 = zmq[end]
    zmt_0 = zmt[end]
    mt_0 = mt[end]
    mm_0 = mm[end]
    q_0 = q[end]
    p_0 = p[end]
    si_0 = si[end]
    mq_0 = mq[end]
    mp_0 = mp[end]
    mr_0 = mr[end]
    r_0 = r[end]
    a_0 = a[end]
    
    # Randomize initial conditions for the repressilator species, mean equal to the GFP steady state
    meanGFP = g[end]
    meanmGFP = mg[end]
    meanrmGFP = rmg[end]
    
    # Extra parameters for the repressilator model
    hcoeff_1 = infer[1] ## inference
    hcoeff_2 = infer[1] ## inference
    hcoeff_3 = infer[1] ## inference
    th = design[2] # theta
    kf = design[1] ## design
    
    # Randomize initial conditions with added noise
    initmg = meanmGFP .+ 0.3 * meanmGFP .* randn(3)
    initrmg = meanrmGFP .+ 0.3 * meanrmGFP .* randn(3)
    initg = meanGFP .+ 0.3 * meanGFP .* randn(3)
    
    mg1_0 = initmg[1]
    mg2_0 = initmg[2]
    mg3_0 = initmg[3]
    rmg1_0 = initrmg[1]
    rmg2_0 = initrmg[2]
    rmg3_0 = initrmg[3]
    g1_0 = initg[1]
    g2_0 = initg[2]
    g3_0 = initg[3]
    
    # Combine all initial conditions into a single vector
    init = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, mg3_0, mg2_0, mg1_0, et_0, rmm_0, zmm_0, zmr_0, rmg2_0, zmp_0, zmq_0, zmt_0, rmg3_0, g3_0, g2_0, g1_0, rmg1_0, mt_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]

    
    # Redefine parameter vector for the repressilator model
    parameters = [cl, nume0, vm, hcoeff_1, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, th, gmax, thetax, Km, Kp, Kt, kf, numg0, kq, numr0, nump0,hcoeff_2, hcoeff_3]
    
    # Simulate repressilator model with a final time tf
    tf = 5e3
    
    
    b = 0;
    dm = 0.1;
    kb = 1;
    ku = 1.0;
    f = cl * k_cm;
    dmg = log(2) / 2;
    dg = infer[2] # inference
    rates = [b, dm, kb, ku, f, dmg, dg];

    u0 = init
    tspan = (0.0, tf)
    p = vcat(rates, parameters)
    prob = ODEProblem(repressilator_odes!, u0, tspan, p)
    sol = solve(prob, Rosenbrock23(), saveat=1)

    return sol[18, :]
end

function solve_points(infer)
    return solve_rep(infer, [1,100])[4000:20:end]
end



lk = ReentrantLock()
using Main.Threads
function ABC_SMC(N, epsilon_T, alpha, prior, perturb, simulate, distance, observed)
    epsilon = Inf
    t = 0
    particles = Vector{Vector{Float64}}()
    weights = Vector{Float64}()
    distances = Vector{Float64}()
    
    for j=tqdm(1:10) #epsilon > epsilon_T
        new_particles = Vector{Vector{Float64}}(undef, N)
        new_weights = Vector{Float64}(undef, N)
        new_distances = Vector{Float64}(undef, N)
        
        @threads for i in 1:N
            safety_count = 0
            while true
                safety_count += 1
                if safety_count > 10_000
                    # avoid infinite loop
                    println("Safety count exceeded")
                    println("epsilon: ", epsilon)
                    return particles, weights, epsilon
                end
                if t == 0
                    theta_starstar = prior()
                else
                    idx = sample(1:N, Weights(weights))
                    theta_star = particles[idx]
                    theta_starstar = perturb(theta_star)
                end
                
                if pdf(Uniform(0,1), theta_starstar[1]) == 0
                    continue
                end
                if pdf(Uniform(0,1), theta_starstar[2]) == 0
                    continue
                end
                
                x_star = simulate(theta_starstar)
                d_star = distance(x_star, observed)
                
                if d_star <= epsilon

                        lock(lk) do
                            new_particles[i] = theta_starstar
                            new_distances[i] = d_star    
                        end
                    break
                end
            end
        end
        # For uniform prior and uniform perturbation kernel the weights are constant
        # If the prior and perturbation kernel are changed, the weights should be computed accordingly
        new_weights = ones(N)
        new_weights /= sum(new_weights)
        epsilon = max(quantile(new_distances, alpha), epsilon_T) # Update epsilon
        particles = new_particles
        weights = new_weights
        distances = new_distances
        
        t += 1
    end
    
    return particles, weights, epsilon
end

# Example usage
prior = () -> [rand(Uniform(0, 1)), rand(Uniform(0, 1))]
perturb = x -> x .+ rand(Uniform(-0.1, 0.1), size(x))
simulate = theta -> solve_points(vcat(1 + 9*theta[1], 10 * theta[2]))

function distance_rmse(x, y)
    return sqrt(mean((x - y) .^ 2))
end


# observed_data = rand(Normal(0, 0.1),100) .+ 0.6

num_sequences = 1  # Number of observed data sequences
observed_data = y

N = 100
epsilon_T = 200
alpha = 0.3

# particles, weights, epsilon = ABC_SMC(N, epsilon_T, alpha, prior, perturb, simulate, distance, observed_data)

particles_1, weights_1, epsilon_1 = ABC_SMC(N, epsilon_T, alpha, prior, perturb, simulate, distance_rmse, observed_data)
particles_2, weights_2, epsilon_2 = ABC_SMC(N, epsilon_T, alpha, prior, perturb, simulate, distance_rmse, observed_data)
particles_3, weights_3, epsilon_3 = ABC_SMC(N, epsilon_T, alpha, prior, perturb, simulate, distance_rmse, observed_data)

particles_1_arr = hcat(particles_1...)
particles_2_arr = hcat(particles_2...)
particles_3_arr = hcat(particles_3...)

particles_1_arr = particles_1_arr.* [9, 10] .+ [1, 0]
particles_2_arr = particles_2_arr.* [9, 10] .+ [1, 0]
particles_3_arr = particles_3_arr.* [9, 10] .+ [1, 0]

all_particles_arr = hcat(particles_1_arr, particles_2_arr, particles_3_arr)
#save to csv

using CSV, DataFrames
CSV.write("particles.csv",  Tables.table(all_particles_arr'), header = [:hcoeff_1, :dg])

scatter(particles_1_arr[1, :], particles_1_arr[2, :], label = "particles 1", color = "red")
scatter!([infer[1]], [infer[2]])

using Serialization
open("particles_1.jls", "w") do f
    serialize(f, particles_1)
end

open("particles_2.jls", "w") do f
    serialize(f, particles_2)
end

open("particles_3.jls", "w") do f
    serialize(f, particles_3)
end

