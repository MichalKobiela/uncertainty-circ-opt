include("../model/reprissilator_host.jl")
using StatsBase
using OrdinaryDiffEq,
      Optimization, Plots
using Optimization, OptimizationBBO

#Read the posterior
particles = CSV.read(string(@__DIR__)*"/particles.csv", DataFrame)
particles_arr = hcat(particles[1:100,1], particles[1:100,2])

# Ground truth
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
    
    t = [t0, tf]
    prob = ODEProblem(ribonew_mc2_odes!, init, tspan, [rates; parameters])
    sol = solve(prob, Rosenbrock23())
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
    
    sol = solve(prob, Rosenbrock23())
    
    y = sol
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

    meanGFP = g[end]
    meanmGFP = mg[end]
    meanrmGFP = rmg[end]
    
    hcoeff_1 = infer[1] #inference
    hcoeff_2 = infer[1] #inference
    hcoeff_3 = infer[1] #inference
    th = design[2] #design
    kf = design[1] #design
    
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

    init = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, mg3_0, mg2_0, mg1_0, et_0, rmm_0, zmm_0, zmr_0, rmg2_0, zmp_0, zmq_0, zmt_0, rmg3_0, g3_0, g2_0, g1_0, rmg1_0, mt_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]
    parameters = [cl, nume0, vm, hcoeff_1, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, th, gmax, thetax, Km, Kp, Kt, kf, numg0, kq, numr0, nump0,hcoeff_2, hcoeff_3]
    
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

function autocorrelation(signal)
    n = length(signal)
    acf = zeros(Float64, n)
    for lag in 0:n-1
        acf[lag+1] = sum(signal[1:n-lag] .* signal[lag+1:n])
    end
    return acf / maximum(acf)
end

# Function to find peaks in the autocorrelation
function freq(acf)
    for i in 2:length(acf)-1
        if acf[i] > acf[i-1] && acf[i] > acf[i+1]
            return float(i/1000)
        end
    end
    return -10
end

function loss_auto(p, desired_oscillations_num = 5)
    d = p[1]
    h = p[2]
    sol = solve(prob, EM(), dt = 0.1, p = [d,d,d,h,h,h,1000*p[3],10*p[4]])[2,1000:2000]
    return (freq(autocorrelation(sol)) - 1/desired_oscillations_num).^2
end

function rectangular_wave(length, period, tau)
    num_of_osc = length÷period
    valley = period-tau
    single_osc = vcat( zeros(valley), ones(tau))
    return repeat(single_osc, num_of_osc+1)[1:length]
  end

target = 1200 .* rectangular_wave(1000,50,50÷4)

CSV.write(string(@__DIR__)*"/target.csv",  Tables.table(target), writeheader=false)

init_freq = 0.104 # Initial frequency of the design used for inference

loss(infer,design) = (freq(autocorrelation(solve_rep(infer, design)[4000:end])) - init_freq/2).^2 * 1000
