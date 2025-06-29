include("warmUp1.jl")

# Extracting the last elements of arrays for initial conditions
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

# Define the ODE system as a function
function ribonew_mc2_gfp_odes!(dydt, y, p, t)
    y = max.(y, 0)
    # Extract rate constants and parameters
    rates = p[1:7]    # First 7 are the rates
    parameters = p[8:end]  # Rest are the parameters

    b, dm, kb, ku, f, dmg, dg = rates
    cl, nume0, vm, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, gmax, thetax, Km, Kp, Kt, numg0, kq, numr0, nump0 = parameters

    # Extract variables from y
    rmr, em, rmp, rmq, rmt, rmg, et, rmm, zmm, zmr, zmp, zmq, zmt, mt, mg, g, mm, q, p, si, mq, mp, mr, r, a = y

    # Intermediate variables
    Kg = gmax / Kp
    gamma = gmax * a / (Kg + a)
    ttrate = (rmq + rmr + rmp + rmt + rmm + rmg) * gamma
    lam = ttrate / aatot
    fr = nr * (r + rmr + rmp + rmt + rmm + rmq + rmg + zmr + zmp + zmt + zmm + zmq) /
         (nr * (r + rmr + rmp + rmt + rmm + rmq + rmg + zmr + zmp + zmt + zmm + zmq) + nx * (p + q + et + em + g))
    nucat = em * vm * si / (Km + si)

    # Define the system of ODEs
    dydt[1] = +kb * r * mr + b * zmr - ku * rmr - gamma / nr * rmr - f * rmr - lam * rmr # rmr
    dydt[2] = +gamma / nx * rmm - lam * em # em
    dydt[3] = +kb * r * mp + b * zmp - ku * rmp - gamma / nx * rmp - f * rmp - lam * rmp # rmp
    dydt[4] = +kb * r * mq + b * zmq - ku * rmq - gamma / nx * rmq - f * rmq - lam * rmq # rmq
    dydt[5] = +kb * r * mt + b * zmt - ku * rmt - gamma / nx * rmt - f * rmt - lam * rmt # rmt
    dydt[6] = +kb * r * mg - ku * rmg - gamma / nx * rmg # rmg
    dydt[7] = +gamma / nx * rmt - lam * et # et
    dydt[8] = +kb * r * mm + b * zmm - ku * rmm - gamma / nx * rmm - f * rmm - lam * rmm # rmm
    dydt[9] = +f * rmm - b * zmm - lam * zmm # zmm
    dydt[10] = +f * rmr - b * zmr - lam * zmr # zmr
    dydt[11] = +f * rmp - b * zmp - lam * zmp # zmp
    dydt[12] = +f * rmq - b * zmq - lam * zmq # zmq
    dydt[13] = +f * rmt - b * zmt - lam * zmt # zmt
    dydt[14] = +(nume0 * a / (thetax + a)) + ku * rmt + gamma / nx * rmt - kb * r * mt - dm * mt - lam * mt # mt
    dydt[15] = +(numg0 * a / (thetax + a)) + ku * rmg + gamma / nx * rmg - kb * r * mg - dmg * mg - lam * mg # mg
    dydt[16] = +gamma / nx * rmg - dg * g - lam * g # g
    dydt[17] = +(nume0 * a / (thetax + a)) + ku * rmm + gamma / nx * rmm - kb * r * mm - dm * mm - lam * mm # mm
    dydt[18] = +gamma / nx * rmq - lam * q # q
    dydt[19] = +gamma / nx * rmp - lam * p # p
    dydt[20] = +(et * vt * s0 / (Kt + s0)) - nucat - lam * si # si
    dydt[21] = +(numq0 * a / (thetax + a) / (1 + (q / kq)^nq)) + ku * rmq + gamma / nx * rmq - kb * r * mq - dm * mq - lam * mq # mq
    dydt[22] = +(nump0 * a / (thetax + a)) + ku * rmp + gamma / nx * rmp - kb * r * mp - dm * mp - lam * mp # mp
    dydt[23] = +(numr0 * a / (thetar + a)) + ku * rmr + gamma / nr * rmr - kb * r * mr - dm * mr - lam * mr # mr
    dydt[24] = +ku * rmr + ku * rmt + ku * rmm + ku * rmp + ku * rmq + gamma / nr * rmr + gamma / nr * rmr + gamma / nx * rmt + gamma / nx * rmm + gamma / nx * rmp + gamma / nx * rmq + ku * rmg + gamma / nx * rmg - kb * r * mr - kb * r * mt - kb * r * mm - kb * r * mp - kb * r * mq - lam * r - kb * r * mg # r
    dydt[25] = +ns * nucat - ttrate - lam * a # a
end

# Define the time span
t0 = 0.0
tf = 1e7  # Final time for simulation
tspan = (t0, tf)


# Define the ODE problem
prob = ODEProblem(ribonew_mc2_gfp_odes!, init, tspan, [rates; parameters])

# Solve using the Rosenbrock method
sol = solve(prob, Rodas4P())

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