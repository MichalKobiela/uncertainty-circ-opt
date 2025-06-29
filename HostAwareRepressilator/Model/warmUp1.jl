using OrdinaryDiffEq

# Native parameters
thetar = 426.8693338968694
k_cm = 0.005990373118888
nr = 7549.0
gmax = 1260.0
cl = 0
nume0 = 4.139172187824451
s0 = 1.0e4
vm = 5800.0
Km = 1.0e3
numr0 = 929.9678874564831
nx = 300.0
kq = 1.522190403737490e+05
Kp = 180.1378030928276
vt = 726.0
nump0 = 0.0
numq0 = 948.9349882947897
Kt = 1.0e3
nq = 4
aatot = 1.0e8
ns = 0.5
thetax = 4.379733394834643

# Define rate constants
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

# Define the differential equations as a function
function ribonew_mc2_odes!(dy, y, p, t)
    y = max.(y, 0)
    rates = p[1:5]
    parameters = p[6:end]
    
    b, dm, kb, ku, f = rates
    thetar, k_cm, nr, gmax, cl, nume0, s0, vm, Km, numr0, nx, kq, Kp, vt, nump0, numq0, Kt, nq, aatot, ns, thetax = parameters
    
    rmr, em, rmp, rmq, rmt, et, rmm, zmm, zmr, zmp, zmq, zmt, mt, mm, q, p, si, mq, mp, mr, r, a = y
    
    Kg = gmax / Kp
    gamma = gmax * a / (Kg + a)
    ttrate = (rmq + rmr + rmp + rmt + rmm) * gamma
    lam = ttrate / aatot
    fr = nr * (r + rmr + rmp + rmt + rmm + rmq + zmr + zmp + zmt + zmm + zmq) /
         (nr * (r + rmr + rmp + rmt + rmm + rmq + zmr + zmp + zmt + zmm + zmq) + nx * (p + q + et + em))
    nucat = em * vm * si / (Km + si)
    
    dy[1] = kb * r * mr + b * zmr - ku * rmr - gamma / nr * rmr - f * rmr - lam * rmr
    dy[2] = gamma / nx * rmm - lam * em
    dy[3] = kb * r * mp + b * zmp - ku * rmp - gamma / nx * rmp - f * rmp - lam * rmp
    dy[4] = kb * r * mq + b * zmq - ku * rmq - gamma / nx * rmq - f * rmq - lam * rmq
    dy[5] = kb * r * mt + b * zmt - ku * rmt - gamma / nx * rmt - f * rmt - lam * rmt
    dy[6] = gamma / nx * rmt - lam * et
    dy[7] = kb * r * mm + b * zmm - ku * rmm - gamma / nx * rmm - f * rmm - lam * rmm
    dy[8] = f * rmm - b * zmm - lam * zmm
    dy[9] = f * rmr - b * zmr - lam * zmr
    dy[10] = f * rmp - b * zmp - lam * zmp
    dy[11] = f * rmq - b * zmq - lam * zmq
    dy[12] = f * rmt - b * zmt - lam * zmt
    dy[13] = (nume0 * a / (thetax + a)) + ku * rmt + gamma / nx * rmt - kb * r * mt - dm * mt - lam * mt
    dy[14] = (nume0 * a / (thetax + a)) + ku * rmm + gamma / nx * rmm - kb * r * mm - dm * mm - lam * mm
    dy[15] = gamma / nx * rmq - lam * q
    dy[16] = gamma / nx * rmp - lam * p
    dy[17] = (et * vt * s0 / (Kt + s0)) - nucat - lam * si
    dy[18] = (numq0 * a / (thetax + a) / (1 + (q / kq)^nq)) + ku * rmq + gamma / nx * rmq - kb * r * mq - dm * mq - lam * mq
    dy[19] = (nump0 * a / (thetax + a)) + ku * rmp + gamma / nx * rmp - kb * r * mp - dm * mp - lam * mp
    dy[20] = (numr0 * a / (thetar + a)) + ku * rmr + gamma / nr * rmr - kb * r * mr - dm * mr - lam * mr
    dy[21] = ku * rmr + ku * rmt + ku * rmm + ku * rmp + ku * rmq +
             gamma / nr * rmr + gamma / nr * rmr +
             gamma / nx * rmt + gamma / nx * rmm + gamma / nx * rmp + gamma / nx * rmq -
             kb * r * mr - kb * r * mt - kb * r * mm - kb * r * mp - kb * r * mq - lam * r
    dy[22] = ns * nucat - ttrate - lam * a
end

# Define the time span and initial conditions
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