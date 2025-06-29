include("warmUp2.jl")
using Plots
using Random
using DataFrames
using CSV
Random.seed!(0)
# Set initial conditions for the GFP species and other variables
rmr_0 = rmr[end]
em_0 = em[end]
rmp_0 =  rmp[end]
rmq_0 =  rmq[end]
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
p_0 =   p[end]
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
hcoeff = 2 ## inference
th = 100
kf = 1 ## design

# Randomize initial conditions with added noise
initmg = meanmGFP .+ 0.3 * meanmGFP .* randn(3)
initrmg = meanrmGFP .+ 0.3 * meanrmGFP .* randn(3)
initg = meanGFP .+ 0.3 * meanGFP .* randn(3)
mg1_0 =  initmg[1]
mg2_0 =  initmg[2]
mg3_0 =  initmg[3]
rmg1_0 = initrmg[1]
rmg2_0 = initrmg[2]
rmg3_0 = initrmg[3]
g1_0 =  initg[1]
g2_0 = initg[2]
g3_0 = initg[3]
# Combine all initial conditions into a single vector
init = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, mg3_0, mg2_0, mg1_0, et_0, rmm_0, zmm_0, zmr_0, rmg2_0, zmp_0, zmq_0, zmt_0, rmg3_0, g3_0, g2_0, g1_0, rmg1_0, mt_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]


numg0 = 25 # Design

# Redefine parameter vector for the repressilator model
parameters = [cl, nume0, vm, hcoeff, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, th, gmax, thetax, Km, Kp, Kt, kf, numg0, kq, numr0, nump0]

# Simulate repressilator model with a final time tf
tf = 5e3


b = 0;
dm = 0.1;
kb = 1;
ku = 1.0;
f = cl * k_cm;
dmg = log(2) / 2;
dg = log(2) / 4; # inference
rates = [b, dm, kb, ku, f, dmg, dg];

function repressilator_odes!(dydt, y, p, t)
    # assign at the begining of the function

    y = max.(y, 0)
    rates = p[1:7]
    parameters = p[8:end]
    
    b = rates[1]
    dm = rates[2]
    kb = rates[3]
    ku = rates[4]
    f = rates[5]
    dmg = rates[6]
    dg = rates[7]
    
    cl = parameters[1]
    nume0 = parameters[2]
    vm = parameters[3]
    hcoeff = parameters[4]
    vt = parameters[5]
    aatot = parameters[6]
    s0 = parameters[7]
    nx = parameters[8]
    numq0 = parameters[9]
    nq = parameters[10]
    nr = parameters[11]
    ns = parameters[12]
    thetar = parameters[13]
    k_cm = parameters[14]
    th = parameters[15]
    gmax = parameters[16]
    thetax = parameters[17]
    Km = parameters[18]
    Kp = parameters[19]
    Kt = parameters[20]
    kf = parameters[21]
    numg0 = parameters[22]
    kq = parameters[23]
    numr0 = parameters[24]
    nump0 = parameters[25]
    
    rmr = y[1]
    em = y[2]
    rmp = y[3]
    rmq = y[4]
    rmt = y[5]
    mg3 = y[6]
    mg2 = y[7]
    mg1 = y[8]
    et = y[9]
    rmm = y[10]
    zmm = y[11]
    zmr = y[12]
    rmg2 = y[13]
    zmp = y[14]
    zmq = y[15]
    zmt = y[16]
    rmg3 = y[17]
    g3 = y[18]
    g2 = y[19]
    g1 = y[20]
    rmg1 = y[21]
    mt = y[22]
    mm = y[23]
    q = y[24]
    p = y[25]
    si = y[26]
    mq = y[27]
    mp = y[28]
    mr = y[29]
    r = y[30]
    a = y[31]

    Kg= gmax/Kp;
	gamma= gmax*a/(Kg+a);
	ttrate= (rmq+rmr+rmp+rmt+rmm+rmg1+rmg2+rmg3)*gamma;
	lam= ttrate/aatot;
	fr= nr*(r+rmr+rmp+rmt+rmm+rmq+rmg1+rmg2+rmg3+zmr+zmp+zmt+zmm+zmq)/(nr*(r+rmr+rmp+rmt+rmm+rmq+rmg1+rmg2+rmg3+zmr+zmp+zmt+zmm+zmq)+nx*(p+q+et+em+g1+g2+g3));
	nucat= em*vm*si/(Km+si);
	ff1= kf/(1+(g3/th)^hcoeff);
	ff2= kf/(1+(g1/th)^hcoeff);
	ff3= kf/(1+(g2/th)^hcoeff);

	dydt[1]= +kb*r*mr+b*zmr-ku*rmr-gamma/nr*rmr-f*rmr-lam*rmr;
	dydt[2]= +gamma/nx*rmm-lam*em;
	dydt[3]= +kb*r*mp+b*zmp-ku*rmp-gamma/nx*rmp-f*rmp-lam*rmp;
	dydt[4]= +kb*r*mq+b*zmq-ku*rmq-gamma/nx*rmq-f*rmq-lam*rmq;
	dydt[5]= +kb*r*mt+b*zmt-ku*rmt-gamma/nx*rmt-f*rmt-lam*rmt;
	dydt[6]= +(ff3*numg0*a/(thetax+a))+ku*rmg3+gamma/nx*rmg3-kb*r*mg3-dmg*mg3-lam*mg3;
	dydt[7]= +(ff2*numg0*a/(thetax+a))+ku*rmg2+gamma/nx*rmg2-kb*r*mg2-dmg*mg2-lam*mg2;
	dydt[8]= +(ff1*numg0*a/(thetax+a))+ku*rmg1+gamma/nx*rmg1-kb*r*mg1-dmg*mg1-lam*mg1;
	dydt[9]= +gamma/nx*rmt-lam*et;
	dydt[10]= +kb*r*mm+b*zmm-ku*rmm-gamma/nx*rmm-f*rmm-lam*rmm;
	dydt[11]= +f*rmm-b*zmm-lam*zmm;
	dydt[12]= +f*rmr-b*zmr-lam*zmr;
	dydt[13]= +kb*r*mg2-ku*rmg2-gamma/nx*rmg2-lam*rmg2;
	dydt[14]= +f*rmp-b*zmp-lam*zmp;
	dydt[15]= +f*rmq-b*zmq-lam*zmq;
	dydt[16]= +f*rmt-b*zmt-lam*zmt;
	dydt[17]= +kb*r*mg3-ku*rmg3-gamma/nx*rmg3-lam*rmg3;
	dydt[18]= +gamma/nx*rmg3-dg*g3-lam*g3;
	dydt[19]= +gamma/nx*rmg2-dg*g2-lam*g2;
	dydt[20]= +gamma/nx*rmg1-dg*g1-lam*g1;
	dydt[21]= +kb*r*mg1-ku*rmg1-gamma/nx*rmg1-lam*rmg1;
	dydt[22]= +(nume0*a/(thetax+a))+ku*rmt+gamma/nx*rmt-kb*r*mt-dm*mt-lam*mt;
	dydt[23]= +(nume0*a/(thetax+a))+ku*rmm+gamma/nx*rmm-kb*r*mm-dm*mm-lam*mm;
	dydt[24]= +gamma/nx*rmq-lam*q;
	dydt[25]= +gamma/nx*rmp-lam*p;
	dydt[26]= +(et*vt*s0/(Kt+s0))-nucat-lam*si;
	dydt[27]= +(numq0*a/(thetax+a)/(1+(q/kq)^nq))+ku*rmq+gamma/nx*rmq-kb*r*mq-dm*mq-lam*mq;
	dydt[28]= +(nump0*a/(thetax+a))+ku*rmp+gamma/nx*rmp-kb*r*mp-dm*mp-lam*mp;
	dydt[29]= +(numr0*a/(thetar+a))+ku*rmr+gamma/nr*rmr-kb*r*mr-dm*mr-lam*mr;
	dydt[30]= +ku*rmr+ku*rmt+ku*rmm+ku*rmp+ku*rmq+gamma/nr*rmr+gamma/nr*rmr+gamma/nx*rmt+gamma/nx*rmm+gamma/nx*rmp+gamma/nx*rmq+ku*rmg1+ku*rmg2+ku*rmg3+gamma/nx*rmg1+gamma/nx*rmg2+gamma/nx*rmg3-kb*r*mr-kb*r*mt-kb*r*mm-kb*r*mp-kb*r*mq-lam*r-kb*r*mg1-kb*r*mg2-kb*r*mg3;
	dydt[31]= +ns*nucat-ttrate-lam*a;
    
    # Kg = gmax/Kp
    # gamma = gmax*a/(Kg+a)
    # ttrate = (rmq+rmr+rmp+rmt+rmm+rmg1+rmg2+rmg3)*gamma
    # lam = ttrate/aatot
    # fr = nr*(r+rmr+rmp+rmt+rmm+rmq+rmg1+rmg2+rmg3+zmr+zmp+zmt+zmm+zmq)/(nr*(r+rmr+rmp+rmt+rmm+rmq+rmg1+rmg2+rmg3+zmr+zmp+zmt+zmm+zmq)+nx*(p+q+et+em+g1+g2+g3))
    # nucat = em*vm*si/(Km+si)
    # ff1 = kf/(1+(g3/th)^hcoeff)
    # ff2 = kf/(1+(g1/th)^hcoeff)
    # ff3 = kf/(1+(g2/th)^hcoeff)
    
    # dydt[1] = +kb*r*mr+b*zmr-ku*rmr-gamma/nr*rmr-f*rmr-lam*rmr # rmr
    # dydt[2] = +gamma/nx*rmm-lam*em # em
    # dydt[3] = +kb*r*mp+b*zmp-ku*rmp-gamma/nx*rmp-f*rmp-lam*rmp # rmp
    # dydt[4] = +kb*r*mq+b*zmq-ku*rmq-gamma/nx*rmq-f*rmq-lam*rmq # rmq
    # dydt[5] = +kb*r*mt+b*zmt-ku*rmt-gamma/nx*rmt-f*rmt-lam*rmt # rmt

    # dydt[6] = +(ff3*numg0*a/(thetax+a))+ku*rmg3+gamma/nx*rmg3-kb*r*mg3-dmg*mg3-lam*mg3 # mg3
    # dydt[7] = +(ff2*numg0*a/(thetax+a))+ku*rmg2+gamma/nx*rmg2-kb*r*mg2-dmg*mg2-lam*mg2 # mg2
    # dydt[8] = +(ff1*numg0*a/(thetax+a))+ku*rmg1+gamma/nx*rmg1-kb*r*mg1-dmg*mg1-lam*mg1 # mg1
    
    # dydt[9] = +gamma/nx*rmt-lam*et # et
    # dydt[10] = +kb*r*mm+b*zmm-ku*rmm-gamma/nx*rmm-f*rmm-lam*rmm # rmm
    # dydt[11] = +f*rmm-b*zmm-lam*zmm # zmm
    # dydt[12] = +f*rmr-b*zmr-lam*zmr # zmr

    # dydt[13] = +kb*r*mg2-ku*rmg2-gamma/nx*rmg2-lam*rmg2 # rmg2

    # dydt[14] = +f*rmp-b*zmp-lam*zmp # zmp
    # dydt[15] = +f*rmq-b*zmq-lam*zmq # zmq
    # dydt[16] = +f*rmt-b*zmt-lam*zmt # zmt

    # dydt[17] = +kb*r*mg3-ku*rmg3-gamma/nx*rmg3-lam*rmg3 # rmg3

    # dydt[18] = +gamma/nx*rmg3-dg*g3-lam*g3 # g3
    # dydt[19] = +gamma/nx*rmg2-dg*g2-lam*g2 # g2
    # dydt[20] = +gamma/nx*rmg1-dg*g1-lam*g1 # g1

    # dydt[21] = +kb*r*mg1-ku*rmg1-gamma/nx*rmg1-lam*rmg1 # rmg1
    # dydt[22] = +(nume0*a/(thetax+a))+ku*rmt+gamma/nx*rmt-kb*r*mt-dm*mt-lam*mt # mt
    # dydt[23] = +(nume0*a/(thetax+a))+ku*rmm+gamma/nx*rmm-kb*r*mm-dm*mm-lam*mm # mm
    # dydt[24] = +gamma/nx*rmq-lam*q # q
    # dydt[25] = +gamma/nx*rmp-lam*p # p
    # dydt[26] = +(et*vt*s0/(Kt+s0))-nucat-lam*si # si
    # dydt[27] = +(numq0*a/(thetax+a)/(1+(q/kq)^nq))+ku*rmq+gamma/nx*rmq-kb*r*mq-dm*mq-lam*mq # mq
    # dydt[28] = +(nump0*a/(thetax+a))+ku*rmp+gamma/nx*rmp-kb*r*mp-dm*mp-lam*mp # mp
    # dydt[29] = +(numr0*a/(thetar+a))+ku*rmr+gamma/nr*rmr-kb*r*mr-dm*mr-lam*mr # mr
    # dydt[30] = +ku*rmr+ku*rmt+ku*rmm+ku*rmp+ku*rmq+gamma/nr*rmr+gamma/nx*rmt-lam*r # r
    # dydt[31] = +ns*nucat-ttrate-lam*a # a
end

u0 = init
tspan = (0.0, tf)
p = vcat(rates, parameters)
prob = ODEProblem(repressilator_odes!, u0, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=1)
y = sol 
rmr = y[1, :]
em = y[2, :]
rmp = y[3, :]
rmq = y[4, :]
rmt = y[5, :]
mg3 = y[6, :]
mg2 = y[7, :]
mg1 = y[8, :]
et = y[9, :]
rmm = y[10, :]
zmm = y[11, :]
zmr = y[12, :]
rmg2 = y[13, :]
zmp = y[14, :]
zmq = y[15, :]
zmt = y[16, :]
rmg3 = y[17, :]
g3 = y[18, :]
g2 = y[19, :]
g1 = y[20, :]
rmg1 = y[21, :]
mt = y[22, :]
mm = y[23, :]
q = y[24, :]
p = y[25, :]
si = y[26, :]
mq = y[27, :]
mp = y[28, :]
mr = y[29, :]
r = y[30, :]
a = y[31, :]

plot(sol.t, g1, label = "solution")

plot(sol.t[4000:end],g1[4000:end], label = "solution")
x = sol.t[4000:20:end]
y = g1[4000:20:end] .+ 50*randn(length(x)) 

scatter!(x,y, label = "data")
plot!(ylabel = "gene1", xlabel = "time")

# Save the data
data = DataFrame(t = sol.t[4000:20:end], g1 = y)
CSV.write("data.csv", data)

traj = DataFrame(t = sol.t[4000:end], g1 = g1[4000:end]) 
CSV.write("traj_infer.csv", traj)