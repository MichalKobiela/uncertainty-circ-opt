using Distributed
using Serialization
@everywhere using BayesianOptimization, GaussianProcesses, Distributions, Distributed
@everywhere using SharedArrays
@everywhere include("setup_opt.jl")

println("setup done")

result = SharedArray{Float64}(100,10,2)

samples = SharedArray{Float64}(100,10,2)


using Serialization
file = open(string(@__DIR__)*"/samples_prior.jls", "w")
serialize(file, Array(samples))
close(file)

print(string(@__DIR__)*"/samples_prior.jls")
println(stderr,string(@__DIR__)*"/samples_prior.jls")
@time @sync @distributed for i=1:100
    sample = rand(2)*10
    samples[i,1,:] = sample
    for j=1:1
        f_l(x) = loss_iter(x,sample)   
        model = ElasticGPE(2,                           
                        mean = MeanConst(0.),         
                        kernel = SEArd([0., 0.], 5.),
                        logNoise = 0.,
                        capacity = 3000)             
        set_priors!(model.mean, [Normal(1, 2)])

        modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-4, 3],       
                                        kernbounds = [[-1, -1, 0], [4, 4, 10]],  
                                        maxeval = 40)
        opt = BOpt(f_l,
                model,
                UpperConfidenceBound(),                   
                modeloptimizer,                        
                [100.0, 0.01], [1000.0, 10.0],                   
                repetitions = 3,                          
                maxiterations = 1000,                     
                sense = Min,                              
                acquisitionoptions = (method = :LD_LBFGS,
                                        restarts = 10, 
                                        maxtime = 0.1,
                                        maxeval = 10), 
                    verbosity = Progress)

        result[i,j,:] = boptimize!(opt)[4]
    end
end

println(result)

file = open(string(@__DIR__)*"/designs_prior.jls", "w")
serialize(file, Array(result))
close(file)

file = open(string(@__DIR__)*"/samples_prior.jls", "w")
serialize(file, Array(samples))
close(file)

