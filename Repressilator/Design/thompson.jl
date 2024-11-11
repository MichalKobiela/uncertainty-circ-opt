using Distributed
@everywhere using BayesianOptimization, GaussianProcesses, Distributions, Distributed
@everywhere using SharedArrays
@everywhere include("setup_opt.jl")

println("setup done")

result = SharedArray{Float64}(100,1,2)

@time @sync @distributed for i=1:100
    sample = samples[i,:]
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
                                        maxeval = 1000), 
                    verbosity = Progress)

        result[i,j,:] = boptimize!(opt)[4]
    end
end


using Serialization
file = open(string(@__DIR__)*"/designs.jls", "w")
serialize(file, Array(result))
close(file)