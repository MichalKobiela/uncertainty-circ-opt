Code for the Risk-averse optimization of genetic circuits under uncertainty by Michal Kobiela, Diego Oyarzún and Michael Gutmann.

Licence: CC-BY 4.0

The repository structure is as follows:
```
├───Repressilator
│   ├───Design
│   │       optimization_prior.jl - Acquisition of prior optimization samples
│   │       setup_opt.jl - Setup of optimization
│   │       thompson.jl - Acquisition of Thompson samples
│   │
│   ├───Evaluation
│   │       eval.jl - Evaluation of the Thompson samples
│   │       eval_prior.jl - Evaluation of prior optimization samples
│   │       exemplar.jl - Evaluation of exemplar designs
│   │
│   ├───Inference
│   │       smc.jl - SMC ABC inference
│   │
│   └───Model
│           repressilator_model.jl - Mathematical model of the Repressilator
│
└───RPA
    ├───Design
    │       optimization_prior.jl - Acquisition of prior optimization samples
    │       thompson.jl -  Acquisition of Thompson samples
    │
    ├───Evaluation
    │       eval.jl - Evaluation of the Thompson samples
    │       eval_prior.jl -  Evaluation of prior optimization samples
    │       exemplar_designs.jl -  Evaluation of exemplar designs
    │
    ├───Inference
    │       mcmc.jl - MCMC inference (NUTS)
    │       save_posterior.jl - Script saving posterior to csv file
    └───Model
            rpa_ode.jl - Mathematical model of the RPA circuit
```
