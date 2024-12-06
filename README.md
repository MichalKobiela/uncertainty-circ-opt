Code for the Risk-averse optimization of genetic circuits under uncertainty (https://www.biorxiv.org/content/10.1101/2024.11.13.623219v1) by Michal Kobiela, Diego Oyarzún and Michael Gutmann.

Synthetic biology aims to engineer biological systems with specified functions. This requires navigating an extensive design space, which is challenging to achieve with wet lab experiments alone. To expedite the design process, mathematical modelling is typically employed to predict circuit function in silico ahead of implementation, which when coupled with computational optimization can be used to automatically identify promising designs. However, circuit models are inherently inaccurate which can result in sub-optimal or non-functional in vivo performance. To mitigate this issue, here we propose to combine Bayesian inference, Thompson sampling, and risk management to find optimal circuit designs. Our approach employs data from non-functional designs to estimate the distribution of the model parameters and then employs risk-averse optimization to select design parameters that are expected to perform well given parameter uncertainty and biomolecular noise. We illustrate the approach by designing robust adaptation circuits and genetic oscillators with a prescribed frequency. The proposed approach provides a novel methodology for the design of robust genetic circuitry.

Toml files with environments are provided.

Licence: CC-BY 4.0

If find the code helpful, please cite our work:

Kobiela, Michal, Diego A. Oyarzun, and Michael U. Gutmann. "Risk-averse optimization of genetic circuits under uncertainty." 

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
