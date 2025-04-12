# adaFuncCI
Code to support *Confidence intervals for functionals in constrained inverse problems via data-adaptive sampling-based calibration* (https://arxiv.org/abs/2502.02674).
A python package containing all code to generate intervals is found in `./adaFuncCI`.

## Installation
Simply run `pip install .` from the top of the directory. Necessary python packages are included in `requirements.txt`, however, the primary non-standard packages to fully use this package are:
1. `cvxpy`
2. `numpy`
3. `polytopewalk` [github](https://github.com/yuachen/polytopewalk)
4. `sklearn`
5. `scipy`

## Primary Components
The jupyter notebook `pilot_study_unfolding.ipynb` provides a step-by-step walk through of each component. This notebook shows how to use the primary pieces of the package:
1. Creating polytope samplers (`from adaFuncCI.sample import polytopeSampler`)
2. Sampling from LLR null distributions (`from adaFuncCI.sample import nullTestStatSampler`)
3. Fitting quantile regression (`from adaFuncCI.max_quantile import maxQuantileQR`)
4. Computing adaOSB confidence interval (`from adaFuncCI.optimize import osb_int`)

## Examples
We provide three examples:
1. `2d_example.ipynb` -- demonstrates the necessary components of the method on an easily understood example.
2. `3d_example.ipynb` -- demonstrates the correct coverage of the intervals in a scenario when OSB intervals do not achieve nominal coverage.
3. `pilot_study_unfolding.ipynb` -- demonstrates the potential length improvement of our intervals over OSB in a realistic particle unfolding simulation study. This example also showcases the MCMC Berger/Boos sampler and Quantile Regression components of the method.
4. `unfolding_smooth_adversarial_result_generation.ipynb` -- generates results for smooth and adversarial experiments.

## Data Generation
The file `data_generation.py` provides the necessary code to generate data from the GMM deconvolution (unfolding) problem scenario. Here, we provide information about the files within.
- `simulation_model_parameters.json`: provides the model parameter for the GMM data generating process.