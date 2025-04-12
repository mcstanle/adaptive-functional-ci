"""
Script to generate data to estimate the coverage of adaOSB in the three
dimensional numerical example.

See ./3d_example.ipynb for a more methodogical breakdown of each step.

1. Generates data with seeds
2. For each observation
    1. Generate sample of parameter values in 1 - eta confidence set
    2. Estimate the 1 - gamma quantiles at each point
    3. Select the point with the highest estimated 1 - gamma quantile
===============================================================================
Author        : Mike Stanley
Created       : Feb 23, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.sample import ellipsoidSampler
from adaFuncCI.max_quantile import maxQuantileRS
from adaFuncCI.llr import llrSolver_3d
import multiprocessing as mp
import numpy as np
from scipy import stats
import sympy
from time import time


def compute_maxq_i(i, data_i, gamma, eta, M, num_samp):
    """
    Compute max quantile for the ith observation.
    """
    # find ith prime random seed
    random_seed = sympy.prime(i + 1)

    # define sampler
    param_sampler = ellipsoidSampler(
        K=np.identity(3),
        Sigma=np.identity(3),
        y=data_i,
        eta=eta
    )

    # draw samples
    param_draws = param_sampler.sample_parameters(M=M, random_seed=i*2)

    # create llr objects
    llr = llrSolver_3d()
    max_q_rs = maxQuantileRS(
        X_train=param_draws,
        llr_solver=llr,
        distr=stats.norm,
        q=gamma,
        disable_tqdm=True
    )

    print(f'Iteration: {i} | Sampling max q')
    maxq = max_q_rs.estimate(
        num_samp=num_samp,
        random_seeds=np.arange(1, M + 1) * random_seed
    )
    print(f'Iteration: {i} | Max q sampled: {maxq}')

    return i, maxq


if __name__ == "__main__":

    # operational parameters
    NUM_CPU = None
    M = 10000  # number of parameters to sample for each observation
    NUM_SAMP = 10000  # number of draws to estimate each gamma quantile

    # set fixed experiment settings
    x_star = np.array([0., 0., 1.])
    h = np.array([1, 1, -1])
    noise_distr = stats.norm
    N = 1000  # number of data draws

    # uncertainty parameters
    alpha = 0.32
    eta = 0.01
    gamma = (alpha - eta) / (1 - eta)

    # generate noise
    np.random.seed(11211)
    noise = noise_distr.rvs(size=(N, 3))
    data = x_star + noise

    # determine number of CPUs to use
    NUM_CPU = None
    pool = mp.Pool(NUM_CPU if NUM_CPU else mp.cpu_count())
    print('Number of available CPUs: %i' % mp.cpu_count())
    print('Starting parallelization...')
    START = time()

    output_max_qs = np.zeros(N)

    def collect_data(data):
        idx = data[0]
        output_max_qs[idx] = data[1]

    for i in range(N):
        pool.apply_async(
            compute_maxq_i,
            args=(
                i,
                data[i],
                gamma,
                eta,
                M,  # number of parameter samples
                NUM_SAMP   # number of samples for gamma quant estimate
            ),
            callback=collect_data
        )
    pool.close()
    pool.join()

    # save array of max gamma quantiles
    SAVE_FP = '../data/3d_experiments'
    SAVE_FP += f'/numObs_{N}_num_param_{M}'
    SAVE_FP += f'_num_quant_{NUM_SAMP}'
    SAVE_FP += f'_alpha{alpha}_eta{eta}.npy'
    with open(SAVE_FP, 'wb') as f:
        np.save(file=f, arr=output_max_qs)
    print(f"Done. Elapsed time {(time() - START) / 60}")
