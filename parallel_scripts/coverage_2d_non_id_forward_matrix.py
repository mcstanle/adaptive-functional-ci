"""
Script to generate data to estimate the coverage of adaOSB in the two
dimensional numerical example.

See ./2d_example_non_id_forward_matrix.ipynb for a more methodogical breakdown
of each step.

1. Generates data with seeds
2. For each observation
    1. Generate sample of parameter values in 1 - eta confidence set
    2. Estimate the 1 - gamma quantiles at each point
    3. Select the point with the highest estimated 1 - gamma quantile
===============================================================================
Author        : Mike Stanley
Created       : Feb 19, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.sample import ellipsoidSampler
from adaFuncCI.max_quantile import maxQuantileRS
from adaFuncCI.llr import llrSolver
import multiprocessing as mp
import numpy as np
from scipy import stats
import sympy
from time import time


def compute_maxq_i(
    i, K, Sigma, h, data_i, gamma, eta, M, num_samp
):
    """
    Computes the max quantile for the ith observation in given collection of
    observations.

    NOTE: assumes forwad model and covariance matrices are identity.

    There are two places where the random seed is invoked
    1. for the parameter sampling from the vgs algorithm
    2. for the gamma quantile estimation
        - max_q_rs.estimate takes an array of seeds. This script is written
          such that the random seed input is a prime, and therefore this
          sequential array is prime * i (i.e. multiples of a prime). This
          ensures that random seeds are unique across parallel instances.

    Parameters
    ----------
        i        (int)    : observation index
        K        (np arr) : forward matrix
        Sigma    (np arr) : covariance matrix
        h        (np arr) : functional of interest
        data_i   (np arr) : 2x1 array observation
        gamma    (float)  : 1 minus quantile level
        eta      (float)  : 1 minus confidence set level
        M        (int)    : number of parameter samples
        num_samp (int)  : number of samples to estimate gamma quantile

    Returns
    -------
        i       (int)   : observation index
        max_q_i (float) : maximum estimated quantile
    """
    # find ith prime random seed
    random_seed = sympy.prime(i + 1)

    # generate parameter samples
    ell_sampler = ellipsoidSampler(
        K=K,
        Sigma=Sigma,
        y=data_i,
        eta=eta
    )
    param_samples = ell_sampler.sample_parameters(
        M=M, random_seed=i
    )

    # estimate gamma quantile at each sample
    llr_2d = llrSolver(K=K, h=h)
    max_q_rs = maxQuantileRS(
        X_train=param_samples,
        llr_solver=llr_2d,
        distr=stats.norm,
        q=gamma,
        disable_tqdm=True
    )

    # obtain the max gamma quantile
    max_q_i = max_q_rs.estimate(
        num_samp=num_samp,
        random_seeds=np.arange(1, M + 1) * random_seed
    )

    return i, max_q_i


if __name__ == "__main__":

    # operational parameters
    X_STAR = np.array([0.75, 0.25])
    h = np.array([1, -1])
    Sigma = np.array([[1., 0.], [0., 2.]])
    NUM_OBS = 12
    NUM_PARAM_SAMP = 1000  # constant across all observations
    NUM_QUANT_SAMP = 500  # number of samples to estimate quantile
    ALPHA = 0.32  # desired level of CI
    ETA = 0.01  # confidence set level
    GAMMA = (ALPHA - ETA) / (1 - ETA)

    # determine number of CPUs to use
    NUM_CPU = None
    pool = mp.Pool(NUM_CPU if NUM_CPU else mp.cpu_count())
    print('Number of available CPUs: %i' % mp.cpu_count())
    print('Starting parallelization...')
    START = time()

    # forward matrix
    beta = 0.4
    K = np.array([[1 - beta, beta], [beta, 1 - beta]])

    # generate data
    DATA_GEN_SEED = 123456
    np.random.seed(DATA_GEN_SEED)
    noise = noise_distr = stats.multivariate_normal(cov=Sigma).rvs(NUM_OBS)
    data = K @ X_STAR + noise

    output_max_qs = np.zeros(NUM_OBS)

    def collect_data(data):
        idx = data[0]
        output_max_qs[idx] = data[1]

    for i in range(NUM_OBS):
        pool.apply_async(
            compute_maxq_i,
            args=(
                i,
                K,
                Sigma,
                h,
                data[i],
                GAMMA,
                ETA,
                NUM_PARAM_SAMP,  # number of parameter samples
                NUM_QUANT_SAMP   # number of samples for gamma quant estimate
            ),
            callback=collect_data
        )
    pool.close()
    pool.join()

    # save array of max gamma quantiles
    SAVE_FP = '../data/2d_experiments'
    SAVE_FP += f'/nonIdK_numObs_{NUM_OBS}_num_param_{NUM_PARAM_SAMP}'
    SAVE_FP += f'_num_quant_{NUM_QUANT_SAMP}'
    SAVE_FP += f'_alpha{ALPHA}_eta{ETA}.npy'
    with open(SAVE_FP, 'wb') as f:
        np.save(file=f, arr=output_max_qs)
    print(f"Done. Elapsed time {(time() - START) / 60}")
