"""
Script to generate data to estimate the coverage of adaOSB in the two
dimensional numerical example where x_star = (2 2) and h = (1 1)

This script is based on ./coverage_2d_inversion.py.

See ./dev_notebooks/2d_example_more_interesting.ipynb for a more
methodogical breakdown of each step.

1. Generates data with seeds
2. For each observation
    1. Generate sample of parameter values in 1 - eta confidence set
    2. Estimate the 1 - gamma quantiles at each point
    3. Compute the following intervals:
        1. MQ Direct
        2. MQ Optimized
        3. MQmu parameter
        4. MQmu functional
        5. OSB
===============================================================================
Author        : Mike Stanley
Created       : May 04, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.sample import ellipsoidSampler
from adaFuncCI.max_quantile import maxQuantileRS
from adaFuncCI.llr import llrSolver_2d_pos
from adaFuncCI.inversion_intervals import solve_llr_fixed_y
from adaFuncCI.inversion_intervals import direct_inversion
from adaFuncCI.inversion_intervals import max_local_quantile_inversion
from adaFuncCI.optimize import osb_int
import multiprocessing as mp
import numpy as np
from scipy import stats
import sympy
from time import time


def compute_intervals_i(
    i, data_i, alpha, gamma, eta, h, M, num_samp,
    method, hp_dict
):
    """
    Computes the following intervals:
    1. MQ Direct
    2. MQ Optimized
    3. MQmu parameter
    4. MQmu functional
    5. OSB

    NOTE: assumes forward model and covariance matrices are identity.

    Parameters
    ----------
        i        (int)    : observation index
        data_i   (np arr) : 2x1 array observation
        alpha    (float)  : original confidence level
        gamma    (float)  : 1 minus quantile level
        eta      (float)  : 1 minus confidence set level
        h        (np arr) : functional
        M        (int)    : number of parameter samples
        num_samp (int)    : number of samples to estimate gamma quantile
        method   (str)    : method used for MQmu func intervals
        hp_dict  (dict)   : hyperparameter dictionary for MQmu func method

    Returns
    -------
        i              (int)    : observation index
        interval_array (np arr) : 5x2 array with above intervals
        qoi_vals       (np arr) : sampled QoI values
        llr_vals_qoi   (np arr) : LLR values for data and sampled QoIs
    """
    # find ith prime random seed
    random_seed = sympy.prime(i + 1)

    # generate parameter samples
    ell_sampler = ellipsoidSampler(
        K=np.identity(2),
        Sigma=np.identity(2),
        y=data_i,
        eta=eta
    )
    param_samples = ell_sampler.sample_parameters(
        M=M, random_seed=i
    )

    # check the parameter sample is larger enough
    # if not, supplement with some repeats
    if param_samples.shape[0] != M:
        print(f'Iteration {i} | param samples is shape {param_samples.shape}')
        discrep = M - param_samples.shape[0]
        param_samples = np.vstack((param_samples, param_samples[:discrep, :]))

    # compute functional values
    qoi_vals = param_samples @ h

    # estimate gamma quantile at each sample
    llr_2d = llrSolver_2d_pos()
    max_q_rs = maxQuantileRS(
        X_train=param_samples,
        llr_solver=llr_2d,
        distr=stats.norm,
        q=gamma,
        disable_tqdm=True
    )
    max_q_rs.estimate(
        num_samp=num_samp,
        random_seeds=np.arange(1, M + 1) * random_seed
    )

    # solve for the LLR at each sampled functional value
    llr_vals_qoi = solve_llr_fixed_y(
        qoi_vals=qoi_vals,
        y=data_i,
        K=np.identity(2),
        h=h,
        disable_tqdm=True
    )

    # -- Interval computation!
    # --- MQ direct
    mq_direct = direct_inversion(
        qoi_vals=qoi_vals,
        llr_vals_qoi=llr_vals_qoi,
        q_hat_vals=max_q_rs.max_quantiles,
        local=False
    )

    # --- MQ Opt
    mq_opt = osb_int(
        y=data_i,
        q=max_q_rs.max_quantiles.max(),
        K=np.identity(2),
        h=h
    )

    # --- MQmu param
    mq_mu_param = direct_inversion(
        qoi_vals=qoi_vals,
        llr_vals_qoi=llr_vals_qoi,
        q_hat_vals=max_q_rs.max_quantiles,
        local=True
    )

    # --- MQmu func
    mq_mu_func = max_local_quantile_inversion(
        qoi_vals=qoi_vals,
        llr_vals_qoi=llr_vals_qoi,
        q_hat_vals=max_q_rs.max_quantiles,
        method=method,
        hyperparams=hp_dict
    )[0]

    # --- OSB Interval
    osb = osb_int(
        y=data_i, q=stats.chi2(1).ppf(1 - alpha),
        K=np.identity(2), h=h
    )

    # stick all these intervals into an array
    interval_array = np.vstack((
        mq_direct, mq_opt,
        mq_mu_param, mq_mu_func,
        osb
    ))

    return i, interval_array, qoi_vals, llr_vals_qoi


if __name__ == "__main__":

    # operational parameters
    X_STAR = np.array([2., 2.])
    h = np.array([1., 1.])
    NUM_OBS = 1000
    NUM_PARAM_SAMP = 1000  # constant across all observations
    NUM_QUANT_SAMP = 10000  # number of samples to estimate quantile
    ALPHA = 0.32  # desired level of CI
    ETA = 0.01  # confidence set level
    GAMMA = (ALPHA - ETA) / (1 - ETA)
    METHOD = 'rolling'
    HP_DICT = {'T': 10}

    # determine number of CPUs to use
    NUM_CPU = None
    pool = mp.Pool(NUM_CPU if NUM_CPU else mp.cpu_count())
    print('Number of available CPUs: %i' % mp.cpu_count())
    print('Starting parallelization...')
    START = time()

    # generate data
    DATA_GEN_SEED = 628
    np.random.seed(DATA_GEN_SEED)
    noise = stats.norm.rvs(size=(NUM_OBS, 2))
    data = X_STAR + noise

    output_intervals = np.zeros(shape=(NUM_OBS, 5, 2))
    output_qoi_vals = np.zeros(shape=(NUM_OBS, NUM_PARAM_SAMP))
    output_llr_vals = np.zeros(shape=(NUM_OBS, NUM_PARAM_SAMP))

    def collect_data(data):
        idx = data[0]
        output_intervals[idx] = data[1]
        output_qoi_vals[idx] = data[2]
        output_llr_vals[idx] = data[3]

    for i in range(NUM_OBS):
        pool.apply_async(
            compute_intervals_i,
            args=(
                i,
                data[i],
                ALPHA,
                GAMMA,
                ETA,
                h,
                NUM_PARAM_SAMP,  # number of parameter samples
                NUM_QUANT_SAMP,   # number of samples for gamma quant estimate
                METHOD,
                HP_DICT
            ),
            callback=collect_data
        )
    pool.close()
    pool.join()

    # save array of max gamma quantiles
    SAVE_FP = '/home/mcstanle/adaOSB_hydra_data/2d_exp'
    SAVE_FP += f'/inverted_ints_numObs_{NUM_OBS}_num_param_{NUM_PARAM_SAMP}'
    SAVE_FP += f'_num_quant_{NUM_QUANT_SAMP}'
    SAVE_FP += f'_alpha{ALPHA}_eta{ETA}_POS.npz'
    with open(SAVE_FP, 'wb') as f:
        np.savez(
            file=f,
            intervals=output_intervals,
            qoi_vals=output_qoi_vals,
            llr_vals=output_llr_vals
        )
    print(f"Done. Elapsed time {(time() - START) / 60}")
