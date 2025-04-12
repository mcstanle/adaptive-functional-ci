"""
Script to generate data to estimate the coverage of adaOSB in the three
dimensional numerical example. This script is based on coverage_3d.py and
coverage_3d_inversion.py.

The key difference in this script is the use of random samples from the
polytope sampler instead of the ellipsoid sampler.

These chains are pre-generated in ../3d_example_polytope_sampler.ipynb.

1. Generates data with seeds
2. For each observation
    1. Generate sample of parameter values in 1 - eta confidence set
    2. Estimate the 1 - gamma quantiles at each point
    3. Fit the five different intervals
===============================================================================
Author        : Mike Stanley
Created       : May 07, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.max_quantile import maxQuantileRS
from adaFuncCI.llr import llrSolver_3d
from adaFuncCI.inversion_intervals import solve_llr_fixed_y
from adaFuncCI.inversion_intervals import direct_inversion
from adaFuncCI.inversion_intervals import max_local_quantile_inversion
from adaFuncCI.optimize import osb_int
import multiprocessing as mp
import numpy as np
from scipy import stats
import sympy
from time import time


def compute_maxq_i(
        i, data_i,
        param_draws,
        alpha, gamma,
        M, num_samp,
        method, hp_dict
):
    """
    Compute max quantile for the ith observation.
    """
    # define the functional
    h = np.array([1, 1, -1])

    # find ith prime random seed
    random_seed = sympy.prime(i + 1)

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

    # compute functional values
    qoi_vals = param_draws @ h

    # solve for the LLR at each sampled functional value
    llr_vals_qoi = solve_llr_fixed_y(
        qoi_vals=qoi_vals,
        y=data_i,
        K=np.identity(3),
        h=h,
        disable_tqdm=True
    )

    # --- MQ direct
    mq_direct = direct_inversion(
        qoi_vals=qoi_vals,
        llr_vals_qoi=llr_vals_qoi,
        q_hat_vals=max_q_rs.max_quantiles,
        local=False
    )

    # --- MQ Opt
    mq_opt = osb_int(
        y=data_i, q=max_q_rs.max_quantiles.max(), K=np.identity(3), h=h
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
        qoi_vals=param_draws @ h,
        llr_vals_qoi=llr_vals_qoi,
        q_hat_vals=max_q_rs.max_quantiles,
        method=method,
        hyperparams=hp_dict
    )[0]

    # --- OSB Interval
    osb = osb_int(
        y=data_i, q=stats.chi2(1).ppf(1 - alpha), K=np.identity(3), h=h
    )

    # stick all these intervals into an array
    interval_array = np.vstack((
        mq_direct, mq_opt,
        mq_mu_param, mq_mu_func,
        osb
    ))

    return i, interval_array, qoi_vals, llr_vals_qoi, max_q_rs.max_quantiles


if __name__ == "__main__":

    # operational parameters
    NUM_CPU = None
    NUM_SAMP = 10000  # number of draws to estimate each gamma quantile
    METHOD = 'rolling'
    HP_DICT = {'T': 10, 'center': True}

    # set fixed experiment settings
    t = 0.03
    x_star = np.array([t, t, 1.])
    h = np.array([1, 1, -1])
    noise_distr = stats.norm
    N = 1000  # number of data draws
    M = 16000  # number of parameter draws per observation

    # uncertainty parameters
    alpha = 0.32
    eta = 0.01
    gamma = alpha - eta

    # generate noise
    np.random.seed(11211)
    noise = noise_distr.rvs(size=(N, 3))
    data = x_star + noise

    # read in the samples
    # SAMPLE_FP = '/home/mcstanle/adaOSB_hydra_data/3d_exp'
    # SAMPLE_FP += '/parameter_draws_1000obs_16k_samples_mixed_t0.1.npy'
    # with open(SAMPLE_FP, 'rb') as f:
    #     param_draws_all = np.load(f)

    # read in the samples from separate files
    SAMPLE_FP = '/home/mcstanle/adaOSB_hydra_data/3d_exp'
    param_draws_all = np.zeros(shape=(N, M, 3))
    for i in range(5):
        with open(
            SAMPLE_FP + '/sample%i_importance_sampler.npy' % i, 'rb'
        ) as f:
            param_draws_all[(200 * i):(200 * (i + 1)), :, :] = np.load(f)

    # determine number of CPUs to use
    NUM_CPU = None
    pool = mp.Pool(NUM_CPU if NUM_CPU else mp.cpu_count())
    print('Number of available CPUs: %i' % mp.cpu_count())
    print('Starting parallelization...')
    START = time()

    output_intervals = np.zeros(shape=(N, 5, 2))
    output_qoi_vals = np.zeros(shape=(N, M))
    output_llr_vals = np.zeros(shape=(N, M))
    output_q_hat_vals = np.zeros(shape=(N, M))

    def collect_data(data):
        idx = data[0]
        output_intervals[idx] = data[1]

        # determine size
        QOI_SIZE = data[2].shape[0]
        LLR_SIZE = data[3].shape[0]
        QHAT_SIZE = data[4].shape[0]
        output_qoi_vals[idx, :QOI_SIZE] = data[2]
        output_llr_vals[idx, :LLR_SIZE] = data[3]
        output_q_hat_vals[idx, :QHAT_SIZE] = data[4]

    for i in range(N):
        pool.apply_async(
            compute_maxq_i,
            args=(
                i,
                data[i],
                param_draws_all[i],
                alpha,
                gamma,
                M,  # number of parameter samples
                NUM_SAMP,   # number of samples for gamma quant estimate
                METHOD,
                HP_DICT
            ),
            callback=collect_data
        )
    pool.close()
    pool.join()

    # save array of max gamma quantiles
    SAVE_FP = '/home/mcstanle/adaOSB_hydra_data/3d_exp'
    SAVE_FP += f'/numObs_{N}_num_param_{M}'
    SAVE_FP += f'_num_quant_{NUM_SAMP}'
    SAVE_FP += f'_alpha{alpha}_eta{eta}'
    SAVE_FP += f'_rolllingT{HP_DICT["T"]}'
    SAVE_FP += '_importance_sampler_t0.03_bbcalib_center_roll.npz'
    with open(SAVE_FP, 'wb') as f:
        np.savez(
            file=f,
            intervals=output_intervals,
            qoi_vals=output_qoi_vals,
            llr_vals=output_llr_vals,
            q_hat_vals=output_q_hat_vals
        )
    print(f"Done. Elapsed time {(time() - START) / 60}")
