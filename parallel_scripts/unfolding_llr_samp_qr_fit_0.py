"""
Script 0 for the unfolding LLR sampling and QR fit. Responsible for those
observations 0-199 and reads in the chains generated from
unfolding_mcmc_chain_gen_0.py. That datafile starts with "chains{}"

Unfolding objects found in
../data/unfold_true_dim80_smear_dim40/experiment_objects.npz.

The MCMC parameters are stored in ../data/mcmc_parameters0.json and apply to
all scripts. This file includes the index of the functional of interest.

For the Quantile regression, we use the hyperparameters found via cross
validation in ../pilot_study_unfolding.ipynb.

For each observation, we want both in and out of sample max prediction.
===============================================================================
Author        : Mike Stanley
Created       : Feb 21, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.llr import llrSolver
from adaFuncCI.max_quantile import maxQuantileQR
from adaFuncCI.sample import nullTestStatSampler
import json
import multiprocessing as mp
import numpy as np
from scipy import stats
from time import time


def llr_and_qr_i(i, K, h, X_train, X_test, qr_hyperparams, gamma):
    """
    Samples LLR at each point generated in the train part of the MCMC chain
    after burn in. Each proocess
    1. Creates LLR object
    2. Creates test statistic object
    3. Sample from the test statistic
    4. Creates QR object
    5. Train quantile regression
    6. Return train max and test max

    Parameters
    ----------
        i              (int)    : process index
        K              (np arr) : forward matrix (n x p)
        h              (np arr) : functional
        X_train        (np arr) : training data
        X_test         (np arr) : testing data
        qr_hyperparams (dict)   : quantile regression hyperparameters
        gamma          (float)  : 1 minus quantile level

    Returns
    -------
        i             (int)    : process index
        estimated_max (tup)    : train/test estimated maxes for each split
        llr_samples   (np arr) : llr samples
        oos_pred      (np arr) : out-of-sample prediction array
    """
    # sample from test statistic
    llr_solver = llrSolver(K=K, h=h)
    teststat_samp = nullTestStatSampler(
        noise_distr=stats.norm,
        test_stat=llr_solver,
        K=K, h=h,
        disable_tqdm=True
    )
    llr_samples = teststat_samp.sample_teststat(
        param_vals=X_train, random_seed=i
    )
    print(f'Iteration {i} | LLR samples done')

    # train quantile regression
    qr = maxQuantileQR(
        X_train=X_train, X_test=X_test, y_train=llr_samples,
        q=gamma, hyperparams=qr_hyperparams
    )
    qr.estimate(random_state_reg=0, random_state_cv=1)
    print(f'Iteration {i} | QR trained')

    return (
        i,
        (qr.maxq_insample.max(), qr.maxq_outsample.max()),
        llr_samples,
        qr.gbt.predict(X_test)
    )


if __name__ == "__main__":

    # operational parameters for this script
    OBS_IDXS = (0, 200)
    NUM_CHAINS = OBS_IDXS[1] - OBS_IDXS[0]
    IDX = 0
    EXP_IDX = 5
    NUM_CPU = None

    # file paths
    EXP_OBJ_FP = '../data/unfold_true_dim80_smear_dim40'
    EXP_OBJ_FP += '/experiment_objects_adversarial_argmax.npz'
    MCMC_PARAM_FP = '../data/parameter_settings_mcmc'
    MCMC_PARAM_FP += f'/mcmc_parameters{EXP_IDX}.json'
    QR_HP_FP = '../data/parameter_settings_qr'
    QR_HP_FP += f'/qr_hyperparameters{EXP_IDX}.json'

    # unfolding objects
    with open(EXP_OBJ_FP, 'rb') as f:
        exp_obj = np.load(f)
        data_tilde = exp_obj['data_tilde']
        K_tilde = exp_obj['K_tilde']
        H = exp_obj['H']

    # mcmc parameters
    with open(MCMC_PARAM_FP, 'rb') as f:
        mcmc_params = json.load(f)

    # compute gamma quantile
    gamma = mcmc_params["alpha"] - mcmc_params["eta"]

    # qr hyperparameters
    with open(QR_HP_FP, 'rb') as f:
        qr_hyperparameters = json.load(f)
    print(f'QR Hyperparameters: {qr_hyperparameters}')

    CHAIN_FP = '/home/mcstanle/adaOSB_hydra_data/unfolding_exp'
    CHAIN_FP += f'/chains{IDX}_func{mcmc_params["functional_idx"]}'
    CHAIN_FP += f'_exp{EXP_IDX}.npy'

    # parameter chains
    with open(CHAIN_FP, 'rb') as f:
        chain_arr = np.load(f)[
            :, :, mcmc_params["burn_in"]:, :
        ]  # chains, train/test, chain len, dim
    print(f'Chain array size: {chain_arr.shape}')

    # determine number of CPUs to use
    pool = mp.Pool(NUM_CPU if NUM_CPU else mp.cpu_count())
    print('Number of available CPUs: %i' % mp.cpu_count())
    print('Starting parallelization...')
    START = time()

    output_data_maxes = np.zeros(shape=(NUM_CHAINS, 2))  # 0: train - 1: test
    output_data_llr_samps = np.zeros(shape=(NUM_CHAINS, chain_arr.shape[2]))
    output_data_oos_pred = np.zeros(shape=(NUM_CHAINS, chain_arr.shape[2]))

    def collect_data(data):
        idx = data[0]
        output_data_maxes[idx, :] = data[1]
        output_data_llr_samps[idx, :] = data[2]
        output_data_oos_pred[idx, :] = data[3]

    for i in range(NUM_CHAINS):
        pool.apply_async(
            llr_and_qr_i,
            args=(
                i,  # iteration index
                K_tilde,
                H[mcmc_params["functional_idx"]],
                chain_arr[i, 0, :, :],  # train
                chain_arr[i, 1, :, :],  # test
                qr_hyperparameters,
                gamma
            ),
            callback=collect_data
        )
    pool.close()
    pool.join()

    # write out data
    SAVE_FP = '/home/mcstanle/adaOSB_hydra_data/unfolding_exp'
    SAVE_FP += f'/qr_maxes{IDX}_func{mcmc_params["functional_idx"]}'
    SAVE_FP += f'_exp{EXP_IDX}_BBcalib.npz'
    with open(SAVE_FP, 'wb') as f:
        np.savez(
            file=f,
            output_data_maxes=output_data_maxes,
            output_data_llr_samps=output_data_llr_samps,
            output_data_oos_pred=output_data_oos_pred,
        )
    print(f"Done. Elapsed time {(time() - START) / 60}")
