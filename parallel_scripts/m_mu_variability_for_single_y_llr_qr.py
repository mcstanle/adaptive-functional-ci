"""
For a single observation, y, we use the NUM_SAMP realization of the data
generated process from m_mu_variability_for_single_y.py to
1. LLR samplings to generate QR data
2. Out-of-sample QR predictions and QoI test values

I.e., the primary outputs are
1. Out-of-sample QR predictions
2. QoI test values

The code in this script is based on:
mq_mu_estimation_unfolding_vs_direct_test_inversion.ipynb. There are other code
pointers within that notebook.
.

NOTE: The QR uses hyperparameters trained in the pilot_study notebook.

NOTE: This code will also allow for us to look at the variability of the test
inversion approach.
===============================================================================
Author        : Mike Stanley
Created       : Apr 08, 2024
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
        pred_oos      (np arr) : out of sample predictions
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

    # out of sample prediction
    pred_oos = qr.gbt.predict(X_test)

    return i, pred_oos


if __name__ == "__main__":

    # operational parameters
    NUM_SAMP = 100
    OBS_IDX = 100
    NUM_CPU = None

    # file paths
    EXP_OBJ_FP = '../data/unfold_true_dim80_smear_dim40'
    EXP_OBJ_FP += '/experiment_objects.npz'
    MCMC_PARAM_FP = '../data/parameter_settings_mcmc'
    MCMC_PARAM_FP += '/mcmc_parameters0.json'

    # read in required objects
    with open(EXP_OBJ_FP, 'rb') as f:
        exp_obj = np.load(f)
        data_tilde = exp_obj['data_tilde']
        K_tilde = exp_obj['K_tilde']
        H = exp_obj['H']

    with open(MCMC_PARAM_FP, 'rb') as f:
        mcmc_params = json.load(f)

    # key quantities from the MCMC file
    M = mcmc_params["chain_length"] - mcmc_params["burn_in"]
    gamma = (mcmc_params["alpha"] - mcmc_params["eta"])
    gamma /= (1 - mcmc_params["eta"])

    # set misc. variables
    n, p = K_tilde.shape
    h = H[mcmc_params["functional_idx"]]

    # QR quantities
    QR_HP_FP = '../data/parameter_settings_qr'
    QR_HP_FP += '/qr_hyperparameters0.json'

    with open(QR_HP_FP, 'rb') as f:
        qr_hyperparameters = json.load(f)

    # read in the arrays from the chain sampler
    CHAIN_FP = '/home/mcstanle/adaOSB_hydra_data/m_mu_experiment'
    CHAIN_FP += f'/obs_idx_{OBS_IDX}_num_samp_{NUM_SAMP}_X_qoi.npz'
    with open(CHAIN_FP, 'rb') as f:
        load_obj = np.load(file=f)
        X = load_obj['X']
        qoi_test_vals = load_obj['qoi_test_vals']

    # data arrays for saving
    llr_train_samples = np.zeros(shape=(NUM_SAMP, M))
    q_hat_oos = np.zeros(shape=(NUM_SAMP, M))

    # determine number of CPUs to use
    pool = mp.Pool(NUM_CPU if NUM_CPU else mp.cpu_count())
    print('Number of available CPUs: %i' % mp.cpu_count())
    print('Starting parallelization...')
    START = time()

    output_data = np.zeros(shape=(NUM_SAMP, M))

    def collect_data(data):
        idx = data[0]
        output_data[idx, :] = data[1]

    for i in range(NUM_SAMP):
        pool.apply_async(
            llr_and_qr_i,
            args=(
                i,  # iteration index
                K_tilde,
                H[mcmc_params["functional_idx"]],
                X[i, 0, :, :],  # train
                X[i, 1, :, :],  # test
                qr_hyperparameters,
                gamma
            ),
            callback=collect_data
        )
    pool.close()
    pool.join()

    # write out data
    SAVE_FP = '/home/mcstanle/adaOSB_hydra_data/m_mu_experiment'
    SAVE_FP += f'/obs_idx_{OBS_IDX}_num_samp_{NUM_SAMP}_qhat.npy'
    with open(SAVE_FP, 'wb') as f:
        np.save(file=f, arr=output_data)
    print(f"Done. Elapsed time {(time() - START) / 60}")
