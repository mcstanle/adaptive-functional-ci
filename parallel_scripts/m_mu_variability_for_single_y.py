"""
For a single observation, y, we sample NUM_SAMP realizations of
1. Train/test datasets
[Deprecated] 2. LLR samplings to generate QR data
[Deprecated] 3. Out-of-sample QR predictions and QoI test values

I.e., the primary outputs are
1. Out-of-sample QR predictions
2. QoI test values

The code in this script is based on:
mq_mu_estimation_unfolding_vs_direct_test_inversion.ipynb. There are other code
pointers within that notebook.

NOTE: I do not parallelize because of the inability of the polytope sampler
code to parallelize.

NOTE: we only retain the sampled parameter values (X) after the burn in. The
burn in length is specified in the MCMC dict.

NOTE: The QR uses hyperparameters trained in the pilot_study notebook.

NOTE: This code will also allow for us to look at the variability of the test
inversion approach.
===============================================================================
Author        : Mike Stanley
Created       : Apr 08, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
# from adaFuncCI.llr import llrSolver
# from adaFuncCI.max_quantile import maxQuantileQR
from adaFuncCI.sample import polytopeSampler
import json
import numpy as np
# from scipy import stats
from time import time


if __name__ == '__main__':

    # operational parameters
    NUM_SAMP = 100
    OBS_IDX = 100

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

    # # QR quantities
    # QR_HP_FP = '../data/parameter_settings_qr'
    # QR_HP_FP += '/qr_hyperparameters0.json'

    # with open(QR_HP_FP, 'rb') as f:
    #     qr_hyperparameters = json.load(f)

    # data arrays for saving
    X = np.zeros(shape=(NUM_SAMP, 2, M, p))
    qoi_test_vals = np.zeros(shape=(NUM_SAMP, M))
    # llr_train_samples = np.zeros(shape=(NUM_SAMP, M))
    # q_hat_oos = np.zeros(shape=(NUM_SAMP, M))

    START = time()
    for i in range(NUM_SAMP):

        # sample train/test sets
        eigrand_sampler = polytopeSampler(
            y=data_tilde[OBS_IDX],
            eta=mcmc_params["eta"],
            K=K_tilde,
            N_hp=mcmc_params["num_hp"],
            r=mcmc_params["radius"],
            random_seed=mcmc_params["random_seed_hp_gen"],
            polytope_type=mcmc_params["polytope_type"],
            alg=mcmc_params["mcmc_alg"]
        )

        # sample train/test
        X_train_i = eigrand_sampler.sample_ensemble(
            M=mcmc_params["chain_length"], burn_in=0
        )
        X_test_i = eigrand_sampler.sample_ensemble(
            M=mcmc_params["chain_length"], burn_in=0
        )

        # subset after burn-in
        X[i, 0, :, :] = X_train_i[mcmc_params['burn_in']:].copy()
        X[i, 1, :, :] = X_test_i[mcmc_params['burn_in']:].copy()

        # compute the QoI values
        qoi_test_vals[i, :] = X[i, 1, :, :] @ h

        # # sample LLR
        # llr_solver = llrSolver(K=K_tilde, h=h)
        # teststat_samp = nullTestStatSampler(
        #     noise_distr=stats.norm,
        #     test_stat=llr_solver,
        #     K=K_tilde, h=h,
        #     disable_tqdm=True
        # )
        # llr_train_samples[i, :] = teststat_samp.sample_teststat(
        #     param_vals=X[i, 0, :, :],
        #     random_seed=None
        # )

        # # train quantile regression
        # qr = maxQuantileQR(
        #     X_train=X[i, 0, :, :],
        #     X_test=X[i, 1, :, :],
        #     y_train=llr_train_samples[i, :],
        #     q=gamma,
        #     hyperparams=qr_hyperparameters
        # )
        # qr.estimate(random_state_reg=0)

        # # obtain out-of-sample predictions
        # q_hat_oos[i, :] = qr.gbt.predict(X[i, 1, :, :])

    # write out data
    SAVE_FP = '/home/mcstanle/adaOSB_hydra_data/m_mu_experiment'
    SAVE_FP += f'/obs_idx_{OBS_IDX}_num_samp_{NUM_SAMP}_X_qoi.npz'
    with open(SAVE_FP, 'wb') as f:
        np.savez(
            file=f,
            X=X,
            qoi_test_vals=qoi_test_vals,
            # llr_train_samples=llr_train_samples,
            # q_hat_oos=q_hat_oos
        )
    print(f"Done. Elapsed time {(time() - START) / 60}")
