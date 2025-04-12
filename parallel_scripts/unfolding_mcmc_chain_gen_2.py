"""
Script 2 for the unfolding MCMC chain generation. Responsible for generating
chains for observations 0-199 for the objects in
../data/unfold_true_dim80_smear_dim40/experiment_objects.npz.

The MCMC parameters are stored in ../data/mcmc_parameters0.json and apply to
all scripts. This file includes the index of the functional of interest.

For each observation, we sample two chains of length specified in
mcmc_parameters0.json; one is the "train" chain on which the quantile
regression will be fit, and the other is the "test" chain on which the out of
sample max will be estimated.

NOTE: for a more explicit example of how the sampler code works, see notebooks
pilot_study.ipynb and 2d_example.ipynb.

NOTE: SAVE_PATH is custom for the server configuration on which these results
were generated.

NOTE: the mcmc_parameters{idx}.json file indexes different experiment
configurations indexed by idx.
===============================================================================
Author        : Mike Stanley
Created       : Feb 20, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.sample import polytopeSampler
import cvxpy as cp
import json
import numpy as np
from time import time


if __name__ == "__main__":

    # operational parameters for this script
    OBS_IDXS = (400, 600)
    NUM_CHAINS = OBS_IDXS[1] - OBS_IDXS[0]
    IDX = 2
    EXP_IDX = 5
    CHEB_SOLVER = cp.ECOS

    # file paths
    EXP_OBJ_FP = '../data/unfold_true_dim80_smear_dim40'
    EXP_OBJ_FP += '/experiment_objects_adversarial_argmax.npz'
    MCMC_PARAM_FP = '../data/parameter_settings_mcmc'
    MCMC_PARAM_FP += f'/mcmc_parameters{EXP_IDX}.json'

    # read in required objects
    with open(EXP_OBJ_FP, 'rb') as f:
        exp_obj = np.load(f)
        data_tilde = exp_obj['data_tilde']
        K_tilde = exp_obj['K_tilde']
        H = exp_obj['H']

    with open(MCMC_PARAM_FP, 'rb') as f:
        mcmc_params = json.load(f)

    print("--- MCMC parameters ---")
    print(mcmc_params)

    # set misc. variables
    n, p = K_tilde.shape
    h = H[mcmc_params["functional_idx"]]

    # sample chains
    START = time()
    sampled_chains = np.zeros(
        shape=(NUM_CHAINS, 2, mcmc_params["chain_length"], p)
    )  # 0:train 1:test

    # define the mixing grid to be used in the sample
    MIX_GRID = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]

    for i in range(NUM_CHAINS):

        # instantiate sampler for ith observation
        eigrand_sampler = polytopeSampler(
            y=data_tilde[i + OBS_IDXS[0]],
            eta=mcmc_params["eta"],
            K=K_tilde,
            h=h,
            N_hp=mcmc_params["num_hp"],
            r=mcmc_params["radius"],
            random_seed=mcmc_params["random_seed_hp_gen"],
            polytope_type=mcmc_params["polytope_type"],
            alg=mcmc_params["mcmc_alg"],
            max_iters=10000,
            cheb_solver=CHEB_SOLVER
        )

        # sample train/test
        train_points_i = eigrand_sampler.sample_mixed_ensemble(
            M=mcmc_params["chain_length"], mix_grid=MIX_GRID
        )
        test_points_i = eigrand_sampler.sample_mixed_ensemble(
            M=mcmc_params["chain_length"], mix_grid=MIX_GRID,
            burn_in=100
        )

        ELAPSED_TIME = (time() - START) / 60
        print(f'It: {i} | Train test generated | Elapsed time: {ELAPSED_TIME}')

        # save chains in output array
        sampled_chains[i, 0, :] = train_points_i
        sampled_chains[i, 1, :] = test_points_i

    # define the save path for the chains
    SAVE_PATH = '/home/mcstanle/adaOSB_hydra_data/unfolding_exp'
    SAVE_PATH += f'/chains{IDX}_func{mcmc_params["functional_idx"]}'
    SAVE_PATH += f'_exp{EXP_IDX}.npy'

    with open(SAVE_PATH, 'wb') as f:
        np.save(file=f, arr=sampled_chains)
