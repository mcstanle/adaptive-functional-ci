"""
Script to generate samples 800-999 for the 3d experiment with the importance
sampler developed in ../../dev_notebooks/3d_example_polytope_sampler.ipynb.
===============================================================================
Author        : Mike Stanley
Created       : May 10, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.sample import polytopeSampler
import numpy as np
from scipy import stats
from time import time


def importance_sampler(
    data_i,
    M,
    eta,
    K,
    h,
    mcmc_hp_dict,
    x_center=np.zeros(3),
    ap_gamma=0.5,
    ap_ord=0.25
):
    """
    Wrapper around the Polytope algorithm that includes an extra
    accept/reject step.

    The accept probability is defined by
    exp(-ap_gamma * norm(x - x_center)_{ap_ord}).

    Parameters
    ----------
        data_i       (np arr) : data vector
        M            (int)    : number of total samples to draw
        eta          (float)  : BB set prob
        K            (np arr) : forward model
        h            (np arr) : functional vector
        mcmc_hp_dict (dict)   : hyperparameter for mcmc algo
        x_center     (np arr) : center location of acceptance prob calc
        ap_gamma     (float)  : "accept-probability" gamma
        ap_ord       (float)  : "accept-probability" order of norm (should <1)

    Returns
    -------
        final_sample (np arr) : complete sample
    """
    # declare sampler
    sampler = polytopeSampler(
        y=data_i,
        eta=eta,
        K=K,
        h=h,
        N_hp=mcmc_hp_dict['N_hp'],
        r=mcmc_hp_dict['radius'],
        random_seed=None,
        polytope_type=mcmc_hp_dict['polytope_type'],
        alg=mcmc_hp_dict['mcmc_alg'],
        disable_tqdm=True
    )

    num_samples = 0
    final_sample = np.zeros(shape=(M, K.shape[1]))
    prev_idx = 0
    curr_idx = 0
    while num_samples < M:

        # draw samples from polytope
        param_draws = sampler.sample_mixed_ensemble(M=M)

        # compute their accept reject probabilies
        accept_probs = np.exp(
            -ap_gamma * np.linalg.norm(
                param_draws - x_center, ord=ap_ord, axis=1
            )
        )

        # decide to accept/reject
        accept_reject = stats.bernoulli(accept_probs).rvs()

        # save accepted points
        curr_idx = prev_idx + accept_reject.sum()

        if curr_idx > M:
            curr_idx = M

        final_sample[prev_idx:curr_idx, :] = param_draws[
            accept_reject == 1, :
        ][:(curr_idx - prev_idx)]

        # updates
        prev_idx = curr_idx
        num_samples += accept_reject.sum()

    return final_sample


if __name__ == "__main__":

    # --- operational parameters
    OBS_IDXS = (800, 1000)
    IDX = 4
    NUM_CHAINS = OBS_IDXS[1] - OBS_IDXS[0]
    M = 16000
    AP_GAMMA = 0.75
    AP_ORD = 0.5

    # --- generate data
    t = 0.03
    x_star = np.array([t, t, 1.])
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

    # --- perform sampling
    # define dictionary with sampler properties
    mcmc_dict = {
        'N_hp': 6,
        'radius': 0.5,
        'polytope_type': 'eigen',
        'mcmc_alg': 'vaidya'
    }

    START = time()
    param_draws_all = np.zeros(shape=(NUM_CHAINS, M, 3))
    for i in range(NUM_CHAINS):

        # generate sample
        param_draws_all[i, :, :] = importance_sampler(
            data_i=data[i + OBS_IDXS[0]],
            M=M,
            eta=eta,
            K=np.identity(3),
            h=h,
            mcmc_hp_dict=mcmc_dict,
            ap_gamma=AP_GAMMA, ap_ord=AP_ORD
        )
        print_msg = f'Sample {i + OBS_IDXS[0]} generated '
        print_msg += f'| Elapsed time: {(time() - START)/60} minutes'
        print(print_msg)

    # define the save path for the chains
    SAVE_PATH = '/home/mcstanle/adaOSB_hydra_data/3d_exp'
    SAVE_PATH += f'/sample{IDX}_importance_sampler.npy'

    with open(SAVE_PATH, 'wb') as f:
        np.save(file=f, arr=param_draws_all)
