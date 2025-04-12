"""
Code to compute inversion intervals and their supporting functions.

Code is based on the code used in
./parallel_scripts/unfolding_interval_solve_i.py.
===============================================================================
Author        : Mike Stanley
Created       : Apr 30, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.llr import llrSolver
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm


def solve_llr_fixed_y(qoi_vals, y, K, h, disable_tqdm=True):
    """
    Solves the LLR for a collection of functional values and one fixed y.

    Since the second optimization in the LLR is common for any evaluation at
    the same observation, we just compute that once to speed up computation.

    Parameters
    ----------
        qoi_vals     (np arr) : array of functional values
        y            (np arr) : observation
        K            (np arr) : linear forward model
        h            (np arr) : linear functional of interest
        disable_tqdm (bool)   : turn off tqdm

    Returns
    -------
        llr evaluations (np arr)
    """
    # instantiate llr
    llr_solver = llrSolver(K=K, h=h)

    # solve opt2 once
    opt_2_sol = llr_solver.solve_opt2(y=y)[0]

    # solve opt1 for each mu
    opt_1_sols = np.zeros(qoi_vals.shape[0])
    for i in tqdm(range(qoi_vals.shape[0]), disable=disable_tqdm):
        opt_1_sols[i] = llr_solver.solve_opt1(y=y, mu=qoi_vals[i])[0]

    return opt_1_sols - opt_2_sol


def I_vals(qoi_arr, mu_grid):
    """
    Assigns index {1,...,N} to each value in qoi_arr where N
    is the total number of bins.

    By convention, bins are defined by their upper endpoint.

    Parameters
    ----------
        qoi_arr (np arr) : functional values
        mu_grid (np arr) : defined grid over functioanl space

    Results
    -------
        bin_assign (np arr) : bin assignments for each point
    """
    bin_assign = np.zeros_like(qoi_arr)
    for i, qoi in enumerate(qoi_arr):

        # smallest index where less than
        lt_i = np.where(qoi <= mu_grid)[0][0]

        bin_assign[i] = lt_i

    return bin_assign


def max_over_bins(qoi_arr, mu_grid, q_hat):
    """
    Given grid of mu's, find the max predicted quantile over each.

    Parameters
    ----------
        qoi_arr (np arr) : quantity of interest values
        mu_grid (np arr) : grid boundary points
        q_hat   (np arr) : predicted quantile values

    Return
    ------
        max_quantiles (np arr) : max quantile for each bin
    """
    # assign each point to bin via QoI
    bin_assigns = I_vals(qoi_arr=qoi_arr, mu_grid=mu_grid)

    max_quantiles = np.zeros(mu_grid.shape[0])
    for k in range(mu_grid.shape[0]):

        # indices where bin is k
        bin_k_idx = np.where(bin_assigns == k)[0]

        # max quantile for that bin
        max_quantiles[k] = q_hat[bin_k_idx].max()

    return max_quantiles


def direct_inversion(qoi_vals, llr_vals_qoi, q_hat_vals, local=True):
    """
    Directly accept/reject sample points based on the estimated test statistic
    quantiles.

    Local --> accept/reject points based on their respsective predicted
    quantile values.

    Global --> accept/reject points based on the MAXIMUM predicted quantile.

    NOTE: this function assumes that LLR values have been computed.

    NOTE: if there are less than 2 accepted functional values, the returned
    interval is the min/max of the given functional values.

    Parameters
    ----------
        qoi_vals     (np arr) : functional values to accept/reject
        llr_vals_qoi (np arr) : LLR evaluated at functionals and data
        q_hat_vals   (np arr) : quantile values at sampled points
        local        (bool)   :


    Returns
    -------
        interval (tup)
    """
    if local:
        admissible_qoi_idxs = llr_vals_qoi <= q_hat_vals
    else:
        admissible_qoi_idxs = llr_vals_qoi <= q_hat_vals.max()

    if admissible_qoi_idxs.sum() < 2:
        return (qoi_vals.min(), qoi_vals.max())
    else:
        return (
            qoi_vals[admissible_qoi_idxs].min(),
            qoi_vals[admissible_qoi_idxs].max()
        )


def max_local_quantile_inversion(
    qoi_vals, llr_vals_qoi, q_hat_vals,
    method='rolling',
    hyperparams={'T': 1000, 'center': True}
):
    """
    Estimate the max_{phi(x) = mu}Q_x function as a function of mu in order to
    accept/reject functional values.

    Supported methods of estimating the max quantile
    1. Rolling max (rolling) - requires dictionary with argument 'T' and center
       to indicate the type of rolling window to use
    2. GP regression (gp_reg) - requires dictionary with arguments
        1. 'num_bins'
        2. 'len_scale'
        3. 'variance'

    Parameters
    ----------
        qoi_vals       (np arr) : functional values
        llr_vals_qoi   (np arr) : computed LLR at functional vals and data
        q_hat_vals     (np arr) : computed quantile values
        method         (str)
        hyperparams    (dict)   : num bins, length scale and variance hp's

    Returns
    -------
        interval   (tup)      : fit interval
        max_q_pred (np arr)   : estimator evaluated over sorted qoi_vals array
    """
    if method == 'rolling':
        assert 'T' in hyperparams
        assert 'center' in hyperparams
    elif method == 'gp_reg':
        assert 'num_bins' in hyperparams
        assert 'len_scale' in hyperparams
        assert 'variance' in hyperparams
    else:
        raise ValueError

    # get sorted versions of previous qoi and llr arrays
    qoi_sort_idx = np.argsort(qoi_vals)
    qoi_vals_sorted = qoi_vals[qoi_sort_idx].copy()
    llr_vals_sorted = llr_vals_qoi[qoi_sort_idx].copy()

    if method == 'rolling':
        q_hat_series = pd.Series(q_hat_vals[qoi_sort_idx])
        max_q_pred = q_hat_series.rolling(
            hyperparams['T'], center=hyperparams['center']
        ).max().dropna()

        llr_vals_sorted = llr_vals_sorted[hyperparams['T'] - 1:].copy()
        qoi_vals_sorted = qoi_vals_sorted[hyperparams['T'] - 1:].copy()

    elif method == 'gp_reg':

        # create mu grid and compute max quantile in each bin
        M = qoi_vals.shape[0]
        mu_grid = qoi_vals_sorted[
            np.arange(
                start=M / hyperparams['num_bins'],
                stop=M + 1 / M,
                step=M / hyperparams['num_bins'], dtype=int
            ) - 1
        ]
        max_q_grid = max_over_bins(
            qoi_arr=qoi_vals, mu_grid=mu_grid, q_hat=q_hat_vals
        )

        # fit the GP Reg
        gpr = GaussianProcessRegressor(
            kernel=RBF(length_scale=hyperparams['len_scale']),
            random_state=0,
            alpha=hyperparams['variance']
        ).fit(mu_grid[:, np.newaxis], max_q_grid)

        # evaluate fitted regression function over qoi_vals values
        max_q_pred = gpr.predict(qoi_vals_sorted[:, np.newaxis])

    # obtain interval by finding smallest/largest_indices
    zero_idxs = np.where(max_q_pred - llr_vals_sorted >= 0)[0]
    if zero_idxs.shape[0] < 2:
        interval = (qoi_vals_sorted[0], qoi_vals_sorted[-1])
    else:
        interval = (
            qoi_vals_sorted[zero_idxs[0]],
            qoi_vals_sorted[zero_idxs[-1]]
        )

    return interval, max_q_pred
