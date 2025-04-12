"""
This script provides the functions necessary to support
unfolding_interval_solve_X.py.
===============================================================================
Author        : Mike Stanley
Created       : Apr 17, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
from adaFuncCI.llr import llrSolver
from adaFuncCI.optimize import osb_int
import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm


def solve_llr_fixed_y(qoi_vals, y, K, h, disable_tqdm=True):
    """
    Solves the LLR for a collection of mu values and one fixed y.
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
    Assigns index {0,...,N} to each value in qoi_arr where N
    is the total number of bins
    """
    bin_assign = np.zeros_like(qoi_arr)
    for i, qoi in enumerate(qoi_arr):

        # smallest index where less than
        lt_i = np.where(qoi <= mu_grid)[0][0]

        bin_assign[i] = lt_i - 1 if lt_i > 0 else 0

    return bin_assign


def max_over_bins(qoi_arr, mu_grid, q_hat):
    """
    Given grid of mu's, find the max predicted quantile over each.

    Parameters
    ----------
        qoi_arr (np arr) : quantity of interest values
        mu_grid (np arr) : grid boundary points
        q_hat   (np arr) : predicted quantile values
    """
    # assign each point to bin via QoI
    bin_assigns = I_vals(qoi_arr=qoi_arr, mu_grid=mu_grid)

    max_quantiles = np.zeros(mu_grid.shape[0] - 1)
    for k in range(mu_grid.shape[0] - 1):

        # indices where bin is k
        bin_k_idx = np.where(bin_assigns == k)[0]

        # max quantile for that bin
        max_quantiles[k] = q_hat[bin_k_idx].max()

    return max_quantiles


def compute_intervals(
    i, y, qoi_vals, q_hat_vals, K, h,
    num_mu_grid, alpha,
    disable_tqdm=True
):
    """
    Computes four intervals:
    1. MQ - Direct inversion
    2. MQ - Optimized
    3. MQmu - Parameter space inversion
    4. MQmu - Functional space inversion

    NOTE: this function requires precomputed test functional values
    and precomputed quantile predictions at test points.

    NOTE: for the mu-functional interval, instead of re-computing the
    llr over a new grid, we simply use the original llr computations
    over the test QoI values for the grid by sorting.

    Parameters
    ----------
        i            (int)    : iteration index for parallelization
        y            (np arr) : observation vector
        qoi_vals     (np arr) : function values at test points
        q_hat_vals   (np arr) : QR predictions at test points
        K            (np arr) : linear forward model
        h            (np arr) : linear functional
        num_mu_grid  (int)    : number of mu grid elements
        alpha        (float)  : 1 - confidence level
        disable_tqdm (bool)   : toggles progress bar for llr values

    Returns
    -------
        mq_direct   (tup)
        mq_opt      (tup)
        mq_mu_param (tup)
        mq_mu_func  (tup)
        osb         (tup)
    """
    # solve for the LLR at each sampled functional value
    llr_vals_qoi_test = solve_llr_fixed_y(
        qoi_vals=qoi_vals,
        y=y,
        K=K,
        h=h,
        disable_tqdm=disable_tqdm
    )

    # --- MQ direct
    admissible_qoi_direct_idxs = llr_vals_qoi_test <= q_hat_vals.max()
    if admissible_qoi_direct_idxs.sum() < 2:
        mq_direct = (qoi_vals.min(), qoi_vals.max())
    else:
        mq_direct = (
            qoi_vals[admissible_qoi_direct_idxs].min(),
            qoi_vals[admissible_qoi_direct_idxs].max()
        )

    # --- MQ Opt
    mq_opt = osb_int(y=y, q=q_hat_vals.max(), K=K, h=h)

    # --- MQmu param
    admissible_qoi_idxs = llr_vals_qoi_test <= q_hat_vals
    if admissible_qoi_idxs.sum() < 2:
        mq_mu_param = (qoi_vals.min(), qoi_vals.max())
    else:
        mq_mu_param = (
            qoi_vals[admissible_qoi_idxs].min(),
            qoi_vals[admissible_qoi_idxs].max()
        )

    # --- MQmu func
    # create mu grid
    mu_grid = np.linspace(
        qoi_vals.min(), qoi_vals.max(), num=num_mu_grid + 1
    )
    max_q_grid = max_over_bins(
        qoi_arr=qoi_vals, mu_grid=mu_grid, q_hat=q_hat_vals
    )

    # fit the GP Reg
    gpr = GaussianProcessRegressor(
        kernel=RBF(length_scale=1e2), random_state=0
    ).fit(mu_grid[:-1][:, np.newaxis], max_q_grid)

    # get sorted versions of previous qoi and llr arrays
    qoi_sort_idx = np.argsort(qoi_vals)
    qoi_vals_sorted = qoi_vals[qoi_sort_idx].copy()
    llr_vals_sorted = llr_vals_qoi_test[qoi_sort_idx].copy()

    # evaluate fitted regression function over qoi_vals values
    m_mu_gp_test = gpr.predict(qoi_vals_sorted[:, np.newaxis])

    # obtain interval by finding smallest/largest_indices
    zero_idxs = np.where(m_mu_gp_test - llr_vals_sorted >= 0)[0]
    if zero_idxs.shape[0] < 2:
        mq_mu_func = (qoi_vals_sorted[0], qoi_vals_sorted[-1])
    else:
        mq_mu_func = (
            qoi_vals_sorted[zero_idxs[0]],
            qoi_vals_sorted[zero_idxs[-1]]
        )

    # --- OSB Interval
    osb = osb_int(y=y, q=stats.chi2(1).ppf(1 - alpha), K=K, h=h)

    print(f'Iteration {i} | Intervals Computed')

    return (
        i,
        mq_direct,
        mq_opt,
        mq_mu_param,
        mq_mu_func,
        osb
    )
