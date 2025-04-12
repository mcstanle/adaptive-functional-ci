"""
Misc functions to handle various tasks.
===============================================================================
Author        : Mike Stanley
Created       : Feb 08, 2024
Last Modified : Feb 12, 2024
===============================================================================
"""
import numpy as np
from scipy import stats


def int_cover(mu_true, interval):
    """
    Returns true if the interval covers the true value.

    NOTE: uses a tolerance of 1e-6 for lower endpoint containing 0.

    Parameters
    ----------
        mu_true  (float) : true functional value
        interval (tuple) :

    Returns
    -------
        covers (bool) : true if interval covers
    """
    covers = True
    if mu_true >= 0.:
        le = 0.
        if interval[0] > 1e-6:
            le = interval[0]
        if (le > mu_true) or (interval[1] < mu_true):
            covers = False
    else:
        if (interval[0] > mu_true) or (interval[1] < mu_true):
            covers = False
    return covers


def compute_percentile_coverage(center, window, n=1000, alpha=0.05):
    """
    Finds coverage for nonparametric fixed-window quantile confidence interval.

    Parameters
    ----------
        center (int)   : center of interval
        window (int)   : size of window
        n      (int)   : number of observations to estimate quantile
        alpha  (float) : upper quantile level

    returns
    -------
        Coverage of center pm window
    """
    uep = center + window
    lep = center - window
    upper_prob = stats.binom(n=n, p=1 - alpha).cdf(uep - 1)
    lower_prob = stats.binom(n=n, p=1 - alpha).cdf(lep - 1)
    return upper_prob - lower_prob


def percentile_ci_idx(
    n=1000,
    alpha_quantile=0.05,
    alpha_interval=0.05,
    window_lb=1,
    window_ub=20
):
    """
    Finds the lower and upper bounds for a particular desired quantile CI.

    NOTE: we use the percentile estimator floor(np)

    Parameters
    ----------
        n              (int)   : number of observations
        alpha_quantile (float) : upper quantile of interest
        alpha_interval (float) : 1 minus confidence level
        window_lb      (int)   : inclusive lower bound for window search
        window_ub      (int)   : exclusive upper bound for window search

    Returns
    -------
        (l_idx, u_idx) : indices defining interval
    """
    center_idx = int((1 - alpha_quantile) * (n + 1))
    window_vals = np.arange(window_lb, window_ub)

    # find the coverage probability for each interval
    percent_probs = np.array([
        compute_percentile_coverage(
            center=center_idx, window=i, n=n, alpha=alpha_quantile
        ) for i in window_vals
    ])

    # find the smallest interval with at least 1 - alpha coverage
    min_idx = np.where(percent_probs >= 1 - alpha_interval)[0][0]

    return center_idx - window_vals[min_idx], center_idx + window_vals[min_idx]


def find_cis(
    samples,
    alpha_quantile=0.05,
    alpha_interval=0.05,
    window_lb=1,
    window_ub=20
):
    """
    Given array of samples, finds nonparametric CIs.

    This function can be used when a collection (m) of samples is given with
    each sample containing k draws.

    Parameters
    ----------
        samples        (np arr) : mxk - m is number of sample collections and
                                  and k is the size of each sample.
        alpha_quantile (float)  : upper quantile level
        alpha_interval (float)  : 1 minus confidence level
        window_lb      (int)    : inclusive lower bound for window search
        window_ub      (int)    : exclusive upper bound for window search
    Returns
    -------
        np arr of sorted values
    """
    m, k = samples.shape

    # sort the samples
    samples_sorted = np.sort(samples, axis=1)

    # compute the bounds
    ci_idxs = percentile_ci_idx(
        n=k, alpha_quantile=alpha_quantile, alpha_interval=alpha_interval,
        window_lb=window_lb, window_ub=window_ub
    )
    if ci_idxs[1] == k:
        ci_idxs = (ci_idxs[0], k - 1)

    return np.array([
        [samples_sorted[i, ci_idxs[0]], samples_sorted[i, ci_idxs[1]]]
        for i in range(m)
    ])
