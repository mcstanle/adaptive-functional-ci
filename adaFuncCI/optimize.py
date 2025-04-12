"""
Code to optimize various quantities.
1. Interval endpoints
2. SSB interval endpoints
===============================================================================
Author        : Mike Stanley
Created       : Feb 08, 2024
Last Modified : May 08, 2024
===============================================================================
"""
import cvxpy as cp
from scipy import stats


def osb_int(
    y, q, K, h,
    solver_slack=cp.ECOS, solver_lb=cp.ECOS, solver_ub=cp.ECOS
):
    """
    Obtains endpoints for OSB interval with linear forward model and linear
    functional.

    NOTE: this code assumes a non-negativity constraint.

    Parameters
    ----------
        y            (np arr)       : observed data
        q            (np arr)       : quantile for use in interval constraint
        K            (np arr)       : linear forward model
        h            (np arr)       : linear functional
        solver_slack (cvxpy solver) : default ECOS
        solver_lb    (cvxpy solver) : default ECOS
        solver_ub    (cvxpy solver) : default ECOS

    Returns
    -------
        tuple of interval endpoints
    """
    m, p = K.shape
    x_lb = cp.Variable(p)
    x_ub = cp.Variable(p)

    # solve the slack optimization
    x_slack = cp.Variable(p)
    prob_slack = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(y - K @ x_slack)),
        constraints=[x_slack >= 0]
    )
    opt_slack = prob_slack.solve(solver=solver_slack)

    # set up the optimization problems
    prob_lb = cp.Problem(
        objective=cp.Minimize(h @ x_lb),
        constraints=[
            cp.sum_squares(y - K @ x_lb) - opt_slack <= q,
            x_lb >= 0
        ]
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(- h @ x_ub),
        constraints=[
            cp.sum_squares(y - K @ x_ub) - opt_slack <= q,
            x_ub >= 0
        ]
    )

    # solve the optimizations
    opt_lb = prob_lb.solve(solver=solver_lb)
    opt_ub = prob_ub.solve(solver=solver_ub)

    return opt_lb, -opt_ub


def ssb_int(
    y, K, h, alpha, include_slack,
    solver_lb=cp.ECOS, solver_ub=cp.ECOS
):
    """
    Obtains endpoints for SSB interval with linear forward model and linear
    functional.

    NOTE: this code assumes a non-negativity constraint.

    NOTE: use include_slack when the pre-image and constraint sets do not
    intersect.

    Parameters
    ----------
        y             (np arr)       : observed data
        K             (np arr)       : linear forward model
        h             (np arr)       : linear functional
        alpha         (float)        : 1 minus conf level
        include_slack (bool)         : include slack term
        solver_lb     (cvxpy solver) : default ECOS
        solver_ub     (cvxpy solver) : default ECOS

    Returns
    -------
        tuple of interval endpoints
    """
    m, p = K.shape
    x_lb = cp.Variable(p)
    x_ub = cp.Variable(p)

    if include_slack:
        x_slack = cp.Variable(p)
        constraint_slack = [x_slack >= 0]
        prob_slack = cp.Problem(
            cp.Minimize(cp.sum_squares(y - K @ x_slack)), constraint_slack
        )
        s2 = prob_slack.solve()
    else:
        s2 = 0

    # compute the SSB cutoff
    ssb_cutoff = stats.chi2(m).ppf(1 - alpha) + s2

    # set up the optimization problems
    prob_lb = cp.Problem(
        objective=cp.Minimize(h @ x_lb),
        constraints=[
            cp.sum_squares(y - K @ x_lb) <= ssb_cutoff,
            x_lb >= 0
        ]
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(- h @ x_ub),
        constraints=[
            cp.sum_squares(y - K @ x_ub) <= ssb_cutoff,
            x_ub >= 0
        ]
    )

    # solve the optimizations
    opt_lb = prob_lb.solve(solver=solver_lb)
    opt_ub = prob_ub.solve(solver=solver_ub)

    return opt_lb, -opt_ub, x_lb, x_ub
