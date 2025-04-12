"""
Log-likelihood ratio computation object. Objects
1. llrSolver -- general LLR solver for gaussian noise
2. llrSolver2d -- solver for 2d example
===============================================================================
Author        : Mike Stanley
Created       : Feb 12, 2024
Last Modified : May 03, 2024
===============================================================================
"""
import cvxpy as cp
import numpy as np
import sys


class llrSolver:
    """
    Class to solve log likelihood ratio optimizations.

    NOTE:
    1. "optimization 1" is the one constrained over the level set defined by
       mu and the parameter constraint set.
    2. "optimization 2" is the one constrained over just the parameter
       constraint set.

    NOTE: solving these LLRs can numerically unstable. If the optimizations are
    solved suboptimally, it is done in the direction of being conservative.
    Namely, for optimization 1, we solve the primal minimization, so suboptimal
    will be an upper bound. For optimization 2, we solve the dual maximization,
    so suboptimal will be a lower bound (therefore upper bound on whole
    quantity because of the subtraction).

    Parameters:
    -----------
        K            (np arr) : forward model linear operator
        h            (np arr) : functional vector
        opt1_solver  (bool)   : solver for optimization 1
        opt2_solver  (bool)   : solver for optimization 1
        max_iters    (int)    : maximum number of iteration for solver
        opt1_verbose (bool)   : toggle on verbose for debugging opt 1
        opt2_verbose (bool)   : toggle on verbose for debugging opt 1
    """
    def __init__(
        self, K, h,
        opt1_solver=cp.ECOS, opt2_solver=cp.ECOS,
        max_iters=1000,
        opt1_verbose=False, opt2_verbose=False
    ):
        self.K = K
        self.h = h
        self.max_iters = max_iters
        self.opt1_solver = opt1_solver
        self.opt2_solver = opt2_solver
        self.opt1_verbose = opt1_verbose
        self.opt2_verbose = opt2_verbose
        self.batch = False

    def solve_opt1(
        self, y, mu,
        abstol_inacc_start=5e-5, abstol_update=10.
    ):
        """
        I.e., solves the optimization for {x: h^tx = mu and x in X}

        NOTE: only supports non-negativity constraints at the moment.

        NOTE: adding in solver error robustness (Jan 16 2024)
        - stops trying when inaccuracy is > 1

        Parameters:
        -----------
            y                  (np arr) : observation vector
            mu                 (float)  : functional level set
            abstol_inacc_start (float)  : see [1]
            abstol_update      (float)  : increase factor for abstol_inacc

        Returns:
        --------
            opt_hp  (float) : optimized objective function value
            optimal (bool)  : 1 if converged
        """
        try_again = True
        abstol_inacc = abstol_inacc_start
        while try_again & (abstol_inacc < 1.):
            try:
                # variables to optimize
                x_hp = cp.Variable(self.h.shape[0])  # 'hp' == 'hyperplane'

                # set up the optimization problems
                constraints = [
                    self.h @ x_hp == mu,
                    x_hp >= 0
                ]
                prob_hp = cp.Problem(
                    objective=cp.Minimize(cp.sum_squares(y - self.K @ x_hp)),
                    constraints=constraints
                )

                # solve the problems
                opt_hp = prob_hp.solve(
                    solver=self.opt1_solver, verbose=self.opt1_verbose,
                    max_iters=self.max_iters,
                    abstol_inacc=abstol_inacc
                )

                # convergence checks
                optimal = 'optimal' == prob_hp.status

                # turn off the freaking loop!!!
                try_again = False

            except cp.SolverError:
                abstol_inacc *= abstol_update
                err_msg = 'abstol caused a problem in opt1 -- again with'
                print(f'{err_msg} {abstol_inacc}', file=sys.stderr)

        if abstol_inacc > 1.:
            print('opt1 did not solve', file=sys.stderr)
            raise cp.SolverError

        return opt_hp, optimal

    def solve_opt2(self, y, abstol_inacc_start=5e-5, abstol_update=10.):
        """
        I.e., solves the optimization for {x: x in X}

        NOTE: only supports non-negativity constraints at the moment.

        NOTE: switching to dual optimization (Dec 19 2023)

        NOTE: adding in solver error robustness (Jan 16 2024)
        - stops trying when inaccuracy is > 1

        Parameters:
        -----------
            y                  (np arr) : observation vector
            abstol_inacc_start (float)  : see [1]
            abstol_update      (float)  : increase factor for abstol_inacc

        Returns:
        --------
            opt_val (float) : optimized objective function value
            optimal (bool)  : 1 if converged
        """
        m, p = self.K.shape
        try_again = True
        abstol_inacc = abstol_inacc_start
        while try_again & (abstol_inacc < 1.):
            try:
                lamb = cp.Variable(m)
                constraint_dual = [self.K.T @ lamb <= 0]
                obj_func = -0.25 * cp.sum_squares(lamb) + lamb @ y
                prob = cp.Problem(
                    objective=cp.Maximize(obj_func),
                    constraints=constraint_dual
                )
                opt_val = prob.solve(
                    solver=self.opt2_solver,
                    verbose=self.opt2_verbose,
                    max_iters=self.max_iters,
                    abstol_inacc=abstol_inacc
                )
                optimal = prob.status == 'optimal'
                try_again = False
            except cp.SolverError:
                abstol_inacc *= abstol_update
                err_msg = 'abstol caused a problem in opt2 -- again with'
                print(f'{err_msg} {abstol_inacc}', file=sys.stderr)

        if abstol_inacc > 1.:
            print('opt2 did not solve', file=sys.stderr)
            raise cp.SolverError

        return opt_val, optimal

    def solve_llr(self, y, mu):
        """
        Solves the log-likelihood hood ratio optimizations.

        Parameters:
        -----------
            y  (np arr) : observation vector
            mu (float)  : functional level set

        Returns:
        --------
            llr_val (float)    : difference of optimized objective functions.
            conv. opt 1 (bool) : convergence indicator for optimization 1
            conv. opt 2 (bool) : convergence indicator for optimization 2
        """
        opt_1 = self.solve_opt1(y=y, mu=mu)
        opt_2 = self.solve_opt2(y=y)
        llr_val = opt_1[0] - opt_2[0]
        if llr_val < -1e-3:
            print(f'Negative LLR Value: {llr_val}')

        return (
            llr_val if llr_val > 1e-8 else 0.,
            opt_1[-1],
            opt_2[-1]
        )


class llrSolver_2d:
    """
    Class for h = (1 -1) -- explicit solution to accelerate computation

    NOTE: This is the Tenorio counter example
    """
    def __init__(self):
        self.h = np.array([1, -1])  # overwrite to particular functional
        self.K = np.identity(2)
        self.batch = False

    def solve_llr(self, y, mu):
        """
        h = (1 -1)

        General analytical solution for mu in mathbb{R}.

        By convention, I set the spanning basis element of the linear subspace
        defined by h (M) to be (sqrt(2)/2 sqrt(2)/2), and the affine space
        defined by A = {x : h^Tx = mu} = M + x_0, where x_0 = (0 -mu)
        """
        mu_pol = -1 if mu <= 0 else 1
        if y[0] + y[1] >= mu_pol * mu:  # S_1
            x_A = np.array([
                0.5 * (y[0] + mu + y[1]),
                0.5 * (y[0] + mu + y[1]) - mu
            ])
        else:  # S_2
            if mu <= 0:
                x_A = np.array([0, -mu])
            elif mu > 0:
                x_A = np.array([mu, 0])
        opt_A = np.dot(y - x_A, y - x_A)

        # optimization B
        if (y[0] >= 0) & (y[1] >= 0):
            opt_B = 0
        elif (y[0] >= 0) & (y[1] < 0):
            opt_B = y[1] ** 2
        elif (y[0] < 0) & (y[1] < 0):
            opt_B = y[0] ** 2 + y[1] ** 2
        else:
            opt_B = y[0] ** 2

        return opt_A - opt_B, x_A


class llrSolver_2d_pos:
    """
    Class for h = (t t) -- explicit solution to accelerate computation
    """
    def __init__(self, t=1):
        self.t = t
        self.h = np.array([t, t])
        self.h_perp = np.array([-t, t])
        self.K = np.identity(2)
        self.batch = False

    def solve_llr(self, y, mu):
        """
        h = (1 -1)

        General analytical solution for mu in mathbb{R}.

        By convention, I set the spanning basis element of the linear subspace
        defined by h (M) to be (sqrt(2)/2 sqrt(2)/2), and the affine space
        defined by A = {x : h^Tx = mu} = M + x_0, where x_0 = (0 -mu)
        """
        intersect_p1 = np.array([0, mu])
        intersect_p2 = np.array([mu, 0])

        # optimization A
        if np.dot(self.h_perp, y) >= np.dot(self.h_perp, intersect_p1):
            opt_A = y[0] ** 2 + (y[1] - mu) ** 2
        elif np.dot(self.h_perp, y) <= np.dot(self.h_perp, intersect_p2):
            opt_A = (y[0] - mu) ** 2 + y[1] ** 2
        else:
            opt_A = (np.dot(self.h, y) - mu) ** 2 / (2 * self.t ** 2)

        # solve optimization B
        if (y[0] >= 0) & (y[1] >= 0):
            opt_B = 0
        elif (y[0] >= 0) & (y[1] < 0):
            opt_B = y[1] ** 2
        elif (y[0] < 0) & (y[1] < 0):
            opt_B = y[0] ** 2 + y[1] ** 2
        else:
            opt_B = y[0] ** 2

        return opt_A - opt_B, opt_A


class llrSolver_3d:
    """
    Class for h = (1 1 -1) -- explicit solution to accelerate computation

    NOTE: This is the Burrus Conjecture counter example

    NOTE: .solve_llr() for this implementation works by accepting a true x and
    a collection of noise realizations.

    The solutions to the two optimizations were found using mathematica and
    implemented in this way to dramatically boost computation speed.
    """
    def __init__(self):
        self.h = np.array([1, 1, -1])  # overwrite to particular functional
        self.K = np.identity(3)
        self.batch = True

    def term13D(self, x, E):
        x1, x2, x3 = x
        e1 = E[:, 0]
        e2 = E[:, 1]
        e3 = E[:, 2]

        x0 = (1/3)*e1
        x4 = (1/3)*e3
        x5 = (2/3)*e2 - x0 + x4
        x6 = (1/3)*e2
        x7 = (2/3)*e1 + x4 - x6
        x8 = (2/3)*e3 + x0 + x6
        x9 = 2*e1
        x10 = 2*e2
        x11 = 2*e3
        x12 = e1**2 + e2**2 + e3**2
        x13 = (1/2)*x1
        x14 = (1/2)*e2
        x15 = (1/2)*e3
        x16 = x14 + x15
        x17 = x13 + x16
        x18 = -x13 + x16
        x19 = x1**2 + x1*x9 + x12
        x20 = x1 + x2
        x21 = -x20
        x22 = 2*x2
        x23 = e2*x22 + x2**2
        x24 = (1/2)*x2
        x25 = (1/2)*e1
        x26 = x15 + x25
        x27 = x24 + x26
        x28 = -x24 + x26
        x29 = x12 + x23
        x30 = -x3
        x31 = x1 + x30
        x32 = x11*x3 + x3**2
        x33 = x2 + x30
        x34 = (1/2)*x3
        x35 = -x14 + x25 - x34
        x36 = -x14 + x25
        x37 = x34 + x36
        x38 = -x37
        x39 = x20 + x30
        x40 = 4*x1 + x9
        x41 = 2*x1
        x42 = x10 + x11
        x43 = (-x39 <= 0)
        x44 = -x10
        x45 = 2*x3
        x46 = 4*x3
        Vs = np.array([-x10*x5 - x11*x8 + x12 + x5**2 + x7**2 - x7*x9 + x8**2, -x10*x17 - x11*x18 + x17**2 + x18**2 + x19, -x11*x21 + x19 + x21**2 + x23, -x11*x28 + x27**2 - x27*x9 + x28**2 + x29, -x10*x31 + x19 + x31**2 + x32, x29 + x32 + x33**2 - x33*x9, -x10*x38 + x12 + x32 + x35**2 - x35*x9 + x38**2])
        Cs = np.array([(-x1 - x7 <= 0) & (-x2 - x5 <= 0) & (-x3 - x8 <= 0), (-x17 - x2 <= 0) & (-x18 - x3 <= 0) & (-e2 + e3 + 3*x1 + x9 <= 0), (x39 <= 0) & (x11 + x22 + x40 <= 0) & (4*x2 + x41 + x42 <= 0), (-x1 - x27 <= 0) & (-x28 - x3 <= 0) & (-e1 + e3 + x10 + 3*x2 <= 0), x43 & (x40 + x44 - x45 <= 0) & (-x41 + x42 + x46 <= 0), x43 & (x11 - x22 + x46 + x9 <= 0) & (4*x2 - x44 - x45 - x9 <= 0), (-x2 + x37 <= 0) & (-x1 + x34 - x36 <= 0) & (e1 + e2 + x11 + 3*x3 <= 0)])
        return Vs[np.argmax(Cs, axis=0), np.arange(Cs.shape[1])]

    def term23D(self, x, E):
        x1, x2, x3 = x
        e1 = E[:, 0]
        e2 = E[:, 1]
        e3 = E[:, 2]
        x0 = 2*e1
        x4 = e1**2 + x0*x1 + x1**2
        x5 = 2*e2
        x6 = e2**2 + x2**2 + x2*x5
        x7 = x4 + x6
        x8 = 2*e3
        x9 = e3**2 + x3**2 + x3*x8
        x10 = (-e1 - x1 <= 0)
        x11 = (-e2 - x2 <= 0)
        x12 = (-e3 - x3 <= 0)
        x13 = (x0 + 2*x1 <= 0)
        x14 = (2*x2 + x5 <= 0)
        x15 = (2*x3 + x8 <= 0)
        Vs = np.array([np.zeros(x4.shape), x4, x6, x7, x9, x4 + x9, x6 + x9, x7 + x9])
        Cs = np.array([x10 & x11 & x12, x11 & x12 & x13, x10 & x12 & x14, x12 & x13 & x14, x10 & x11 & x15, x11 & x13 & x15, x10 & x14 & x15, x13 & x14 & x15])
        return Vs[np.argmax(Cs, axis=0), np.arange(Cs.shape[1])]

    def solve_llr(self, x, E):
        return self.term13D(x, E) - self.term23D(x, E)
