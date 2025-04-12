"""
Code to sample from the confidence set pre-image and the null LLR
distributions.

Two primary approaches
1. Polytope sampler --> good for high dimensions - uses Vaidya, Dikin, or John
   samplers.
2. Ellipsoid sampler --> good for low dimensions (<10)

NOTE: in the ellipsoid sampler, it is possible for an observation to be such
that no sampled points are within the constraints, in which case the sampler
descends into an infinite loop of doom. In this case, the sampler is
limited to only performing a certain number of while loop iterations.
===============================================================================
Author        : Mike Stanley
Created       : Feb 08, 2024
Last Modified : May 27, 2024
===============================================================================
"""
from adaFuncCI.optimize import ssb_int
import cvxpy as cp
import numpy as np
import polytopewalk.pwalk as pwalk
from scipy import stats
from sklearn.preprocessing import normalize
# import sympy
from tqdm import tqdm


class PreimageSampler:
    """
    Parent class of the pre-image sampler

    Parameters:
    -----------
        y   (np arr) : observed data vector
        eta (float)  : 1 - eta is confidence level of conf. set in obs space
    """
    def __init__(self, y, eta):
        self.y = y
        self.eta = eta

    def sphere_sample(self, N, s, random_seed):
        """ draw from the s - 1 sphere in s dimensions """
        np.random.seed(random_seed)
        sphere_draws = stats.multivariate_normal(
            mean=np.zeros(s), cov=np.identity(s)
        ).rvs(N)
        return sphere_draws

    def sample_conf_set(self):
        pass

    def sample_ensemble(self):
        pass


class polytopeSampler(PreimageSampler):
    """
    Sampler of convex body.

    Available samplers: 'hyperrectangle', 'random', 'eigen', 'eigen+random'

    Two-step procedure to obtain sample points.
    1. Sample support hyperplanes for convex set with different methods
        1. Hyper-rectangle
        2. Simplex-like
        3. Random Polytope
        4. Eigen polytope
        5. Eigen + Random Polytope
    2. Use random walk around polytope to sample points within

    NOTE:
    1. assumes that the forward model has been transformed for unit covariance
    2. Assumes non-negativity constraint on parameters
    3. Supports three random walk samplers; Dikin, Vaidya, and John.
    4. Hyperplanes are sampled upon initialization unless provided explicitly

    Parameters (for init)
    ---------------------
        y             (np arr) : (n x 1) data vector
        eta           (float)  : 1 minus confidence level
        K             (np arr) : (n x p) forward model matrix
        N_hp          (int)    : (m) number of hyperplanes
        r             (float)  : radius parameter for random walk
        A             (np arr) : (m x p) matrix defining hyperplanes
        b             (np arr) : (m x 1) vector defining hyperplanes
        h             (np arr) : functional of interest
        polytope_type (str)    : type of polytope to generate
        alg           (str)    : name of algorithm to use
        disable_tqdm  (bool)   : disable TQDM progress bar
        max_iters     (int)    : max number of iteration for ECOS algo in cvxpy
    """
    def __init__(
        self, y, eta, K,
        N_hp, r, random_seed,
        A=None, b=None, h=None,
        polytope_type='eigen+random',
        alg='vaidya', disable_tqdm=True,
        max_iters=1000, cheb_solver=cp.ECOS
    ):
        super().__init__(y, eta)
        self.K = K
        self.h = h
        self.N_hp = N_hp
        self.r = r
        self.random_seed = random_seed
        self.r_cheb = None
        self.cheb_center = None
        self.disable_tqdm = disable_tqdm
        self.max_iters = max_iters
        self.cheb_solver = cheb_solver
        if alg == 'vaidya':
            self.sampler = pwalk.generateVaidyaWalkSamples
        elif alg == 'john':
            self.sampler = pwalk.generateJohnWalkSamples
        elif alg == 'dikin':
            self.sampler = pwalk.generateDikinWalkSamples
        else:
            raise ValueError('Specified sampler is unsupported')

        # list of permitted polytope types
        POLYTOPE_TYPES = [
            'hyperrectangle',
            'random',
            'eigen',
            'eigen+random'
        ]

        # create polytope if not provided
        if (A is None) and (b is None):

            assert polytope_type in POLYTOPE_TYPES

            if polytope_type == 'hyperrectangle':

                self.A, self.b = self.create_hyperrect_obj()

            elif polytope_type == 'random':

                # create the random hyperplanes
                A_rand, b_rand = self.create_random_hyperplanes()

                # create the hyperrectangle hyperplanes
                A_hr, b_hr = self.create_hyperrect_obj()

                self.A = np.vstack((A_rand, A_hr))
                self.b = np.concatenate((b_rand, b_hr))

            elif polytope_type == 'eigen':

                # create eigen hyperplanes
                A_eig, b_eig = self.create_eig_polytope()

                # create the hyperrectangle hyperplanes
                A_hr, b_hr = self.create_hyperrect_obj()

                self.A = np.vstack((A_eig, A_hr))
                self.b = np.concatenate((b_eig, b_hr))

            elif polytope_type == 'eigen+random':

                # create the random hyperplanes
                A_rand, b_rand = self.create_random_hyperplanes()

                # create eigen hyperplanes
                A_eig, b_eig = self.create_eig_polytope()

                # create the hyperrectangle hyperplanes
                A_hr, b_hr = self.create_hyperrect_obj()

                self.A = np.vstack((A_rand, A_eig, A_hr))
                self.b = np.concatenate((b_rand, b_eig, b_hr))

        else:

            # explicitly set not-used attributes to None
            self.rand_dirs = None
            self.boundary_points = None

            # set A and b with given values
            self.A = A
            self.b = b

        # compute Chebychev center
        cheb_center, r_cheb = self.chebyshev_center(A=self.A, b=self.b)
        self.cheb_center = cheb_center
        self.r_cheb = r_cheb

        # compute SSB interval
        ssb_success = False
        include_slack = False
        while not ssb_success:

            # compute interval
            func_ssb_obj = ssb_int(
                y=self.y,
                K=self.K,
                h=self.h,
                alpha=self.eta,
                include_slack=include_slack
            )
            self.ssb_interval = (func_ssb_obj[0], func_ssb_obj[1])

            # check if bounds are infinite
            if np.abs(self.ssb_interval[0]) == np.inf:
                include_slack = True
            else:
                ssb_success = True

        self.x_lb_ssb = func_ssb_obj[2].value
        self.x_ub_ssb = func_ssb_obj[3].value

    def create_hyperrect_obj(self):
        """
        Creates A and b for polytope sampler

        Parameters
        ----------
            None

        Returns
        -------
            A (np arr)
            b (np arr)
        """
        # dimensions
        m, p = self.K.shape
        q_chi2 = stats.chi2(df=m).ppf(1 - self.eta)

        # create the A matrix
        A = np.zeros(shape=(2 * p, p))

        # create b matrix
        b = np.zeros(2 * p)

        # create unit vector matrix
        id_mat = np.identity(p)

        # filling out A_hyp_rec and b_hyp_rec
        for i in range(p):

            # find the lower/upper support points
            ei_l, ei_u, bi_l, bi_u = self.supporting_hyperplanes(
                norm_vec=id_mat[i], q=q_chi2
            )

            # fill out (A, b) -- 2 constraints per normal vector
            A[2 * i, :] = -id_mat[i]
            A[2 * i + 1, :] = id_mat[i]
            b[2 * i] = -bi_l
            b[2 * i + 1] = bi_u

        return A, b

    def create_random_hyperplanes(self):
        """
        Sample N_hp - 1 random hyperplanes. The final hyperplane is the vector
        (1 ... 1)^T to ensure the polytope is always bounded.

        Parameters
        ----------

        Returns
        -------
            A (np arr) : normal vectors
            b (np arr) : bounds
        """
        # generate hyperplanes
        m, p = self.K.shape
        q_chi2 = stats.chi2(df=m).ppf(1 - self.eta)

        # sample random directions
        self.rand_dirs = self.sphere_sample(
            N=self.N_hp - 1, s=p, random_seed=self.random_seed
        )
        self.rand_dirs = normalize(self.rand_dirs)

        # include one vector in non-positive orthant to ensure boundedness
        self.rand_dirs = np.vstack((self.rand_dirs, -np.ones(p)))

        # STEP 1: obtain boundary points
        self.boundary_points = np.zeros(shape=(self.N_hp, p))
        A = np.zeros(shape=(self.N_hp, p))
        b = np.zeros(shape=self.N_hp)
        for i in tqdm(range(self.N_hp), disable=self.disable_tqdm):

            # find supporting hyperplanes
            Ai_l, Ai_u, bi_l, bi_u = self.supporting_hyperplanes(
                norm_vec=self.rand_dirs[i, :], q=q_chi2
            )

            # find intersection of supporting hyperplane and set
            self.boundary_points[i, :] = -Ai_l
            A[i, :] = -Ai_l
            b[i] = -bi_l

        return A, b

    def create_eig_polytope(self, max_iters=5000):
        """
        Polytope defined by principal axes of confidence set ellipsoid. Uses
        2xp components defined by the eigenvectors of the ellipsoid

        Parameters
        ----------
            max_iters (int) : maximum number of optimization steps in cvxpy

        Returns
        -------
            A (np arr)
            b (np arr)
        """
        # problem dimensions
        n, p = self.K.shape
        q_chi2 = stats.chi2(df=n).ppf(1 - self.eta)

        # determine principal axis directions
        Omega, P = np.linalg.eigh(self.K.T @ self.K, UPLO='L')

        # create the A matrix
        A = np.zeros(shape=(2 * p, p))

        # create b matrix
        b = np.zeros(2 * p)

        # eigenvector portion portion
        for i in range(p):

            # find the lower/upper support points
            ei_l, ei_u, bi_l, bi_u = self.supporting_hyperplanes(
                norm_vec=P[:, i], q=q_chi2,
                max_iters=max_iters
            )

            # fill out (A, b) -- 2 constraints per normal vector
            A[2 * i, :] = -P[:, i]
            A[2 * i + 1, :] = P[:, i]
            b[2 * i] = -bi_l
            b[2 * i + 1] = bi_u

        return A, b

    def chebyshev_center(self, A, b, verbose=False):
        """
        Computes Chebyshev center of polytope.

        Parameters
        ----------
            A (np arr) : vectors defining hyperplane normal vectors
            b (np arr) : vector defining hyperplane levels

        Returns
        -------
        """
        x_cheb = cp.Variable(self.K.shape[1])
        r = cp.Variable(1)
        cheb_constraints = [
            A[i, :] @ x_cheb + r * np.linalg.norm(A[i, :]) <= b[i]
            for i in range(A.shape[0])
        ]
        prob_cheb = cp.Problem(
            objective=cp.Maximize(r),
            constraints=cheb_constraints
        )
        prob_cheb.solve(
            solver=self.cheb_solver, verbose=verbose, max_iters=self.max_iters
        )
        # assert prob_cheb.status == 'optimal'

        return x_cheb.value, r.value

    def supporting_hyperplanes(
        self, norm_vec, q,
        solver=cp.ECOS, verbose=False, max_iters=1000
    ):
        """
        Given a direction, finds supporting hyperplanes for the pre-image of
        the set {y: (y_obs - y)^T Sigma_inv (y_obs - y) <= chi^2_m} intersected
        with the non-negative orthant.

        NOTE: finds both min and max of hyperplane over confidence set.

        Parameters
        ----------
            norm_vec  (np arr) : normal vector defining hyperplane
            q         (np arr) : quantile of the quad term (typically chi2_m)
            solver    (cp obj) : solver for the optimization
            verbose   (bool)   :
            max_iters (int)    : max number of iterations

        Returns
        -------
            x_min.value    (np arr) : min int. point of h-plane and constraints
            x_max.value    (np arr) : max ""
            prob_min.value (float)  : min hyperplane level set value
            prob_max.value (float)  : max ""
        """
        p = norm_vec.shape[0]

        # minimization
        x_min = cp.Variable(p)
        constraints_min = [
            cp.sum_squares(self.y - self.K @ x_min) <= q,
            x_min >= 0
        ]
        prob_min = cp.Problem(
            objective=cp.Minimize(norm_vec @ x_min),
            constraints=constraints_min
        )
        prob_min.solve(solver=solver, verbose=verbose, max_iters=max_iters)

        # maximization
        x_max = cp.Variable(p)
        constraints_max = [
            cp.sum_squares(self.y - self.K @ x_max) <= q,
            x_max >= 0
        ]
        prob_max = cp.Problem(
            objective=cp.Maximize(norm_vec @ x_max),
            constraints=constraints_max
        )
        prob_max.solve(solver=solver, verbose=verbose, max_iters=max_iters)

        return x_min.value, x_max.value, prob_min.value, prob_max.value

    def sample_ensemble(
        self, M,
        start_pos=None,
        random_seed=None,
        burn_in=0
    ):
        """
        Sample from convex body defined by the norm
        residual constraint and the parameter constraints.

        Parameters
        ----------
            M           (int)    : number of ensemble elements to generate
            start_pos   (np arr) : if none use chebyshev point
            random_seed (int)    : seed for random directions
            burn_in     (int)    : number of steps for sampler to approximate a
                                   warm start.

        Returns
        -------
            sampled_points (np arr) : (M, p) - points sampled from region.
        """
        # starting position
        if start_pos is None:
            x_start = self.cheb_center
        else:
            x_start = start_pos.copy()

        # sample points within generated polytope
        np.random.seed(random_seed)
        sampled_points = self.sampler(
            x_start, self.A, self.b, self.r, M + burn_in
        )[:, -M:].T

        return sampled_points

    def sample_mixed_ensemble(
        self, M,
        mix_grid=[0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        burn_in=0
    ):
        """
        Generates a sampled set of design points of length M. The
        procedure is to generate K chains of length M / K, flatten them
        together, and shuffle the indices.

        The K chains are distinguished by their starting positions. The
        starting positions are selected along the lines connecting the SSB
        endpoints and the Chebyshev center.

        NOTE: this version of the sampler has no burn-in since the mixing of
        any individual chain does not matter.

        NOTE: see ./dev_notebooks/sampler_properties_in_functional_space.ipynb
        for the original code.

        NOTE: if the Chebyshev center is None (can happen for numerical
        reasons), take the starting points along the lines connecting the SSB
        endpoints.

        Parameters
        ----------
            M           (int)    : number of ensemble elements to generate
            mix_grid    (list)   : mixing coefficients
            burn_in     (int)    : burn-in steps

        Returns
        -------
            chain_flat (np arr) : sample chain
        """
        # create a series of points between SSB points
        grid_size = 2 * len(mix_grid)
        start_points = np.zeros(shape=(grid_size, self.K.shape[1]))

        if self.cheb_center is not None:
            center_point = self.cheb_center
        else:
            center_point = 0.5 * self.x_lb_ssb + 0.5 * self.x_ub_ssb

        for i, t in enumerate(mix_grid):

            # mix with LB
            start_points[i, :] = (1 - t) * self.x_lb_ssb + t * center_point

            # mix with UB
            start_points[
                i + len(mix_grid), :
            ] = (1 - t) * self.x_ub_ssb + t * center_point

        # sample chains
        chain_len = int(M / grid_size)
        chains = np.zeros(shape=(grid_size, chain_len, self.K.shape[1]))
        for i in tqdm(range(grid_size), disable=self.disable_tqdm):

            # sample chain i
            chains[i, :, :] = self.sample_ensemble(
                M=chain_len,
                start_pos=start_points[i],
                burn_in=burn_in
            )

        # create train/test
        # -- flatten array
        chain_flat = np.reshape(
            chains, newshape=(M, self.K.shape[1]), order='C'
        )
        # randomly shuffle the array
        np.random.shuffle(chain_flat)

        return chain_flat


class ellipsoidSampler:
    """
    Given and observation y, samples from the ellipsoid pre-image. Works well
    in low dimensional cases with full-rank linear forward model.

    NOTE: assumes non-negativity constraints

    Parameters
    ----------
        K     (np arr) : linear forward model
        Sigma (np arr) : noise covariance matrix
        y     (np arr) : observed data
        eta   (float)  : 1 minus confidence level of set
    """
    def __init__(self, K, Sigma, y, eta):
        self.K = K
        self.Sigma = Sigma
        self.y = y
        self.eta = eta

        # compute cholesky decomp of (K^T Sigma^-1 K)^-1
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        KT_Sig_K_inv = np.linalg.inv(self.K.T @ self.Sigma_inv @ self.K)
        self.L = np.linalg.cholesky(KT_Sig_K_inv)

    def sphere_sample(self, N, s, random_seed=None):
        """
        Draw from the s - 1 sphere in s dimensions

        Parameters
        ----------
            N           (int) : number of samples
            s           (int) : dimension of sphere
            random_seed (int) : for sphere draws (default None)

        Returns
        -------
            sphere_draws (np arr) : N x s
        """
        np.random.seed(random_seed)
        sphere_draws = stats.multivariate_normal(
            mean=np.zeros(s), cov=np.identity(s)
        ).rvs(N)
        return sphere_draws

    def vgs(self, N, p, random_seed=None):
        """
        Voekler/Gosmann/Stuart algorithm for sampling uniformly at random from
        a ball of dimension n.

        Parameters
        ----------
            N           (int) : number of samples
            p           (int) : dimension of ball
            random_seed (int) : for sphere draws (default None)

        Returns
        -------
        """
        sphere_draws = self.sphere_sample(
            N=N, s=p + 2, random_seed=random_seed
        )
        if N == 1:
            sphere_draws = sphere_draws[np.newaxis, :]
        sphere_draws_norm = normalize(sphere_draws)

        return sphere_draws_norm[:, :p]

    def sample_parameters(
        self, M, random_seed=None, tot_num_rounds=100, return_null=True
    ):
        """
        Sample parameters within the appropriate ball of observation.

        NOTE: assumes non-negative constraint

        NOTE: on the rare occasion that no points are returned, the function
        returns an array of zeros.

        Parameters:
        ----------
            M              (int)  : number of points to sample
            random_seed    (int)  : for sphere draws (default None)
            tot_num_rounds (int)  : number of tries with accept/reject sampler
            return_null    (bool) : switch to return ersatz null array if none

        Returns:
        -------
            x_samp (np arr) : (M, p)
        """
        # problem dimensions
        n, p = self.K.shape

        # create a first batch of sampled points
        chi_quant = np.sqrt(stats.chi2(p).ppf(1 - self.eta))
        x_samp = self.vgs(N=M, p=p, random_seed=random_seed)
        x_samp = chi_quant * x_samp @ self.L.T + self.y

        if tot_num_rounds == 0:
            # drop rows outside constraints
            outside_constr_idx = np.where(
                np.sum(x_samp < 0, axis=1) > 0
            )[0]
            x_samp = np.delete(x_samp, outside_constr_idx, axis=0)

        insuff_samples = True
        prime_idx = 1
        num_rounds = 0
        while insuff_samples and (num_rounds < tot_num_rounds):

            # print(f'Entering resample for {num_rounds}th time')
            # find how many do not satisfy the constraints
            outside_constr_idx = np.where(
                np.sum(x_samp < 0, axis=1) > 0
            )[0]

            num_insuff = outside_constr_idx.shape[0]
            # print(f'Round {num_rounds} | num insuff {num_insuff}')
            if num_insuff == 0:
                insuff_samples = False
                continue

            # generate some new points
            if random_seed is not None:
                # new_random_seed = random_seed * sympy.prime(prime_idx)
                new_random_seed = random_seed * num_rounds
            else:
                new_random_seed = None
            x_samp_supp = self.vgs(
                N=num_insuff, p=p,
                random_seed=new_random_seed
            )
            x_samp_supp = chi_quant * x_samp_supp @ self.L.T + self.y

            # replace the bad points
            x_samp[outside_constr_idx, :] = x_samp_supp

            prime_idx += 1
            num_rounds += 1

            # drop rows outside constraints
            # outside_constr_idx = np.where(
            #     np.sum(x_samp < 0, axis=1) > 0
            # )[0]
            # x_samp = np.delete(x_samp, outside_constr_idx, axis=0)

        if return_null:
            if outside_constr_idx.shape[0] == M:
                x_samp = np.zeros(shape=(100, p))

        return x_samp


class nullTestStatSampler:
    """
    Samples from the null distributions of a test statistic given the parameter
    values, noise distribution, and test statistic.

    Limitations:
    1. assumes deterministic forward model and known noise distribution
    2. linear functional
    3. noise_distr needs to have a .rvs() method (scipy)

    Parameters
    ----------
        noise_distr (scipy)     : distribution object like stats.norm
        test_stat   (llrSolver) : object to compute test statistic
        K           (np arr)    : linear forward model
        h           (np arr)    : linear functional
    """
    def __init__(
        self, noise_distr, test_stat, K, h, disable_tqdm=True
    ):
        self.noise_distr = noise_distr
        self.test_stat = test_stat
        self.K = K
        self.h = h
        self.disable_tqdm = disable_tqdm

    def sample_teststat(self, param_vals, random_seed=None):
        """
        Given sampled parameter values, sample from test stat distributions.

        Parameters
        ----------
            param_vals  (np arr)    : M x p collection of "true" parameters
        """
        # dimensions
        M = param_vals.shape[0]
        n = self.K.shape[0]

        # sample noise
        np.random.seed(random_seed)
        noise = self.noise_distr.rvs(size=(M, n))

        # true functional values
        mu_trues = param_vals @ self.h

        # compute synthetic observations
        obs = param_vals @ self.K.T + noise

        # compute test statistics
        test_stat_samples = np.zeros(M)
        for i in tqdm(range(M), disable=self.disable_tqdm):
            test_stat_samples[i] = self.test_stat.solve_llr(
                y=obs[i], mu=mu_trues[i]
            )[0]

        return test_stat_samples
