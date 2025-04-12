"""
Classes to handle max quantile estimation. Varieties
1. Quantile regression
2. Random Sample
===============================================================================
Author        : Mike Stanley
Created       : Feb 13, 2024
Last Modified : May 02, 2024
===============================================================================
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss
from tqdm import tqdm


class maxQuantileQR:
    """
    Estimates max quantile over pre-image using quantile regression.

    .estimate() method performs the following steps
    1. Cross validation to find hyperparameters to use in regression model
    2. Fits regression model
    3. Returns in/out sample max predictions

    NOTE: this class currently uses sklearn's gradient boosted regressor with
    pinball loss. But any regression algorithm can work.

    NOTE: the grid of hyperparameter options is set by default in
    .cv_hyperparam_tune.

    NOTE: if this object is instantiated with the hyperparameters_dict
    argument, then the .estimate() does not do cross validation.

    Parameters
    ----------
        X_train (np arr) : M x p
        X_test  (np arr) : M' x p
        y_train (np arr) : M -- LLR samples corresponding to X_train
        q       (float)  : level of quantile to estimate
        hyperparams (dict) : hyperparameters for regressor
    """
    def __init__(self, X_train, X_test, y_train, q, hyperparams):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.q = q
        self.hyperparams_dict = hyperparams
        self.gbt = None
        self.maxq_insample = 0.0
        self.maxq_outsample = 0.0

    def cv_hyperparam_tune(self, random_state_reg=None, random_state_cv=None):
        """
        Tune hyperparameters. For Gradient boosted regressor, we care about
        the following hyperparameters:
        1. max_depth
        2. learning_rate
        3. min_samples_leaf
        4. min_samples_split
        5. n_estimators

        This approach is based on this: https://scikit-learn.org/stable/
        auto_examples/ensemble/plot_gradient_boosting_quantile.html#
        sphx-glr-auto-examples-ensemble-plot-gradient-boosting-quantile-py

        Parameter
        ---------
            random_state_reg (int) : random state of regressor
            random_state_cv  (int) : random state of cv

        Returns
        -------
            sets the self.hyperparams_dict attribute with the above
            hyperparameters as keys
        """
        param_grid = dict(
            learning_rate=[0.01, 0.05, 0.1, 0.15, 0.2],
            max_depth=[2, 5, 10],
            min_samples_leaf=[1, 5, 10, 20],
            min_samples_split=[5, 10, 20, 30, 50],
        )

        neg_mean_pinball_loss_scorer = make_scorer(
            mean_pinball_loss,
            alpha=1 - self.q,
            greater_is_better=False,  # maximize the negative loss
        )

        gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=1 - self.q,
            random_state=random_state_reg
        )

        search_p = HalvingRandomSearchCV(
            gbr,
            param_grid,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=2,
            random_state=random_state_cv
        ).fit(self.X_train, self.y_train)

        # save the results
        self.hyperparams_dict = search_p.best_params_

    def estimate(self, random_state_reg=None, random_state_cv=None):
        """
        Perform the following steps:
        1. Cross validation to find hyperparameters to use in regression model
        2. Fits regression model
        3. set in/out sample max predictions class attributes

        Parameters
        ----------
            random_state_reg (int) : random state of regressor
            random_state_cv  (int) : random state of cv

        Returns
        -------
            sets self.maxq_insample and self.maxq_outsample
        """
        if self.hyperparams_dict is None:

            # set the hyperparameters
            self.cv_hyperparam_tune(
                random_state_reg=random_state_reg,
                random_state_cv=random_state_cv
            )

        # fit quantile regressor
        self.gbt = GradientBoostingRegressor(
            loss='quantile',
            max_depth=self.hyperparams_dict['max_depth'],
            learning_rate=self.hyperparams_dict['learning_rate'],
            min_samples_leaf=self.hyperparams_dict['min_samples_leaf'],
            min_samples_split=self.hyperparams_dict['min_samples_split'],
            n_estimators=self.hyperparams_dict['n_estimators'],
            alpha=1 - self.q
        )
        self.gbt.fit(
            self.X_train, self.y_train
        )

        # compute in/out sample predictions
        self.maxq_insample = self.gbt.predict(self.X_train).max()
        self.maxq_outsample = self.gbt.predict(self.X_test).max()


class maxQuantileRS:
    """
    Estimates max quantile over pre-image using the empirical max. This method
    is only tractable in cases where the true quantile can be sufficiently
    estimated at any given point.

    .estimate() method performs the following steps
    1. for each data point provided in X_train, estimate true quantile
    2. select the empirical max of the above

    NOTE: we assume that llr_solver has attribute "h" for the true functional
    of interest and "K" for the linear forward model.

    Parameters
    ----------
        X_train      (np arr)    : M x p
        llr_solver   (llrSolver) : class with method .solve_llr()
        distr        (scipy)     : distribution object with .rvs() method
        q            (float)     : level of quantile to estimate
        disable_tqdm (bool)      : disable tqdm progress bar for estimate()
    """
    def __init__(self, X_train, llr_solver, distr, q, disable_tqdm):
        self.X_train = X_train
        self.llr_solver = llr_solver
        self.distr = distr
        self.q = q
        self.disable_tqdm = disable_tqdm

    def estimate_quantile(self, x, num_samp=10000, random_seed=None):
        """
        Estimate the 1-qth quantile at x.

        Parameters
        ----------
            x           (np arr) : p x 1 parameter vector
            num_samp    (int)    : number of samples used to estimate quantile
            random_seed (int)    : seed for noise generation

        Returns
        -------
            estimated_q (float)
        """
        # generate noise
        np.random.seed(random_seed)
        noise = self.distr.rvs(size=(num_samp, self.X_train.shape[1]))

        # create syntetic data
        data_syn = self.llr_solver.K @ x + noise  # num_samp x n

        # solve the llr optimizations over sample
        llr_vals = np.zeros(num_samp)

        if self.llr_solver.batch:
            llr_vals = self.llr_solver.solve_llr(
                x=x, E=noise
            )

        else:
            for i in range(num_samp):
                llr_vals[i] = self.llr_solver.solve_llr(
                    y=data_syn[i], mu=np.dot(self.llr_solver.h, x)
                )[0]

        return np.percentile(llr_vals, q=(1 - self.q) * 100)

    def estimate(self, num_samp=10000, random_seeds=None):
        """
        For each data point in self.X_train estimate the 1-qth quantile and
        then pick the empirical max out of all estimated quantiles.

        Parameters
        ----------
            num_samp     (int)    : number of samples for quantile estimates
            random_seeds (np arr) : random seeds for generating noise
                                    across samples

        Return
        ------
            max_quantile_val (float) : the maximum of max_quantiles
                also saves attribute
        """
        self.max_quantiles = np.zeros(self.X_train.shape[0])
        for i in tqdm(range(self.X_train.shape[0]), disable=self.disable_tqdm):
            self.max_quantiles[i] = self.estimate_quantile(
                x=self.X_train[i],
                num_samp=num_samp,
                random_seed=random_seeds[i] if random_seeds is not None
                else None
            )

        return self.max_quantiles.max()
