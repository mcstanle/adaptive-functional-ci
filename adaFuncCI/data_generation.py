"""
Objects and functions to generate simulation data.
===============================================================================
Author        : Mike Stanley
Created       : Feb 08, 2024
Last Modified : Feb 20, 2024
===============================================================================
"""
import json
import numpy as np
import os
from scipy import stats
from scipy.integrate import quad


class dataGenerator:
    """
    Parent class to data generation objects.
    """
    def sample(self):
        """ Sample from the data generating process"""
        pass

    def sample_ensemble(self):
        """ Sample n times from the data generating process """
        pass


class unfoldingGenerator(dataGenerator):
    """
    Capabilities:
    1. Generate histogram data for the unfolding problem.
    2. create K matrix and bin means
    3. sample data

    NOTE: Generates data from the Gaussian mixture model and reads in
    generation parameters from simulation_model_parameters.json.

    NOTE: by convention, K and true/smear means are stored in
    ./data/unfold_true_dimXX_smear_dimYY where XX is the dimension of the true
    vector and YY the dimension of the smear vector.
    - if the directory exists, then the forward matrix and bin means are
      loaded in
    - else, the directory is created and the the files are created

    Parameters
    ----------
        dim_true     (int) : dimension of true bins
        dim_smear    (int) : dimension of smeared bins
        parameter_fp (str) : path to parameter file
    """
    def __init__(
        self,
        dim_true,
        dim_smear,
        parameter_fp=os.getcwd() + '/data/simulation_model_parameters.json'
    ):
        # Get the directory of the current file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        obj_dir = module_dir + '/../data'

        # super().__init__()
        self.parameter_fp = parameter_fp if (parameter_fp is None) else \
            obj_dir + '/simulation_model_parameters.json'
        self.data_gen_dict = self.get_param_dict(self.parameter_fp)
        self.dim_true = dim_true
        self.dim_smear = dim_smear
        self.edges_true = np.linspace(
            self.bin_bounds[0], self.bin_bounds[1], num=self.dim_true + 1
        )
        self.edges_smear = np.linspace(
            self.bin_bounds[0], self.bin_bounds[1], num=self.dim_smear + 1
        )

        # check if there are already K and means computed
        obj_dir += f'/unfold_true_dim{self.dim_true}_smear_dim{self.dim_smear}'

        if os.path.isdir(obj_dir):

            # load objects
            with open(obj_dir + '/K.npy', 'rb') as f:
                self.K = np.load(f)
            with open(obj_dir + '/means_true.npy', 'rb') as f:
                self.means_true = np.load(f)
            with open(obj_dir + '/means_smear.npy', 'rb') as f:
                self.means_smear = np.load(f)
        else:

            # create directory
            os.mkdir(obj_dir)

            # compute objects
            self.K = self.compute_K_gmm()
            true_means, smear_means = self.compute_GMM_bin_means()
            self.means_true = true_means
            self.means_smear = smear_means

            # save objects
            with open(obj_dir + '/K.npy', 'wb') as f:
                np.save(file=f, arr=self.K)
            with open(obj_dir + '/means_true.npy', 'wb') as f:
                np.save(file=f, arr=self.means_true)
            with open(obj_dir + '/means_smear.npy', 'wb') as f:
                np.save(file=f, arr=self.means_smear)

        # transformation to unit covariance
        self.Sigma_data = np.diag(self.means_smear)
        self.L_data = np.linalg.cholesky(self.Sigma_data)
        self.L_data_inv = np.linalg.inv(self.L_data)

        # transform the matrix
        self.K_tilde = self.L_data_inv @ self.K

    def get_param_dict(self, fp):
        """
        Reads in parameter file and unpacks parameters.

        Parameters
        ----------
            fp (str) : path to parameter file

        Returns
        -------
            dictionary with keys "gmm", "wide_bin_deconvolution_bins" and
            "smear_strenth"
        """
        with open(fp, 'rb') as f:
            param_dict = json.load(f)
        self.pi = param_dict['gmm']['pi']
        self.mu = param_dict['gmm']['mu']
        self.sigma = param_dict['gmm']['sigma']
        self.T = param_dict['gmm']['T']
        self.sigma_n = param_dict['gmm']['sigma_n']
        self.bin_bounds = (
            param_dict['wide_bin_deconvolution_bins']['bin_lb'],
            param_dict['wide_bin_deconvolution_bins']['bin_ub']
        )
        self.sigma_smear = param_dict['smear_strength']

        return param_dict

    def sample_gmm(self):
        """
        Sample points from the gaussian mixture model. Reads pi, mu, sigma and
        T parameters from read-in parameters.

        Parameters:
        -----------
            None

        Returns:
        --------
            sampled_data (np arr) : sampled data
        """
        # sample the number of data points
        tau = stats.poisson(mu=self.T).rvs()

        # select mixture components
        mix_comps = stats.bernoulli.rvs(p=self.pi[1], size=tau)

        # generate normal data
        comp0_samp = stats.norm(loc=self.mu[0], scale=self.sigma[0]).rvs(tau)
        comp1_samp = stats.norm(loc=self.mu[1], scale=self.sigma[1]).rvs(tau)

        # generate data
        sampled_data = np.zeros(tau)
        for i in range(tau):
            if mix_comps[i] == 1:
                sampled_data[i] = comp1_samp[i]
            else:
                sampled_data[i] = comp0_samp[i]

        return sampled_data

    def generate_hists(self, num_bins_true, num_bins_smear):
        """
        Generate one realization of histogram data

        Parameters:
        -----------
            num_bins_true  (np arr) : # of bins to use for the true hist.
            num_bins_smear (np arr) : # of bins to use for the smeared hist.

        Returns:
        --------
            tuple of histogram output (bin counts, bin edges)
            - true histogram
            - smeared histogram
        """
        mix_data = self.sample_gmm()
        smearing_noise = stats.norm(
            loc=0, scale=self.sigma_smear
        ).rvs(len(mix_data))
        smeared_mix_data = mix_data + smearing_noise

        # generate the histogram data
        hist_true = []
        hist_smear = []

        for num_bin in num_bins_true:

            hist_true_i = np.histogram(
                a=mix_data,
                bins=np.linspace(
                    self.bin_bounds[0], self.bin_bounds[1], num=(num_bin + 1)
                )
            )
            hist_true.append(hist_true_i)

        for num_bin in num_bins_smear:
            hist_smear_i = np.histogram(
                a=smeared_mix_data,
                bins=np.linspace(
                    self.bin_bounds[0], self.bin_bounds[1], num=(num_bin + 1)
                )
            )
            hist_smear.append(hist_smear_i)

        return hist_true, hist_smear

    def intensity_f(self, x):
        """
        Evaluate the intensity function at x.

        In accordance with the data generating process, the intensity function
        is based on the GMM.

        Parameters
        ----------
            x     (float)  : value at which to evaluate the intensity func

        Returns
        -------
            float of intensity function evaluation
        """
        norm0 = stats.norm(loc=self.mu[0], scale=self.sigma[0])
        norm1 = stats.norm(loc=self.mu[1], scale=self.sigma[1])

        return_val = self.pi[0] * self.T * norm0.pdf(x)
        return_val += + self.pi[1] * self.T * norm1.pdf(x)
        return return_val

    def compute_K_gmm(self):
        """
        Compute the smearing matrix K for the GMM model.

        Parameters
        ----------
            None

        Returns
        -------
            K (np array) : dimension dim_smear X dim_true
        """
        K = np.zeros(shape=(self.dim_smear, self.dim_true))

        for j in range(self.dim_true):

            # compute the denominator
            denom_eval = quad(
                func=self.intensity_f,
                a=self.edges_true[j],
                b=self.edges_true[j + 1]
            )

            for i in range(self.dim_smear):

                # compute the numerator
                int_eval = quad(
                    func=lambda x: self.intensity_f(x) * self.inner_int(
                        x, S_lower=self.edges_smear[i],
                        S_upper=self.edges_smear[i + 1]
                    ),
                    a=self.edges_true[j],
                    b=self.edges_true[j + 1]
                )

                K[i, j] = int_eval[0] / denom_eval[0]

        return K

    def inner_int(self, y, S_lower, S_upper):
        """
        Find the inner integral of the K matrix component calc

        Parameters
        ----------
            y       (float) : variable of which we want the inner integral to
                              sbe a function
            S_lower (float) : lower bound of segment
            S_upper (float) : upper bound of segment

        Returns
        -------
            float of inner integral evaluated at y
        """
        upper = (S_upper - y)/self.sigma_smear
        lower = (S_lower - y)/self.sigma_smear
        return stats.norm.cdf(upper) - stats.norm.cdf(lower)

    def compute_GMM_bin_means(self):
        """
        With some GMM intensity function and some domain bidding, compute
        the mean count for each bin in both the true and smeared spaces.

        Parameters
        ----------
            None

        Returns
        -------
            bin mean counts
            - true_means  (np arr)
            - smear_means (np arr)
        """
        NUM_REAL_BINS = self.edges_true.shape[0] - 1
        true_means = np.zeros(NUM_REAL_BINS)

        for i in range(NUM_REAL_BINS):
            true_means[i] = quad(
                func=self.intensity_f,
                a=self.edges_true[i],
                b=self.edges_true[i + 1]
            )[0]

        smear_means = self.K @ true_means

        return true_means, smear_means

    def sample(self):
        """
        Samples one time from data generating distribution using the smear
        bin means and drawing from a poisson distribution accordingly.

        Parameters
        ----------
            None

        Returns
        -------
            y (np arr) : observation vector
        """
        y = np.zeros(self.dim_smear)
        for i in range(self.dim_smear):
            y[i] = stats.poisson(self.means_smear[i]).rvs()
        return y

    def sample_ensemble(self, num_elements, random_seed=None):
        """
        Samples one time from data generating distribution using the smear
        bin means and drawing from a poisson distribution accordingly.

        Parameters
        ----------
            num_elements (int) : number of ensemble elements
            random_seed  (int) : baseline random seed

        Returns
        -------
            y (np arr) : observation vector
        """
        y = np.zeros(shape=(num_elements, self.dim_smear))
        for i in range(self.dim_smear):
            if random_seed:
                np.random.seed(random_seed * (i + 1))
            y[:, i] = stats.poisson(self.means_smear[i]).rvs(num_elements)
        return y
