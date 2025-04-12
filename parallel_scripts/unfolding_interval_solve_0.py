"""
Script 0 for computing five intervals based on the output from
unfolding_mcmc_chain_gen_0.py (for the parameter samples) and
unfolding_llr_samp_qr_fit_0.py (for the predicted QR maxes, llr samples and
OOS QR predictions).

This scripts is responsible for observations 0-199 and reads in the chains
generated from unfolding_mcmc_chain_gen_0.py.

The code in this script was developed in
../dev_notebooks/test_inversion_code_sandbox.ipynb.

The output contains the following five intervals for the 200 observations:
1. MQ Direct: direct test inversion using the max OOS predicted quantile
2. MQ Optimized: adaOSB optimized interval with max OOS predicted quantile
3. MQmu Parameter: direct test inversion with individual predicted quantiles
4. MQmu Functional: test inversion using estimated max quantile function curve
5. OSB: original chi2_1 interval for comparison
===============================================================================
Author        : Mike Stanley
Created       : Apr 17, 2024
Last Modified : Apr 12, 2025
===============================================================================
"""
import json
import multiprocessing as mp
from time import time
from adaFuncCI.inversion_intervals import solve_llr_fixed_y
from adaFuncCI.inversion_intervals import direct_inversion
from adaFuncCI.inversion_intervals import max_local_quantile_inversion
from adaFuncCI.optimize import osb_int
import numpy as np
from scipy import stats


def compute_intervals(
    i, y, qoi_vals, q_hat_vals, K, h,
    alpha, func_inv_method, hyperparams,
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

    NOTE: the GP parameters are stored in dictionary with keys
    1. num_bins
    2. len_scale
    3. variance

    Parameters
    ----------
        i               (int)    : iteration index for parallelization
        y               (np arr) : observation vector
        qoi_vals        (np arr) : function values at test points
        q_hat_vals      (np arr) : QR predictions at test points
        K               (np arr) : linear forward model
        h               (np arr) : linear functional
        alpha           (float)  : 1 - confidence level
        func_inv_method (str)    : method for the functional inversion -- GP
                                   reg or rolling max
        hyperparams  (dict)      : dict with func_inv_method parameters
        disable_tqdm    (bool)   : toggles progress bar for llr values

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
    mq_direct = direct_inversion(
        qoi_vals=qoi_vals,
        llr_vals_qoi=llr_vals_qoi_test,
        q_hat_vals=q_hat_vals,
        local=False
    )

    # --- MQ Opt
    mq_opt = osb_int(y=y, q=q_hat_vals.max(), K=K, h=h)

    # --- MQmu param
    mq_mu_param = direct_inversion(
        qoi_vals=qoi_vals,
        llr_vals_qoi=llr_vals_qoi_test,
        q_hat_vals=q_hat_vals,
        local=True
    )

    # --- MQmu func
    mq_mu_func = max_local_quantile_inversion(
        qoi_vals=qoi_vals,
        llr_vals_qoi=llr_vals_qoi_test,
        q_hat_vals=q_hat_vals,
        method=func_inv_method,
        hyperparams=hyperparams
    )[0]

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


if __name__ == '__main__':

    # operational parameters
    OBS_IDXS = (0, 200)
    NUM_CHAINS = OBS_IDXS[1] - OBS_IDXS[0]
    IDX = 0
    EXP_IDX = 5
    NUM_CPU = None

    # define method and dictionary for functional space inversion technique
    FUNC_INV_METH = 'rolling'
    # HYP_DICT = {'num_bins': 20, 'len_scale': 1e2, 'variance': 0.1}
    HYP_DICT = {'T': 1000, 'center': True}

    # file paths
    EXP_OBJ_FP = '../data/unfold_true_dim80_smear_dim40'
    EXP_OBJ_FP += '/experiment_objects_adversarial_argmax.npz'
    MCMC_PARAM_FP = '../data/parameter_settings_mcmc'
    MCMC_PARAM_FP += f'/mcmc_parameters{EXP_IDX}.json'

    # unfolding objects
    with open(EXP_OBJ_FP, 'rb') as f:
        exp_obj = np.load(f)
        data_tilde = exp_obj['data_tilde']
        K_tilde = exp_obj['K_tilde']
        H = exp_obj['H']

    # mcmc parameters
    with open(MCMC_PARAM_FP, 'rb') as f:
        mcmc_params = json.load(f)

    # compute gamma quantile
    gamma = mcmc_params["alpha"] - mcmc_params["eta"]

    # define functional of interest
    h = H[mcmc_params["functional_idx"]]

    # parameter samples/chains
    CHAIN_FP = '/home/mcstanle/adaOSB_hydra_data/unfolding_exp'
    CHAIN_FP += f'/chains{IDX}_func{mcmc_params["functional_idx"]}'
    CHAIN_FP += f'_exp{EXP_IDX}.npy'

    with open(CHAIN_FP, 'rb') as f:
        chain_arr = np.load(f)[
            :, :, mcmc_params["burn_in"]:, :
        ]  # chains, train/test, chain len, dim
    print(f'Chain array size: {chain_arr.shape}')

    # read in OOS QR predictions
    QR_PRED_FP = '/home/mcstanle/adaOSB_hydra_data/unfolding_exp'
    QR_PRED_FP += f'/qr_maxes{IDX}_func{mcmc_params["functional_idx"]}'
    QR_PRED_FP += f'_exp{EXP_IDX}_BBcalib.npz'
    with open(QR_PRED_FP, 'rb') as f:
        data_obj = np.load(f)
        q_hat_test = data_obj['output_data_oos_pred']

    # determine number of CPUs to use
    pool = mp.Pool(NUM_CPU if NUM_CPU else mp.cpu_count())
    print('Number of available CPUs: %i' % mp.cpu_count())
    print('Starting parallelization...')
    START = time()

    output_intervals = np.zeros(shape=(NUM_CHAINS, 5, 2))  # 5 intervals

    def collect_data(data):
        idx = data[0]
        output_intervals[idx, :, :] = np.vstack(data[1:])

    for i in range(NUM_CHAINS):
        pool.apply_async(
            compute_intervals,
            args=(
                i,  # iteration index
                data_tilde[i + OBS_IDXS[0]],
                chain_arr[i, 1, :, :] @ h,  # test functional values
                q_hat_test[i, :],
                K_tilde,
                h,
                mcmc_params["alpha"],
                FUNC_INV_METH,
                HYP_DICT
            ),
            callback=collect_data
        )
    pool.close()
    pool.join()

    SAVE_FP = '/home/mcstanle/adaOSB_hydra_data/unfolding_exp'
    SAVE_FP += f'/intervals{IDX}_func{mcmc_params["functional_idx"]}'
    SAVE_FP += f'_exp{EXP_IDX}_rolling_BBcalib.npy'
    with open(SAVE_FP, 'wb') as f:
        np.save(file=f, arr=output_intervals)
    print(f"Done. Elapsed time {(time() - START) / 60}")
