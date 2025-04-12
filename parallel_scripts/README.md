Contains both the key to describe the settings for each experiment in addition to the operational instructions for executing the scripts in this directory.

# Experiment Key
| Experiment Number | alpha | eta  | chain length | Number of HP | Radius | Burn in | functional IDX | polytope type  | mcmc alg | true x    |
| ----------------- | ----- | ---- | ------------ | ------------ | ------ | ------- | -------------- | -------------- | -------- | --------- 
| 0                 | 0.32  | 0.01 | 25k          | 200          | 0.5    | 10k     | 6              | eigen + random | vaidya   | true mean |
| 1                 | 0.32  | 0.05 | 25k          | 200          | 0.5    | 10k     | 6              | eigen + random | vaidya   | true mean |
| 2                 | 0.32  | 0.1  | 25k          | 200          | 0.5    | 10k     | 6              | eigen + random | vaidya   | true mean |
| 3                 | 0.32  | 0.01 | 21k          | 200          | 0.5    | 0       | 6              | eigen + random | vaidya   | true mean |
| 4                 | 0.32  | 0.01 | 21k          | 200          | 0.5    | 0       | 6              | eigen + random | vaidya   | Pau Adv   |
| 5                 | 0.32  | 0.01 | 21k          | 200          | 0.5    | 0       | 6              | eigen + random | vaidya   | Argmax arv   |

To aid in organization, there are separate json files for each of these experiments. MCMC files are kept in `../data/parameter_settings_mcmc/` while QR files are kept in `../data/parameter_settings_qr`.

## Misc Experiment Notes
1. Experiment 3 is the first to incorporate the mixed sampler. Due to its superior sampling coverage, the QR parameters are notably different than the other experiements.

# Operational Instructions
There are two types of files in this directory; MCMC chain sampling files and LLR sampling quantile regresion (QR) fitting files. For all experiments, we sample 1000 realizations of data. These are used to estimate interval coverage and length. For each file type, there are five scripts to aid in the parallelization. The scripts are split up in this way because the MCMC algorithm is not parallelzable across data realizations. It is unclear why this is the case, but it is a fact we noticed in practice.

Since the parameter contolling each run are organizing according to experiment number as denoted in the above table, to run the scripts, one simply nees to modify the `EXP_IDX` operational parameter in each script.

__Steps__
1. Update `EXP_IDX` in code
2. Push from local and pull on hydra machine
3. Creat screen session and activate correct conda environment: `mc_sampling`
4. Clean up scratch in `/home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch`
4. Run one of the following commands

Use the following command to run
`python unfolding_mcmc_chain_gen_0.py 1> /home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stdout_mcmc_0.txt 2> /home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stderr_mcmc_0.txt`
`python unfolding_llr_samp_qr_fit_0.py 1> /home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stdout_llr_qr_0.txt 2> /home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stderr_llr_qr_0.txt`
`python unfolding_llr_samp_qr_fit_0.py 1>/home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stdout_llr_qr_0_bbcal.txt 2>/home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stderr_llr_qr_0_bbcalib.txt`
`python unfolding_interval_solve_0.py 1>/home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stdout_interval_0_bbcal.txt 2>/home/mcstanle/adaOSB_hydra_data/unfolding_exp/scratch/stderr_interval_0_bbcalib.txt`