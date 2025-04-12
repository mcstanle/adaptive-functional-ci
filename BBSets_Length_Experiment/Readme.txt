In order to reproduce the Berger-Boos set experiments: 

1) Run ExperimentBBSet.py to populate the data folder
Usage

python ExperimentBBSet.py --nameofrun <run_name> --x <x1 x2 x3> --alpha <alpha_value>

Required Arguments

--nameofrun: String identifier for the run
--x: Three space-separated float values representing x-coordinates
--alpha: A float value for the alpha parameter

Examples
# Example 1: Basic usage
python script.py --nameofrun experiment1 --x 2 2 0 --alpha 0.32

nameofrun: Identifies the run in output files/logs
x: Three-dimensional coordinate (x1, x2, x3)
alpha: 1-alpha is the desired coverage

In order to reproduce the results of the paper, the names of runs and parameters are:
ExperimentBBSet.py --nameofrun fixed_220_32 --x 2 2 0 --alpha 0.32
ExperimentBBSet.py --nameofrun fixed_330_32 --x 3 3 0 --alpha 0.32 
ExperimentBBSet.py --nameofrun fixed_550_32 --x 5 5 0 --alpha 0.32


2) Run getlengths.py

python getlengths.py --nameofrun <run_name> 

3) Use the plotting notebook to plot the results of the paper or your own. 

