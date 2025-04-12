#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../CodeFirstPaper/')
from optimization_utils import *
LLR3D_vec = jax.jit(LLR3D_vec)
import numpy as np
import csv
import os
from tqdm import tqdm

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Set the name of the run and other parameters from the command line.")

# Argument for the name of the run
parser.add_argument(
    "--nameofrun",
    type=str,
    required=True,
    help="The name of the run"
)

# Argument for x as a list of three floats
parser.add_argument(
    "--x",
    type=float,
    nargs=3,  # Expecting exactly three float values
    required=True,
    help="Array x as three space-separated float numbers, e.g., --x 2 2 0"
)

# Argument for alpha as a float
parser.add_argument(
    "--alpha",
    type=float,
    required=True,
    help="Alpha value as a float, e.g., --alpha 0.32"
)

# Parse arguments
args = parser.parse_args()

# Access the nameofrun argument
NAME_RUN = args.nameofrun

# Convert the list of floats to a numpy array
x = np.array(args.x)

# Assign the alpha value
α = args.alpha

# (Optional) Print the parsed values for verification
print(f"Run Name: {NAME_RUN}")
print(f"x: {x}")
print(f"α (alpha): {α}")


#(1-α) = (1-η)(1-γ), in particular 1-η > 1-α and so 0 < η < α
# η --> 0 recovers the non BB setting 

h = h3
ηs = np.linspace(0, 0.32, 500)[1:-1]
η_smallest = ηs[0] #We will go from ths to α
γs = α - ηs #I will only ever be interested at quantiles at levels 1-γ for γ in γs



def sampler(p, r, N):
    """Samples N points from non-negative orthant and the ball centered at p and radius r: ||p-x|| <= r and x>=0"""
    accepted = []
    d = len(p)
    while len(accepted) < N:
        #Sample from the unit sphere
        u = onp.random.normal(0,1,d+2)  # an array of (d+2) normally distributed random variables
        norm = onp.sum(u**2) **(0.5)
        u = u/norm
        x = u[0:d] #take the first d coordinates
        #x\sim U(B(0,1))
        #r*x+p \sim U(B(p,r))
        x = r*x + p
        if onp.min(x) >= 0:
            accepted.append(x)
    return np.array(accepted)


def append_array_to_csv(array, file_path):
    # Ensure the array is a 1D or 2D array
    if len(array.shape) == 1:
        array = array.reshape(1, -1)
    
    # Open the CSV file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write each row of the array to the CSV file
        for row in array:
            writer.writerow(row)


def write_points(x, y, points, id_run):
    file_name = 'points'+id_run+'output.csv'
    append_array_to_csv(x, file_name)
    append_array_to_csv(y, file_name)
    append_array_to_csv(points, file_name)

#Sample y 
for i in tqdm(range(50)):
    y = x + onp.random.normal(size = (3))
    #Create the set {z: ||z-y||^2_2 \leq Q(\chi^2_p; 1-\eta)}
    BB_radius_sq = chi2(df = 3).ppf(1-η_smallest)
    #Sample points in the intersection of BB set and non-negative orthant
    points = sampler(y, onp.sqrt(BB_radius_sq), N = 10000)
    #BB_quantiles = [numerical_quantile(i, level = γ) for i in t(points)]#Parallel(n_jobs = -1)(delayed(lambda aux: numerical_quantile(aux, level = γ))(i) for i in t(points))
    #max_BB_quantile = max(BB_quantiles)
    write_points(x, y, points, NAME_RUN+str(i+1))
    print(i)

def work_on(id_run):
    points = np.loadtxt('points'+id_run+'output.csv', delimiter =',')[2:]
    quantile_file = 'quantiles'+id_run+'output.csv'
    if os.path.exists(quantile_file):
        Npointssofar = len(np.loadtxt(quantile_file, delimiter = ','))
    else:
        Npointssofar = 0
    E = onp.random.normal(size = (int(1e6),3))
    batch_results = []
    N = 1000
    for i in tqdm(range(Npointssofar, len(points))):
        res = LLR3D_vec(points[i], E)
        quantiles = onp.quantile(res, 1-γs)
        batch_results.append(quantiles)
        if (i-Npointssofar + 1) % N == 0 or (i + 1) == len(points):
            batch_array = np.vstack(batch_results)
            append_array_to_csv(batch_array, quantile_file)
            batch_results = []  # Clear the batch list


# In[113]:

print('---')

#work_on('new_run1')
for i in tqdm(range(50)):
    print('working on '+str(i))
    work_on(NAME_RUN+str(i+1))


