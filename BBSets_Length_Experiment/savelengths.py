import sys
from optimization_utils import *
import os
#name_of_run = 'new_quantiles_run1'


import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Set the name of the run from the command line.")
parser.add_argument("--nameofrun", type=str, required=True, help="The name of the run")

# Parse arguments
args = parser.parse_args()

# Access the nameofrun argument
name_of_run = args.nameofrun

directory = ''
LLR3D_vec = jax.jit(LLR3D_vec)
h = h3

Nfiles = 50
α = 0.32
#(1-α) = (1-η)(1-γ), in particular 1-η > 1-α and so 0 < η < α
# η --> 0 recovers the non BB setting 
ηs = np.linspace(0, 0.32, 500)[1:-1]
η_smallest = ηs[0] #We will go from ths to α
#ηs = onp.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
γs = α + ηs#1-(1-α)/(1-ηs) #I will only ever be interested at quantiles at levels 1-γ for γ in γs

files = [name_of_run+str(i) for i in range(1,Nfiles+1)] #Double check if they start at 0 or 1

l = []
for i in t(files):
    points_file = directory+'points'+i+'output.csv'
    quantiles_file = directory+'quantiles'+i+'output.csv'
    M1 = onp.loadtxt(points_file,delimiter = ',')[2:]
    M2 = onp.loadtxt(quantiles_file,delimiter = ',')
    M1 = M1[-len(M2):]
    assert(len(M1) == len(M2))
    joined = np.hstack((M1, M2))
    l.append(joined)

MM = np.vstack(l) #x_0, x_1, x_2, and the gamma quantiles

l = []
for i in t(files):
    points_file = directory+'points'+i+'output.csv'
    quantiles_file = directory+'quantiles'+i+'output.csv'
    M1 = onp.loadtxt(points_file,delimiter = ',')[:2]
    #print(M1)
    l.append(M1[-1])
    #M2 = onp.loadtxt(quantiles_file,delimiter = ',')
    #assert(len(M1) == len(M2))
    #joined = np.hstack((M1, M2))
    #l.append(joined)#np.append(big_array, joined, axis = 0)

ys = onp.array(l)

# E = onp.random.normal(size = (int(1e8),3))
# samples = LLR3D_vec([0,0,5],E)
# onp.quantile(samples, 1-α)
#max_q_η0 = 1.1642553806304932

def filtr(p, r):
    #Returns the points in the database such that ||x-p||<= r
    return MM[np.linalg.norm(MM[:,:3]-p, axis = 1)<= r]

def term1(y):
    z = cp.Variable(3)
    prob = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(y - z)),
        constraints=[z >= 0])
    opt = prob.solve(solver=cp.ECOS)
    return opt

max_q_η0 = 1.1642553806304932
#This is the lengthy part!
data = onp.zeros((len(ys), len(ηs)+1))
for a, y in t(enumerate(ys)):
    print(a, '/', len(ys))
    t1 = term1(y)
    data[a,0] = t1 + max_q_η0
    for b,η in enumerate(ηs):
        c_η = onp.sqrt(chi2(df = 3).ppf(1-η))
        filtered = filtr(y, c_η)
        if len(filtered) > 10000:
            approx_max_quantile = np.max(filtered[:, 3+b])
            if c_η <= t1+ approx_max_quantile:
                pass
                #print(c_η, t1, approx_max_quantile)
            data[a,b+1] = min(c_η, t1 +approx_max_quantile)
        else:
            data[a, b+1] = -1

valid_rows = np.all(data != -1, axis=1)
# Keep only the valid columns
filtered_array = data[valid_rows]
good_ys = ys[valid_rows]
#filtered_array.shape
np.save('filtered_array'+name_of_run+'.npy', filtered_array)
np.save('good_ys'+name_of_run+'.npy', good_ys)
print('finished correctly')
#good_ys = ys[valid_rows]
#lengths = onp.zeros(filtered_array.shape)
#for i in t(range(len(filtered_array))):
#    y = good_ys[i]
#    for j in range(len(filtered_array)[0]):
#        constant = filtered_array[i,j]
#        interval = interval_opt(y, constant, h)
#        lengths[i,j] = interval[1]-interval[0]

#np.save('lengths_final'+name_of_run+'.npy', lengths)
