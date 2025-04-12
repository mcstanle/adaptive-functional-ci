import cvxpy as cp
import jax 
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.notebook import tqdm as t
from scipy.stats import chi2
import pickle
from joblib import Parallel, delayed

def Lc(μ, y, K, A, b, h):
    n = len(y)
    
    x1 = cp.Variable(n)
    cost1 = cp.sum_squares(K @ x1 - y)
    prob1 = cp.Problem(cp.Minimize(cost1), [h.T@x1 == μ, A@x1 <= b])

    x2 = cp.Variable(n)
    cost2 = cp.sum_squares(K @ x2 - y)
    prob2 = cp.Problem(cp.Minimize(cost2), [A@x2 <= b])
   
    return prob1.solve() - prob2.solve()


def Lc_debug(μ, y, K, A, b, h):
    n = len(y)
    
    x1 = cp.Variable(n)
    cost1 = cp.sum_squares(K @ x1 - y)
    prob1 = cp.Problem(cp.Minimize(cost1), [h.T@x1 == μ, A@x1 <= b])

    x2 = cp.Variable(n)
    cost2 = cp.sum_squares(K @ x2 - y)
    prob2 = cp.Problem(cp.Minimize(cost2), [A@x2 <= b])
   
    return prob1.solve(),prob2.solve()


K2, A2, b2, h2, x_star2 = (np.array([[1., 0.],
        [0., 1.]]), -np.eye(2),
 np.array([0., 0.]),
 np.array([1, -1]),
 np.array([1,1]))
μstar2 = h2.T@x_star2

def Lc1_fast(x, e):
    x1, x2 = x
    e1, e2 = e
    if -e1/2 - e2/2 - x1 <= 0 and -e1/2 - e2/2 - x2 <= 0:
        return e1**2 + 2*((e1/2 + e2/2)**2) - 2*e1*(e1/2 + e2/2) - 2*e2*(e1/2 + e2/2) + e2**2
    elif x1 - x2 <= 0 and 2*e1 + 2*e2 + 4*x1 <= 0:
        return e1**2 + 2*e1*x1 + e2**2 + 2*e2*x1 + 2*x1**2
    elif x2 - x1 <= 0 and 2*e1 + 2*e2 + 4*x2 <= 0:
        return e1**2 + 2*e1*x2 + e2**2 + 2*e2*x2 + 2*x2**2

def Lc2_fast(x, e):
    x1, x2 = x
    e1, e2 = e
    if -e1 - x1 <= 0 and -e2 - x2 <= 0:
        return 0.0
    elif -e2 - x2 <= 0 and 2 * e1 + 2 * x1 <= 0:
        return e1**2 + 2*e1*x1 + x1**2
    elif -e1 - x1 <= 0 and 2 * e2 + 2 * x2 <= 0:
        return e2**2 + 2*e2*x2 + x2**2
    elif 2 * e1 + 2 * x1 <= 0 and 2 * e2 + 2 * x2 <= 0:
        return e1**2 + 2*e1*x1 + e2**2 + 2*e2*x2 + x1**2 + x2**2


def LLR2D(x, e):
    return Lc1_fast(x, e) - Lc2_fast(x, e)

K3, A3, b3, h3, x_star3 = (np.array([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]]),
 np.array([[-1., -0., -0.],
        [-0., -1., -0.],
        [-0., -0., -1.]]),
 np.array([0., 0., 0.]),
 np.array([ 1,1,-1]),
 np.array([0.0, 0.0, 1]))
μ3 = h3.T@x_star3

def Lc3D_fast1(x, e):
    x1, x2, x3 = x
    e1, e2, e3 = e
    # Case 1
    if (-2 * e1 / 3 + e2 / 3 - e3 / 3 - x1 <= 0 and 
        e1 / 3 - 2 * e2 / 3 - e3 / 3 - x2 <= 0 and 
        -e1 / 3 - e2 / 3 - 2 * e3 / 3 - x3 <= 0):
        return (e1**2 + 
                (-e1/3 + 2*e2/3 + e3/3)**2 - 
                2 * e2 * (-e1/3 + 2*e2/3 + e3/3) + 
                (2*e1/3 - e2/3 + e3/3)**2 + 
                (e1/3 + e2/3 + 2*e3/3)**2 - 
                2 * e1 * (2*e1/3 - e2/3 + e3/3) - 
                2 * e3 * (e1/3 + e2/3 + 2*e3/3) + 
                e2**2 + e3**2)
    
    # Case 2
    elif (-e2/2 - e3/2 - x1/2 - x2 <= 0 and 
          -e2/2 - e3/2 + x1/2 - x3 <= 0 and 
          2 * e1 - e2 + e3 + 3 * x1 <= 0):
        return (e1**2 + 
                2 * e1 * x1 + 
                e2**2 + 
                (e2/2 + e3/2 + x1/2)**2 - 
                2 * e2 * (e2/2 + e3/2 + x1/2) + 
                (e2/2 + e3/2 - x1/2)**2 - 
                2 * e3 * (e2/2 + e3/2 - x1/2) + 
                e3**2 + x1**2)

    # Case 3
    elif (x1 + x2 - x3 <= 0 and 
          2 * e1 + 2 * e3 + 4 * x1 + 2 * x2 <= 0 and 
          2 * e2 + 2 * e3 + 2 * x1 + 4 * x2 <= 0):
        return (e1**2 + 
                2 * e1 * x1 + 
                e2**2 + 
                2 * e2 * x2 + 
                e3**2 - 
                2 * e3 * (-x1 - x2) + 
                x1**2 + 
                (-x1 - x2)**2 + 
                x2**2)

    # Case 4
    elif (-e1/2 - e3/2 - x1 - x2/2 <= 0 and 
          -e1/2 - e3/2 + x2/2 - x3 <= 0 and 
          -e1 + 2 * e2 + e3 + 3 * x2 <= 0):
        return (e1**2 + 
                (e1/2 + e3/2 + x2/2)**2 - 
                2 * e1 * (e1/2 + e3/2 + x2/2) + 
                (e1/2 + e3/2 - x2/2)**2 - 
                2 * e3 * (e1/2 + e3/2 - x2/2) + 
                e2**2 + 
                2 * e2 * x2 + 
                e3**2 + 
                x2**2)

    # Case 5
    if (-x1 - x2 + x3 <= 0 and 
        2 * e1 - 2 * e2 + 4 * x1 - 2 * x3 <= 0 and 
        2 * e2 + 2 * e3 - 2 * x1 + 4 * x3 <= 0):
        return (e1**2 + 
                2 * e1 * x1 + 
                e2**2 - 
                2 * e2 * (x1 - x3) + 
                e3**2 + 
                2 * e3 * x3 + 
                x1**2 + 
                (x1 - x3)**2 + 
                x3**2)

    # Case 6
    elif (-x1 - x2 + x3 <= 0 and 
          -2 * e1 + 2 * e2 + 4 * x2 - 2 * x3 <= 0 and 
          2 * e1 + 2 * e3 - 2 * x2 + 4 * x3 <= 0):
        return (e1**2 - 
                2 * e1 * (x2 - x3) + 
                e2**2 + 
                2 * e2 * x2 + 
                e3**2 + 
                2 * e3 * x3 + 
                x2**2 + 
                (x2 - x3)**2 + 
                x3**2)

    # Case 7
    elif (-e1/2 + e2/2 - x1 + x3/2 <= 0 and 
          e1/2 - e2/2 - x2 + x3/2 <= 0 and 
          e1 + e2 + 2 * e3 + 3 * x3 <= 0):
        return (e1**2 + 
                (-e1/2 + e2/2 - x3/2)**2 - 
                2 * e2 * (-e1/2 + e2/2 - x3/2) + 
                (e1/2 - e2/2 - x3/2)**2 - 
                2 * e1 * (e1/2 - e2/2 - x3/2) + 
                e2**2 + 
                e3**2 + 
                2 * e3 * x3 + 
                x3**2)

# Example usage
        
def Lc3D_fast2(x, e):
    x1, x2, x3 = x
    e1, e2, e3 = e

    if -e1 - x1 <= 0 and -e2 - x2 <= 0 and -e3 - x3 <= 0:
        return 0
    elif -e2 - x2 <= 0 and -e3 - x3 <= 0 and 2 * e1 + 2 * x1 <= 0:
        return e1**2 + 2*e1*x1 + x1**2
    elif -e1 - x1 <= 0 and -e3 - x3 <= 0 and 2 * e2 + 2 * x2 <= 0:
        return e2**2 + 2*e2*x2 + x2**2
    elif -e3 - x3 <= 0 and 2 * e1 + 2 * x1 <= 0 and 2 * e2 + 2 * x2 <= 0:
        return e1**2 + 2*e1*x1 + e2**2 + 2*e2*x2 + x1**2 + x2**2
    elif -e1 - x1 <= 0 and -e2 - x2 <= 0 and 2 * e3 + 2 * x3 <= 0:
        return e3**2 + 2*e3*x3 + x3**2
    elif -e2 - x2 <= 0 and 2 * e1 + 2 * x1 <= 0 and 2 * e3 + 2 * x3 <= 0:
        return e1**2 + 2*e1*x1 + e3**2 + 2*e3*x3 + x1**2 + x3**2
    elif -e1 - x1 <= 0 and 2 * e2 + 2 * x2 <= 0 and 2 * e3 + 2 * x3 <= 0:
        return e2**2 + 2*e2*x2 + e3**2 + 2*e3*x3 + x2**2 + x3**2
    elif 2 * e1 + 2 * x1 <= 0 and 2 * e2 + 2 * x2 <= 0 and 2 * e3 + 2 * x3 <= 0:
        return e1**2 + 2*e1*x1 + e2**2 + 2*e2*x2 + e3**2 + 2*e3*x3 + x1**2 + x2**2 + x3**2
    
def LLR3D(x, e):
    return Lc3D_fast1(x, e) - Lc3D_fast2(x, e)


def Lc3dVec1(x, E):
    
    x1, x2, x3 = x
    e1 = E[:,0]
    e2 = E[:,1]
    e3 = E[:,2]

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
    return Vs[np.argmax(Cs, axis = 0), np.arange(Cs.shape[1])] #It is a bit slower but it is safer

    

def Lc3dVec2(x,E):
    
    x1, x2, x3 = x
    e1 = E[:,0]
    e2 = E[:,1]
    e3 = E[:,2]
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
    return Vs[np.argmax(Cs, axis = 0), np.arange(Cs.shape[1])] #It is a bit slower but it is safer


    
"""
       C1 = (-2 * e1 / 3 + e2 / 3 - e3 / 3 - x1 <= 0) & (e1 / 3 - 2 * e2 / 3 - e3 / 3 - x2 <= 0) & (-e1 / 3 - e2 / 3 - 2 * e3 / 3 - x3 <= 0)
    V1 = (e1**2 + (-e1/3 + 2*e2/3 + e3/3)**2 - 2 * e2 * (-e1/3 + 2*e2/3 + e3/3) + 
                (2*e1/3 - e2/3 + e3/3)**2 + 
                (e1/3 + e2/3 + 2*e3/3)**2 - 
                2 * e1 * (2*e1/3 - e2/3 + e3/3) - 
                2 * e3 * (e1/3 + e2/3 + 2*e3/3) + 
                e2**2 + e3**2)

    C2 = (-e2/2 - e3/2 - x1/2 - x2 <= 0) & (-e2/2 - e3/2 + x1/2 - x3 <= 0) & (2 * e1 - e2 + e3 + 3 * x1 <= 0)
    V2 = (e1**2 + 2 * e1 * x1 + 
                    e2**2 + 
                    (e2/2 + e3/2 + x1/2)**2 - 
                    2 * e2 * (e2/2 + e3/2 + x1/2) + 
                    (e2/2 + e3/2 - x1/2)**2 - 
                    2 * e3 * (e2/2 + e3/2 - x1/2) + 
                    e3**2 + x1**2)

    C3 = (x1 + x2 - x3 <= 0) & (2 * e1 + 2 * e3 + 4 * x1 + 2 * x2 <= 0) & (2 * e2 + 2 * e3 + 2 * x1 + 4 * x2 <= 0)
    V3 = (e1**2 + 
                    2 * e1 * x1 + 
                    e2**2 + 
                    2 * e2 * x2 + 
                    e3**2 - 
                    2 * e3 * (-x1 - x2) + 
                    x1**2 + 
                    (-x1 - x2)**2 + 
                    x2**2)

    C4 = (-e1/2 - e3/2 - x1 - x2/2 <= 0) & (-e1/2 - e3/2 + x2/2 - x3 <= 0) & (-e1 + 2 * e2 + e3 + 3 * x2 <= 0)

    V4 = (e1**2 + 
                    (e1/2 + e3/2 + x2/2)**2 - 
                    2 * e1 * (e1/2 + e3/2 + x2/2) + 
                    (e1/2 + e3/2 - x2/2)**2 - 
                    2 * e3 * (e1/2 + e3/2 - x2/2) + 
                    e2**2 + 
                    2 * e2 * x2 + 
                    e3**2 + 
                    x2**2)

    C5 =  (-x1 - x2 + x3 <= 0) & (2 * e1 - 2 * e2 + 4 * x1 - 2 * x3 <= 0) & (2 * e2 + 2 * e3 - 2 * x1 + 4 * x3 <= 0)

    V5 = (e1**2 + 
                    2 * e1 * x1 + 
                    e2**2 - 
                    2 * e2 * (x1 - x3) + 
                    e3**2 + 
                    2 * e3 * x3 + 
                    x1**2 + 
                    (x1 - x3)**2 + 
                    x3**2)

    C6 = (-x1 - x2 + x3 <= 0) & (-2 * e1 + 2 * e2 + 4 * x2 - 2 * x3 <= 0) & (2 * e1 + 2 * e3 - 2 * x2 + 4 * x3 <= 0)

    V6 = (e1**2 - 2 * e1 * (x2 - x3) + 
                    e2**2 + 
                    2 * e2 * x2 + 
                    e3**2 + 
                    2 * e3 * x3 + 
                    x2**2 + 
                    (x2 - x3)**2 + 
                    x3**2)

    C7 = (-e1/2 + e2/2 - x1 + x3/2 <= 0) & (e1/2 - e2/2 - x2 + x3/2 <= 0) & (e1 + e2 + 2 * e3 + 3 * x3 <= 0)

    V7  =(e1**2 + 
                    (-e1/2 + e2/2 - x3/2)**2 - 
                    2 * e2 * (-e1/2 + e2/2 - x3/2) + 
                    (e1/2 - e2/2 - x3/2)**2 - 
                    2 * e1 * (e1/2 - e2/2 - x3/2) + 
                    e2**2 + 
                    e3**2 + 
                    2 * e3 * x3 + 
                    x3**2)
"""


def LLR3D_vec(x, E):
    return Lc3dVec1(x, E) - Lc3dVec2(x, E)



def closest_point(y, verbose=False):
    """
    In the event feasible region is empty, this function is called
    to return the projection of y to the constraint set.
    """
    x = cp.Variable(3)
    prob = cp.Problem(
        objective=cp.Minimize(cp.sum_squares(y - x)),
        constraints=[x >= 0]
    )
    opt = prob.solve(solver=cp.ECOS, verbose=verbose)
    assert prob.status == 'optimal'
    return x.value

def interval_opt(y, q, h, verbose=False):
    """ q is the constraint on the norm residuals """
    x_lb = cp.Variable(3)
    x_ub = cp.Variable(3)
    
    # define optimizations
    prob_lb = cp.Problem(
        objective=cp.Minimize(h @ x_lb),
        constraints=[
            cp.sum_squares(y - x_lb) <= q,
            x_lb >= 0
        ]
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(- h @ x_ub),
        constraints=[
            cp.sum_squares(y - x_ub) <= q,
            x_ub >= 0
        ]
    )
    
    # solve the optimizations
    opt_lb = prob_lb.solve(solver=cp.ECOS, verbose=verbose)
    opt_ub = prob_ub.solve(solver=cp.ECOS, verbose=verbose)
    
    # when intervals diverge return interval with one point
    if prob_lb.status != 'optimal':
        x_proj = closest_point(y)
        out_lb = np.dot(h, x_proj)
    else:
        out_lb = opt_lb
    if prob_ub.status != 'optimal':
        x_proj = closest_point(y)
        out_ub = np.dot(h, x_proj)
    else:
        out_ub = -opt_ub

    return out_lb, out_ub
