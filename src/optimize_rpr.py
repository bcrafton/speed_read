
import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
    
##########################################

def optimize_rpr(error, delay, threshold):
    xb, wb, rpr = np.shape(error)
    assert (np.shape(error) == np.shape(delay))

    weight = np.reshape(error, -1)
    value = np.reshape(delay, -1)

    ##########################################

    selection = cvxpy.Variable((xb * wb * rpr), boolean=True)
    select_constraint = []
    for i in range(0, xb * wb * rpr, rpr):
        select_constraint.append( cvxpy.sum(selection[i : i + rpr]) == 1 )

    ##########################################

    weight_constraint = weight @ selection <= threshold

    ##########################################

    total_value = value @ selection

    ##########################################

    knapsack_problem = cvxpy.Problem(cvxpy.Minimize(total_value), select_constraint + [weight_constraint])

    ##########################################

    knapsack_problem.solve(solver=cvxpy.GLPK_MI, verbose=True)

    ##########################################

    select = np.array(selection.value, dtype=int)
    rpr_lut = np.reshape(select, (xb, wb, rpr))

    ones = np.sum(rpr_lut, axis=2)
    assert (np.all(ones == 1))

    scale = np.arange(1, rpr + 1, dtype=int)
    rpr_lut = np.sum(rpr_lut * scale, axis=2)
    assert (np.all(rpr_lut > 0))

    ##########################################
    
    return rpr_lut
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
