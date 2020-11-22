
import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import cvxopt
cvxopt.glpk.options["maxiters"] = 5
cvxopt.glpk.options["show_progress"] = False

'''
from cvxopt import solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 0.1
'''

##########################################

def optimize_rpr(error, mean, delay, threshold):
    xb, wb, rpr = np.shape(error)
    assert (np.shape(error) == np.shape(delay))

    weight1 = np.reshape(error, -1)
    weight2 = np.reshape(mean, -1)
    value = np.reshape(delay, -1)

    ##########################################

    selection = cvxpy.Variable((xb * wb * rpr), boolean=True)
    select_constraint = []
    for i in range(0, xb * wb * rpr, rpr):
        # print (i, i + rpr)
        select_constraint.append( cvxpy.sum(selection[i : i + rpr]) == 1 )

    ##########################################

    # cvxpy.atoms.elementwise.abs.abs(weight2 @ selection) <= threshold
    weight_constraint1 = weight1 @ selection <= threshold
    weight_constraint2 = weight2 @ selection <= threshold
    weight_constraint3 = weight2 @ selection >= -threshold

    ##########################################

    total_value = value @ selection

    ##########################################

    knapsack_problem = cvxpy.Problem(cvxpy.Minimize(total_value), select_constraint + [weight_constraint1, weight_constraint2, weight_constraint3])

    ##########################################

    # knapsack_problem.solve(solver=cvxpy.GLPK_MI)
    # print(cvxpy.installed_solvers())
    # ['CVXOPT', 'ECOS', 'GLPK', 'GLPK_MI', 'OSQP', 'SCS']
    knapsack_problem.solve(solver='GLPK_MI')

    if knapsack_problem.status != "optimal":
        print("status:", knapsack_problem.status)
        print (selection.value)
        assert (knapsack_problem.status == "optimal")

    ##########################################

    select = np.array(selection.value, dtype=np.int32)
    rpr_lut = np.reshape(select, (xb, wb, rpr))

    ones = np.sum(rpr_lut, axis=2)
    assert (np.all(ones == 1))

    scale = np.arange(1, rpr + 1, dtype=np.int32)
    rpr_lut = np.sum(rpr_lut * scale, axis=2)
    assert (np.all(rpr_lut > 0))

    ##########################################
    
    return rpr_lut
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
