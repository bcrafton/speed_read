
import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import cvxopt
cvxopt.glpk.options["maxiters"] = 1
cvxopt.glpk.options['show_progress'] = True
# cvxopt.glpk.options['tm_lim'] = 30
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_ON'

cvxopt.solvers.options['maxiters'] = 1
cvxopt.solvers.options['show_progress'] = True
# cvxopt.solvers.options['tm_lim'] = 30
cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_ON'

# /home/brian/env/py3/lib/python3.5/site-packages/cvxpy/reductions/solvers/conic_solvers/glpk_mi_conif.py
# search in here to play with params.

# https://github.com/cvxgrp/cvxpy/pull/254
# https://github.com/cvxgrp/cvxpy/pull/254/commits/c4207f2e2616ac2a3e2e6c906c0660d36dcc6814

# https://github.com/cvxgrp/cvxpy/issues/251
# https://github.com/cvxgrp/cvxpy/issues/351

##########################################

def optimize_rpr(error, mean, delay, threshold):
    xb, wb, step, rpr = np.shape(error)
    assert (np.shape(error) == np.shape(delay))

    weight1 = np.reshape(error, -1)
    weight2 = np.reshape(mean, -1)
    value = np.reshape(delay, -1)

    ##########################################

    selection = cvxpy.Variable((xb * wb * step * rpr), boolean=True)
    select_constraint = []
    for i in range(0, xb * wb * step * rpr, step * rpr):
        # print (i, i + rpr)
        select_constraint.append( cvxpy.sum(selection[i : i + step * rpr]) == 1 )

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
    
    # knapsack_problem.solve(solver='GLPK_MI', options=cvxopt.glpk.options, glpk={'msg_lev': 'GLP_MSG_ON'}, verbose=True)
    knapsack_problem.solve(solver='GLPK_MI', options=cvxopt.glpk.options, glpk={'msg_lev': 'GLP_MSG_ON'}, verbose=True)

    if knapsack_problem.status != "optimal":
        print("status:", knapsack_problem.status)
        # print (selection.value)
        # assert (knapsack_problem.status == "optimal")

    ##########################################

    select = np.array(selection.value, dtype=np.int32)
    select = np.reshape(select, (xb, wb, step, rpr))

    # check 1
    ones = np.sum(select, axis=(2, 3))
    assert (np.all(ones == 1))
    # check 2
    ones = np.sum(select)
    assert (ones == 64)

    ##########################################
    
    scale = np.arange(1, rpr + 1, dtype=np.int32)
    rpr_lut = np.sum(select * scale, axis=(2, 3))
    assert (np.all(rpr_lut > 0))

    ##########################################

    scale = np.arange(1, step + 1, dtype=np.int32).reshape(-1, 1)
    step_lut = np.sum(select * scale, axis=(2, 3))
    assert (np.all(step_lut > 0))

    print (step_lut)
    print (rpr_lut)

    ##########################################
    
    return rpr_lut
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
