
import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import cvxopt
cvxopt.glpk.options["maxiters"] = 1
cvxopt.glpk.options['show_progress'] = True
cvxopt.glpk.options['tm_lim'] = 30000
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_ON'

cvxopt.solvers.options['maxiters'] = 1
cvxopt.solvers.options['show_progress'] = True
cvxopt.solvers.options['tm_lim'] = 30000
cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_ON'

# /home/brian/env/py3/lib/python3.5/site-packages/cvxpy/reductions/solvers/conic_solvers/glpk_mi_conif.py
# search in here to play with params.

# https://github.com/cvxgrp/cvxpy/pull/254
# https://github.com/cvxgrp/cvxpy/pull/254/commits/c4207f2e2616ac2a3e2e6c906c0660d36dcc6814

# https://github.com/cvxgrp/cvxpy/issues/251
# https://github.com/cvxgrp/cvxpy/issues/351

##########################################

def optimize_rpr(error, mean, delay, valid, threshold):
    xb, wb, step, rpr, sar = np.shape(error)
    assert (np.shape(error) == np.shape(delay))

    weight1 = np.reshape(error, -1)
    weight2 = np.reshape(mean, -1)
    value = np.reshape(delay, -1)
    valid = np.reshape(valid, -1)

    ##########################################

    # https://www.cvxpy.org/tutorial/advanced/index.html#attributes
    # [shape, boolean, integer, nonneg, nonpos]
    # ValueError: Cannot set more than one special attribute in Variable.
    selection = cvxpy.Variable((xb * wb * step * rpr * sar), integer=True)
    select_constraint = []

    for i in range(xb * wb):
        start = (i + 0) * (step * rpr * sar)
        end   = (i + 1) * (step * rpr * sar)
        select_constraint.append( cvxpy.sum(1.0 * (selection[start:end] >= 1)) == 1 )

    for i in range(xb * wb * step * rpr * sar):
        select_constraint.append( selection[i] >= 0 )
        select_constraint.append( selection[i] <= 8 )

    ##########################################

    weight_constraint1 = weight1 @ 1.0 * (selection >= 1) <= threshold
    weight_constraint2 = weight2 @ 1.0 * (selection >= 1) <= threshold
    weight_constraint3 = weight2 @ 1.0 * (selection >= 1) >= -threshold

    valid_constraint = (valid @ selection) == 64

    ##########################################

    total_value = value @ 1.0 * (selection >= 1)

    ##########################################

    knapsack_problem = cvxpy.Problem(cvxpy.Minimize(total_value), select_constraint + [weight_constraint1, weight_constraint2, weight_constraint3, valid_constraint])

    ##########################################

    knapsack_problem.solve(solver='GLPK_MI', options=cvxopt.glpk.options, glpk={'msg_lev': 'GLP_MSG_ON'}, verbose=True)
    
    if knapsack_problem.status != "optimal":
        print("status:", knapsack_problem.status)

    ##########################################

    print (selection.value)
    select = np.array(selection.value, dtype=np.int32)
    select = np.reshape(select, (xb, wb, step, rpr, sar))

    # check 1
    ones = np.sum(select, axis=(2, 3, 4))
    assert (np.all(ones == 1))

    # check 2
    ones = np.sum(select)
    assert (ones == 64)

    ##########################################

    scale = np.arange(step, dtype=np.int32).reshape(-1, 1, 1)
    step_lut = np.sum(select * scale, axis=(2, 3, 4))

    ##########################################

    scale = np.arange(1, rpr + 1, dtype=np.int32).reshape(1, -1, 1)
    rpr_lut = np.sum(select * scale, axis=(2, 3, 4))
    assert (np.all(rpr_lut > 0))

    ##########################################

    sar_lut = np.sum(select[:, :, :, :, 1:2], axis=(2, 3, 4))

    ##########################################

    return rpr_lut, step_lut, sar_lut
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    