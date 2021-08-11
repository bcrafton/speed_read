
import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import cvxopt
cvxopt.glpk.options["maxiters"] = 1
cvxopt.glpk.options['show_progress'] = True
cvxopt.glpk.options['tm_lim'] = 60000 # 30000
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_ON'

cvxopt.solvers.options['maxiters'] = 1
cvxopt.solvers.options['show_progress'] = True
cvxopt.solvers.options['tm_lim'] = 60000 # 30000
cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_ON'

# /home/brian/env/py3/lib/python3.5/site-packages/cvxpy/reductions/solvers/conic_solvers/glpk_mi_conif.py
# search in here to play with params.

# https://github.com/cvxgrp/cvxpy/pull/254
# https://github.com/cvxgrp/cvxpy/pull/254/commits/c4207f2e2616ac2a3e2e6c906c0660d36dcc6814

# https://github.com/cvxgrp/cvxpy/issues/251
# https://github.com/cvxgrp/cvxpy/issues/351

##########################################

def optimize_rpr(error, mean, delay, area, valid, threshold):
    xb, wb, step, rpr, adc, sar = np.shape(error)
    # print (xb, wb, step, rpr, adc, sar)
    assert (np.shape(error) == np.shape(delay))

    error = error.reshape(xb, wb, step, rpr, adc, sar, 1)
    mean  = mean.reshape( xb, wb, step, rpr, adc, sar, 1)
    delay = delay.reshape(xb, wb, step, rpr, adc, sar, 1)
    valid = valid.reshape(xb, wb, step, rpr, adc, sar, 1)
    area  = area.reshape( xb, wb, step, rpr, adc, sar, 1)

    error = np.tile(error, (1, 1, 1, 1, 1, 1, 8))
    mean  = np.tile(mean,  (1, 1, 1, 1, 1, 1, 8))
    delay = np.tile(delay, (1, 1, 1, 1, 1, 1, 8))
    valid = np.tile(valid, (1, 1, 1, 1, 1, 1, 8))
    area  = np.tile(area,  (1, 1, 1, 1, 1, 1, 8))

    scale = np.arange(1, 8 + 1, dtype=np.int32)
    area  = area * scale
    delay = delay / scale

    error = np.reshape(error, -1)
    mean  = np.reshape(mean, -1)
    delay = np.reshape(delay, -1)
    valid = np.reshape(valid, -1)
    area  = np.reshape(area, -1)

    ##########################################

    # https://www.cvxpy.org/tutorial/advanced/index.html#attributes
    # [shape, boolean, integer, nonneg, nonpos]
    # ValueError: Cannot set more than one special attribute in Variable.
    selection = cvxpy.Variable((xb * wb * step * rpr * adc * sar * 8), boolean=True)
    select_constraint = []

    for i in range(xb * wb):
        start = (i + 0) * (step * rpr * adc * sar * 8)
        end   = (i + 1) * (step * rpr * adc * sar * 8)
        select_constraint.append( cvxpy.sum(selection[start:end]) == 1 )

    ##########################################

    error_constraint = error @ selection <= threshold
    mean_constraint1 = mean @ selection <= threshold
    mean_constraint2 = mean @ selection >= -threshold

    valid_constraint = (valid @ selection) == 64

    area_constraint = area @ selection <= 256

    ##########################################

    total_delay = delay @ selection

    ##########################################

    knapsack_problem = cvxpy.Problem(cvxpy.Minimize(total_delay), select_constraint + [area_constraint, error_constraint, mean_constraint1, mean_constraint2, valid_constraint])

    ##########################################

    knapsack_problem.solve(solver='GLPK_MI', options=cvxopt.glpk.options, glpk={'msg_lev': 'GLP_MSG_ON'}, verbose=True)

    if knapsack_problem.status != "optimal":
        print("status:", knapsack_problem.status)

    ##########################################

    select = np.array(selection.value, dtype=np.int32)
    select = np.reshape(select, (xb, wb, step, rpr, adc, sar, 8))

    # check 1
    ones = np.sum(select, axis=(2, 3, 4, 5, 6))
    assert (np.all(ones == 1))

    # check 2
    ones = np.sum(select)
    assert (ones == 64)

    ##########################################

    scale = np.arange(step, dtype=np.int32).reshape(-1, 1, 1, 1, 1)
    step_lut = np.sum(select * scale, axis=(2, 3, 4, 5, 6))

    ##########################################

    scale = np.arange(1, rpr + 1, dtype=np.int32).reshape(1, -1, 1, 1, 1)
    rpr_lut = np.sum(select * scale, axis=(2, 3, 4, 5, 6))
    assert (np.all(rpr_lut > 0))

    ##########################################

    scale = np.arange(1, adc + 1, dtype=np.int32).reshape(1, 1, -1, 1, 1)
    adc_lut = np.sum(select * scale, axis=(2, 3, 4, 5, 6))

    ##########################################

    sar_lut = np.sum(select[:, :, :, :, :, 1:2, :], axis=(2, 3, 4, 5, 6))

    ##########################################

    scale = np.arange(1, 8 + 1, dtype=np.int32).reshape(1, 1, 1, 1, -1)
    num_lut = np.sum(select * scale, axis=(2, 3, 4, 5, 6))

    ##########################################

    print (step_lut)
    print (rpr_lut)
    print (adc_lut)
    print (num_lut)
    print (sar_lut)

    # print (select.flatten() @ error)
    # print (select.flatten() @ mean)

    # print (error[ np.where(select.flatten() > 0) ])
    # print (mean[ np.where(select.flatten() > 0) ])

    ##########################################

    return rpr_lut, step_lut, sar_lut, adc_lut, num_lut
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
