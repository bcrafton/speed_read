
import os
os.environ["OMP_NUM_THREADS"] = "8"

import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import mosek
import cvxopt
cvxopt.glpk.options["maxiters"] = 1
cvxopt.glpk.options['show_progress'] = True
# cvxopt.glpk.options['tm_lim'] = 120000 # 30000
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_ON'
cvxopt.glpk.options['abstol'] = 1e-2
cvxopt.glpk.options['reltol'] = 1e-2

cvxopt.solvers.options['maxiters'] = 1
cvxopt.solvers.options['show_progress'] = True
# cvxopt.solvers.options['tm_lim'] = 120000 # 30000
cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_ON'
cvxopt.solvers.options['abstol'] = 1e-2
cvxopt.solvers.options['reltol'] = 1e-2

# /home/brian/env/py3/lib/python3.5/site-packages/cvxpy/reductions/solvers/conic_solvers/glpk_mi_conif.py
# search in here to play with params.

# https://github.com/cvxgrp/cvxpy/pull/254
# https://github.com/cvxgrp/cvxpy/pull/254/commits/c4207f2e2616ac2a3e2e6c906c0660d36dcc6814

# https://github.com/cvxgrp/cvxpy/issues/251
# https://github.com/cvxgrp/cvxpy/issues/351

##########################################

def optimize_rpr(error, mean, delay, valid, area_adc, area_sar, area, threshold, Ns):
    (best_lut, best_N, best_delay) = (None, 0, 1e12)
    for N in Ns:
        if N > area: continue
        lut, current_delay = optimize_rpr_kernel(error, mean, delay, valid, area_adc, area_sar, area, threshold, N)
        if current_delay < best_delay: (best_lut, best_N, best_delay) = (lut, N, current_delay)

    assert (best_N > 0)
    rpr_lut, adc_lut, sar_lut = best_lut
    return rpr_lut, adc_lut, sar_lut, best_N

##########################################

def optimize_rpr_kernel(error, mean, delay, valid, area_adc, area_sar, area, threshold, N):
    
    # TODO:
    # make dim all caps
    xb, wb, rpr, adc, sar = np.shape(error)
    
    # print (xb, wb, rpr, adc, sar)
    assert (np.shape(error) == np.shape(mean))
    assert (np.shape(error) == np.shape(delay))
    assert (np.shape(error) == np.shape(valid))

    assert (len(area_adc) == adc)
    assert (len(area_sar) == sar)

    ##########################################

    error = error.reshape(xb, wb, rpr, adc, sar)
    mean  = mean.reshape( xb, wb, rpr, adc, sar)
    delay = delay.reshape(xb, wb, rpr, adc, sar)
    valid = valid.reshape(xb, wb, rpr, adc, sar)

    # N = 4
    assert (N in [1, 2, 4, 8])
    # AREA = 512

    delay = delay / N
    area_adc = area_adc * N
    area_sar = area_sar * N

    error = np.reshape(error, -1)
    mean  = np.reshape(mean,  -1)
    delay = np.reshape(delay, -1)
    valid = np.reshape(valid, -1)

    ##########################################

    # https://www.cvxpy.org/tutorial/advanced/index.html#attributes
    # [shape, boolean, integer, nonneg, nonpos]
    # ValueError: Cannot set more than one special attribute in Variable.
    selection = cvxpy.Variable((xb * wb * rpr * adc * sar), boolean=True)
    select_constraint = []

    selection = cvxpy.reshape(expr=selection, shape=(xb*wb, rpr*adc*sar), order='C')
    for i in range(xb * wb):
        select_constraint.append( cvxpy.sum(selection[i]) == 1 )
    selection = cvxpy.reshape(expr=selection, shape=(xb*wb*rpr*adc*sar), order='C')

    ##########################################

    adc_var = cvxpy.Variable(shape=adc, boolean=True)
    adc_constraint = []

    selection_n = cvxpy.reshape(expr=selection, shape=(xb*wb*rpr*adc*sar), order='C')

    selection_n = cvxpy.reshape(expr=selection_n, shape=(xb*wb, rpr*adc*sar), order='C')
    selection_n = cvxpy.sum(selection_n, axis=0)
    selection_n = cvxpy.reshape(expr=selection_n, shape=(       rpr*adc,sar), order='C')
    selection_n = cvxpy.sum(selection_n, axis=1)
    selection_n = cvxpy.reshape(expr=selection_n, shape=(       rpr,adc    ), order='C')
    selection_n = cvxpy.transpose(expr=selection_n)

    selection_n = cvxpy.reshape(expr=selection_n, shape=(adc, rpr), order='C')
    selection_n = cvxpy.sum(selection_n, axis=1)
    selection_n = cvxpy.reshape(expr=selection_n, shape=(adc,    ), order='C')

    constraint = cvxpy.sum(adc_var) == 1
    adc_constraint.append(constraint)
    for i in range(adc):
        constraint = 64 * cvxpy.sum(adc_var[i:]) >= cvxpy.sum(selection_n[i])
        adc_constraint.append(constraint)

    ##########################################

    sar_var = cvxpy.Variable(shape=sar, boolean=True)
    sar_constraint = []

    selection_n = cvxpy.reshape(expr=selection, shape=(xb*wb*rpr*adc*sar), order='C')

    selection_n = cvxpy.reshape(expr=selection_n, shape=(xb*wb, rpr*adc*sar), order='C')
    selection_n = cvxpy.sum(selection_n, axis=0)
    selection_n = cvxpy.reshape(expr=selection_n, shape=(       rpr*adc,sar), order='C')
    selection_n = cvxpy.transpose(expr=selection_n)

    selection_n = cvxpy.reshape(expr=selection_n, shape=(sar, rpr*adc), order='C')
    selection_n = cvxpy.sum(selection_n, axis=1)
    selection_n = cvxpy.reshape(expr=selection_n, shape=(sar,        ), order='C')

    constraint = cvxpy.sum(sar_var) == 1
    sar_constraint.append(constraint)
    for i in range(sar):
        constraint = 64 * cvxpy.sum(sar_var[i:]) >= cvxpy.sum(selection_n[i])
        sar_constraint.append(constraint)

    ##########################################

    rpr_constraint = []

    selection_n = cvxpy.reshape(expr=selection, shape=(xb*wb*rpr*adc*sar), order='C')

    selection_n = cvxpy.reshape(expr=selection_n, shape=(xb*wb*rpr, adc*sar), order='C')
    selection_n = cvxpy.sum(selection_n, axis=1)
    selection_n = cvxpy.reshape(expr=selection_n, shape=(xb*wb, rpr), order='C')
    selection_n = cvxpy.transpose(expr=selection_n)
    selection_n = cvxpy.reshape(expr=selection_n, shape=(rpr*xb*wb), order='C')
    selection_n = cvxpy.reshape(expr=selection_n, shape=(rpr*xb, wb), order='C')

    for i in range(rpr * xb):
        for j in range(0, wb, N):
            for k in range(N):
                constraint = selection_n[i][j] == selection_n[i][j + k]
                rpr_constraint.append(constraint)

    ##########################################

    error_constraint = (error @ selection) <= threshold
    mean_constraint1 = ( mean @ selection) <= threshold
    mean_constraint2 = ( mean @ selection) >= -threshold
    valid_constraint = (valid @ selection) == 64

    ##########################################

    area1 = area_adc @ adc_var
    area2 = area_sar @ sar_var
    area_constraint = (area1 + area2) <= area
    
    ##########################################

    total_delay = (delay @ selection)

    ##########################################

    knapsack_problem = cvxpy.Problem(cvxpy.Minimize(total_delay), select_constraint + adc_constraint + sar_constraint + rpr_constraint + [area_constraint, error_constraint, mean_constraint1, mean_constraint2, valid_constraint])

    ##########################################

    # https://github.com/cvxpy/cvxpy/issues/314
    # https://www.cvxpy.org/tutorial/advanced/index.html ... ctrl+f "mosek_params"
    # https://docs.mosek.com/latest/pythonapi/parameters.html#doc-all-parameter-list
    # For a linear problem, if bfs=True, then the basic solution will be retrieved instead of the interior-point solution. 
    # This assumes no specific MOSEK parameters were used which prevent computing the basic solution.

    # mosek.dparam.mio_max_time
    # mosek.dparam.optimizer_max_time
    # mosek.iparam.num_threads
    
    # iparam.mio_heuristic_level

    # iparam.mio_max_num_branches
    # iparam.mio_max_num_relaxs

    # iparam.mio_max_num_solutions
    # The mixed-integer optimizer can be terminated after a certain number of different feasible solutions has been located. If this parameter has the value , then the mixed-integer optimizer will be terminated when feasible solutions have been located.

    # iparam.mio_mode
    # Controls whether the optimizer includes the integer restrictions when solving a (mixed) integer optimization problem.
    
    # iparam.mio_node_optimizer
    # Controls which optimizer is employed at the non-root nodes in the mixed-integer optimizer.
    # free, intpnt, conic, primal, simplex, dual simplex, free simplex, mixed_int ...
    # > mixed int is invalid ? 

    # iparam.mio_node_selection
    # Controls the node selection strategy employed by the mixed-integer optimizer.
    # free, first, best, pseudo 

    # dparam.intpnt_tol_infeas
    # dparam.intpnt_co_tol_infeas
    # dparam.intpnt_qo_tol_infeas
    # Infeasibility tolerance used by the interior-point optimizer for quadratic problems. Controls when the interior-point optimizer declares the model primal or dual infeasible. A small number means the optimizer gets more conservative about declaring the model infeasible.

    # is our problem {linear, conic, quadratic} ?

    # dparam.intpnt_tol_rel_step
    # Relative step size to the boundary for linear and quadratic optimization problems.
    
    # dparam.intpnt_tol_step_size
    # Minimal step size tolerance. If the step size falls below the value of this parameter, then the interior-point optimizer assumes that it is stalled. In other words the interior-point optimizer does not make any progress and therefore it is better to stop.

    # mio_rel_gap_const
    # mio_tol_abs_gap
    # mio_tol_abs_relax_int

    # Generally the difference between a best known solution, e.g. the incumbent solution in mixed integer programming, and a value that bounds the best possible solution. 
    # One such measure is the duality gap. 
    # The term is often qualified as the absolute gap, which is the magnitude of the difference between the best known solution and the best bound, 
    # or as the relative gap, which is the absolute gap divided by the best bound. 

    # iparam.mio_cut_knapsack_cover
    # iparam.mio_cut_selection_level
    # Controls how aggressively generated cuts are selected to be included in the relaxation.

    ##########################################
    # '''
    print ('Starting Solve')
    mosek_params = {mosek.dparam.mio_tol_rel_gap: 1e-2, 
                    mosek.dparam.intpnt_co_tol_rel_gap: 1e-2, 
                    mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                    mosek.dparam.mio_max_time: 5000,
                    mosek.dparam.optimizer_max_time: 5000,
                    # mosek.iparam.mio_heuristic_level: 10000000,
                    # mosek.iparam.mio_max_num_branches: 32,
                    # mosek.iparam.mio_max_num_relaxs: 32,
                    # 
                    # mosek.iparam.mio_node_optimizer: mosek.optimizertype.dual_simplex,
                    # mosek.iparam.mio_node_optimizer: mosek.optimizertype.intpnt,
                    # 
                    # mosek.iparam.mio_mode: mosek.miomode.satisfied,
                    # mosek.iparam.mio_node_selection: mosek.mionodeseltype.best,
                    # 
                    # mosek.dparam.intpnt_tol_infeas: 1.,
                    # 
                    # mosek.dparam.intpnt_tol_step_size: 1.0,
                    # mosek.dparam.intpnt_tol_rel_step: 0.999999,
                    mosek.iparam.intpnt_starting_point : mosek.startpointtype.satisfy_bounds,
                    mosek.iparam.presolve_use: mosek.presolvemode.on,
    }
    knapsack_problem.solve(solver='MOSEK', verbose=True, mosek_params=mosek_params)
    if knapsack_problem.status != "optimal":
        print("status:", knapsack_problem.status)
    # '''
    ##########################################
    '''
    knapsack_problem.solve(solver='GLPK_MI', options=cvxopt.glpk.options, glpk={'msg_lev': 'GLP_MSG_ON'}, verbose=True)
    if knapsack_problem.status != "optimal":
        print("status:", knapsack_problem.status)
    '''
    ##########################################

    print (np.around(adc_var.value))
    print (np.around(sar_var.value))

    # wanted to check constraints were set.
    # got some weird values like: 
    # [1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    # [ 1.  0. -0.  1.  1.  0.  0.  1.]

    # but when we did:
    # total_delay = (delay @ selection) + cvxpy.sum(sar_var) + cvxpy.sum(adc_var)
    # it fixed the weird issue
    # could have also put in an additional constraint

    # put in the other constraints ... 
    # sar_constraint.append( sar_var[i * sar + j] <= sar_var[i * sar + j - 1] )
    # works just fine.

    select = np.array(selection.value, dtype=float)
    # have gotten 65 "ones" ... one is 5.77315973e-15
    # have gotten 63 "ones" ... assume its 1 - 5.77315973e-15
    # print ( select[np.where(select > 0)] )
    # select = 1 * (select > 0) * (select > 1e-3)
    select = np.around(select).astype(int)
    select = np.reshape(select, (xb, wb, rpr, adc, sar))

    # check 1
    ones1 = np.sum(select)
    # check 
    ones2 = np.sum(select, axis=(2, 3, 4))

    # print (ones1)
    # print (ones2)

    assert (ones1 == 64)
    assert (np.all(ones2 == 1))

    ##########################################

    scale = np.arange(rpr, dtype=np.int32).reshape(-1, 1, 1)
    rpr_lut = np.sum(select * scale, axis=(2, 3, 4))

    ##########################################

    scale = np.arange(adc, dtype=np.int32).reshape(1, -1, 1)
    adc_lut = np.sum(select * scale, axis=(2, 3, 4))

    ##########################################

    scale = np.arange(sar, dtype=np.int32).reshape(1, 1, -1)
    sar_lut = np.sum(select * scale, axis=(2, 3, 4))

    ##########################################

    print ('Total Error:', select.flatten() @ error)
    
    print ('Total Mean:', select.flatten() @ mean)

    total_delay = select.flatten() @ delay
    print ('Total Delay:', select.flatten() @ delay)

    ##########################################
    
    area1 = np.around(adc_var.value) @ area_adc
    area2 = np.around(sar_var.value) @ area_sar
    print ('Total Area:', (area1 + area2)) 

    ##########################################

    return (rpr_lut, adc_lut, sar_lut), total_delay
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
