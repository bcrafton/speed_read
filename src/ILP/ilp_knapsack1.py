
import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

#########################################

table = np.load('table.npy', allow_pickle=True).item()

capacity = 1.

value = table['value'].reshape(-1)
assert (np.all(value > 0))
value = 1. / value

weight = table['weight'].reshape(-1)

#########################################

selection = cvxpy.Variable((64 * 64), boolean=True)

select_constraint = []
for i in range(0, 64 * 64, 64):
    select_constraint.append( cvxpy.sum(selection[i : i + 64]) == 1 )

#########################################

weight_constraint = weight @ selection <= 1.

#########################################

total_value = value @ selection

#########################################

# We tell cvxpy that we want to maximize total utility 
# subject to weight_constraint. All constraints in 
# cvxpy must be passed as a list
knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_value), select_constraint + [weight_constraint])

# Solving the problem
knapsack_problem.solve(solver=cvxpy.GLPK_MI)

print (selection.value)

#########################################
