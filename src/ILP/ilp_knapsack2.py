
import cvxpy
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

#########################################

table = np.load('table.npy', allow_pickle=True).item()

capacity = 1.

value = table['value'].reshape(-1)
assert (np.all(value > 0))
# value = 1. / value

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
knapsack_problem = cvxpy.Problem(cvxpy.Minimize(total_value), select_constraint + [weight_constraint])

# Solving the problem
knapsack_problem.solve(solver=cvxpy.GLPK_MI, verbose=True)

# print (selection.value)

#########################################

rpr = np.array(selection.value)

rpr_lut = np.reshape(rpr, (64, 64))
ones = np.sum(rpr_lut, axis=1)
# print (ones)

scale = np.arange(1, 64 + 1)
rpr_lut = np.sum(rpr_lut * scale, axis=1)

print (np.reshape(rpr_lut, (8, 8)))

#########################################

sum_error = 0
for i, select in enumerate(rpr):
    if select:
        sum_error += weight[i]
print (sum_error)

#########################################

sum_value = 0
for i, select in enumerate(rpr):
    if select:
        sum_value += value[i]
print (sum_value)

#########################################















