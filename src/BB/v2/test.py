
import numpy as np
import copy

############################

# https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html

capacity = 750
weight = np.array([70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120], dtype=np.int)
value = np.array([135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240], dtype=np.int)
N = len(value)
truth = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]

############################

print (value)

ratio = value / weight
order = np.argsort(ratio)
order2 = np.argsort(order)
value = value[order]

x = np.array(range(15))
print (x)
x = x[order]
print (x)
# x[order] = x
# print (x)

print (order)

############################

print(x[order2])













