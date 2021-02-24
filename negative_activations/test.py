
import numpy as np

#############################################################

N = 10
x = np.random.randint(low=-8, high=7+1, size=N)
w = np.random.randint(low=-8, high=7+1, size=(N, N))
truth = x @ w

#############################################################

x = x + 8
w = w + 8

y = x @ w

bias = N * 8 * 8
bias0 = 8 * np.sum(x)
bias1 = 8 * np.sum(w, axis=0)

# which axis to sum along ? 
# print (w[:, 0] @ x)
# print (y[0])

y = y + bias - bias0 - bias1
print (y)
print (truth)

#############################################################

