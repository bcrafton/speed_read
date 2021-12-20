
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng()
pts = 1000

'''
a = rng.normal(0, 1, size=pts)
b = rng.normal(10, 1, size=pts)
x = np.concatenate((a, b))
'''

x = np.random.uniform(low=-1, high=1, size=1000)

plt.hist(x, bins=500)
plt.show()

k2, p = stats.normaltest(x)
alpha = 1e-3

print("p = {:g}".format(p))
if p < alpha:
    # null hypothesis: x comes from a normal distribution
    # so this branch means its NOT normal
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
