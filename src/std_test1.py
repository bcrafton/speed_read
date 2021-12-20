
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N = 10000
'''
P = [0.1, 0.4, 0.2, 0.3]
R = [1, 2, 3, 4]
'''
N_R = 10
P = np.random.uniform(low=0, high=1, size=N_R)
P = P / np.sum(P)
R = np.arange(1, N_R + 1, 1)
# print (P)
# print (R)

xs = []
for p, r in zip(P, R):
    x = np.random.normal(loc=0., scale=np.sqrt(r), size=int(N * p))
    xs.extend(x.tolist())

std = np.sqrt(np.sum( np.array(R) * np.array(P) ))
print (np.std(xs), std)

k2, p = stats.normaltest(xs)
print (p)
# print (k2)
# plt.hist(xs, bins=50)
# plt.show()

'''
xs = np.random.normal(loc=0, scale=1, size=1000000)
k2, p = stats.normaltest(xs)
print (k2)
'''
