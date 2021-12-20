
import numpy as np
import matplotlib.pyplot as plt

N = 10000
'''
P = [0.1, 0.4, 0.2, 0.3]
R = [1, 2, 3, 4]
'''
P = np.random.uniform(low=0, high=1, size=10)
P = P / np.sum(P)
R = np.arange(1, 10 + 1, 1)

xs = []
for p, r in zip(P, R):
    x = np.random.normal(loc=0., scale=np.sqrt(r), size=int(N * p))
    xs.extend(x.tolist())
print (np.std(xs))

plt.hist(xs, bins=50)
plt.show()

std = np.sqrt(np.sum( np.array(R) * np.array(P) ))
print (std)
