
import numpy as np
import matplotlib.pyplot as plt

def mse(x):
    return np.sqrt(np.mean(x ** 2))

def mae(x):
    return np.mean(np.abs(x))

x1 = np.random.normal(loc=0, scale=1, size=10000)
x2 = np.random.normal(loc=0, scale=1, size=10000)
print (mse(x1 + x2))
print (mae(x1 + x2))

N = 10
means = 0.1 * np.arange(N)
mses = np.zeros(N)
scales = np.zeros(N)
for i in range(N):
    y1 = x1 - means[i]
    y2 = x2 - means[i]
    mses[i] = mse(y1 + y2) - means[i] ** 2
    # scales[i] = 2 ** ((1 + means[i]) / (2 + means[i]))
    scales[i] = 2 ** ((1 + means[i]) / 2)

plt.plot(means, mses)
plt.plot(means, scales)
plt.show()
