
import numpy as np
import matplotlib.pyplot as plt

def mse(x):
    return np.sqrt(np.mean(x ** 2))

N = 50
mses = np.zeros(N)
for i in range(N):
    mu = -0.10 * i
    x1 = np.random.normal(loc=mu, scale=1., size=100000)
    x2 = np.random.normal(loc=0,  scale=1., size=100000)
    y = x1 - x2
    mses[i] = mse(y)



plt.plot(np.arange(N), mses)
plt.ylim(bottom=0)
plt.show()
