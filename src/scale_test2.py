
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def mse(x):
    return np.sqrt(np.mean(x ** 2))

#####################################################

x1 = np.random.normal(loc=0., scale=1., size=100000)
x2 = np.random.normal(loc=0., scale=1., size=100000)
print (mse(x1 - x2))

#####################################################

N = 10
R = 10
mses = np.zeros(shape=(N, R))
colors = ['white', 'blue', 'red', 'black', 'green']

#####################################################

for i in range(N):
    for j in range(R):
        mu1 = -0.10 * i
        mu2 = 0.
        x1 = 0
        x2 = 0
        for _ in range(j):
            x1 += np.random.normal(loc=mu1, scale=1., size=100000)
            x2 += np.random.normal(loc=mu2, scale=1., size=100000)
        y = x1 - x2
        mses[i][j] = mse(y)

x = np.arange(N)
for j in range(1, R):
    plt.plot(x, mses[:, j], marker='.')

#####################################################

p = np.polyfit( np.arange(N), mses[:, 1], deg=2 )
print (p)
p = np.polyfit( np.arange(N), mses[:, 2], deg=2 )
print (p)
p = np.polyfit( np.arange(N), mses[:, 3], deg=2 )
print (p)

p = np.polyfit( np.arange(N), mses[:, 4], deg=2 )
print (p)
p = np.polyfit( np.arange(N), mses[:, 5], deg=2 )
print (p)
p = np.polyfit( np.arange(N), mses[:, 6], deg=2 )
print (p)

#####################################################

for i in range(N):
    for j in range(R):        
        # mses[i][j] = np.sqrt(2 * j) + (0.1 * i * j) ** 2 
        mses[i][j] = np.sqrt(2 * j) + (0.1 / 2 * i * j) ** 2 

x = np.arange(N)
for j in range(1, R):
    plt.plot(x, mses[:, j], marker='D')

#####################################################

plt.ylim(bottom=0)
plt.show()




