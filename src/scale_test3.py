
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

E = 0.2
N = 10
R = 10
mses = np.zeros(shape=(N, R))
colors = ['white', 'blue', 'red', 'black', 'green']

x = np.arange(N)

#####################################################

for i in range(N):
    for j in range(R):
        mu1 = -E * i
        mu2 = 0.
        x1 = 0
        x2 = 0
        for _ in range(j):
            x1 += np.random.normal(loc=mu1, scale=1., size=100000)
            x2 += np.random.normal(loc=mu2, scale=1., size=100000)
        y = x1 - x2
        mses[i][j] = mse(y)


for j in range(1, R):
    plt.plot(x, mses[:, j], marker='.')

#####################################################

def f(x, a, b):
    return (a * x ** 2) + b

ys = []
for j in range(1, R):        
    popt, pcov = curve_fit(f=f, xdata=x, ydata=mses[:, j])
    plt.plot(x, f(x, *popt), marker='D')
    ys.append(popt[0])

plt.cla()
plt.plot(np.arange(R - 1), ys, marker='.')
plt.show()

print (ys)
for i in range(1, R - 1):
    print (ys[i] - ys[i - 1])

#####################################################
'''
for i in range(N):
    for j in range(R):        
        # mses[i][j] = np.sqrt(2 * j) + (0.1 * i * j) ** 2 
        mses[i][j] = np.sqrt(2 * j) + (0.1 / 2 * i * j) ** 2 

for j in range(1, R):
    plt.plot(x, mses[:, j], marker='^')
'''
#####################################################

plt.ylim(bottom=0)
plt.show()




