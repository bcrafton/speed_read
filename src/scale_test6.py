
import numpy as np

def mse(x):
    return np.sqrt(np.mean(x ** 2))

MU = [0.1, 0.2, 0.3, 0.4]
N = [1, 2, 3, 4, 5]

MSE  = np.zeros(shape=( len(MU), len(N) ))
mean = np.zeros(shape=( len(MU), len(N) ))
std  = np.zeros(shape=( len(MU), len(N) ))

#############################

for i, mu in enumerate(MU):
    for j, n in enumerate(N):
        x = np.random.normal(loc=n*mu, scale=np.sqrt(n), size=100000)
        std[i][j]  = np.std(x)
        mean[i][j] = np.mean(x)
        MSE[i][j]  = mse(x)

print (mean)
print ()
print (std)
print ()
print (MSE)
print ()
print ('-------------------')

#############################

for i, mu in enumerate(MU):
    for j, n in enumerate(N):
        std[i][j] = np.sqrt(MSE[i][j] ** 2 - mean[i][j] ** 2)

print (std)
print ()

#############################
