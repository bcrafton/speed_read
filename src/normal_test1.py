
import numpy as np
import matplotlib.pyplot as plt

####################################

def MAE(x):
    return np.mean(np.abs(x - np.mean(x)))

N = 8
y = 0
for _ in range(N):
    y += np.random.normal(loc=0, scale=100, size=100000).astype(int)

x = np.arange(0, len(y))
mae = MAE(y)
mean = np.mean(y)
print (mean, mae / np.sqrt(N))

plt.scatter(x, y)
plt.hlines(y=mean, xmin=np.min(x), xmax=np.max(x), color='black')
plt.hlines(y=mae, xmin=np.min(x), xmax=np.max(x), color='black')
plt.show()

####################################

