
import numpy as np

x = np.random.uniform(low=20, high=100, size=100)
y = np.random.uniform(low=0, high=100, size=100)
e = x - y

mse = np.sqrt(np.mean( e ** 2 ))
print (mse)

std = np.std(e)
print (std)

mean = np.mean(e)
print (mean)

mse = np.sqrt(std ** 2 + mean ** 2)
print (mse)
