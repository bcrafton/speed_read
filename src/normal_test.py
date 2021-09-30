
import numpy as np
import matplotlib.pyplot as plt

####################################

x = np.random.normal(loc=0, scale=100, size=100000).astype(int)
# plt.hist(x, bins=200)
# plt.show()

####################################

mae = np.mean(np.abs(x - np.mean(x)))
print (np.mean(x))
print (np.std(x))
print (mae)
print ()

####################################

val, count = np.unique(x, return_counts=True)
mean = np.sum(val * count) / np.sum(count)
std = np.sqrt(np.sum(count / np.sum(count) * (val - mean) ** 2))
mae = np.sum(count / np.sum(count) * np.abs(val - mean))

print (mean)
print (std)
print (mae)
print ()

####################################

val, count = np.unique(x, return_counts=True)
pmf = count / np.sum(count)
mean = np.sum(val * pmf)
std = np.sqrt(np.sum(pmf * (val - mean) ** 2))
mae = np.sum(pmf * np.abs(val - mean))

print (mean)
print (std)
print (mae)
print ()

####################################

