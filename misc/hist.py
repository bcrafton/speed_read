
import numpy as np
import matplotlib.pyplot as plt

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()

plt.cla()
x = np.reshape(weights[0][0], -1)
plt.hist(x, bins=100)
plt.show()

plt.cla()
x = np.reshape(weights[1][0], -1)
plt.hist(x, bins=100)
plt.show()

plt.cla()
x = np.reshape(weights[2][0], -1)
plt.hist(x, bins=100)
plt.show()
