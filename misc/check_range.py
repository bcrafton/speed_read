
import numpy as np
import matplotlib.pyplot as plt

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()
x = weights[2][0]

print (np.shape(x))
x = np.reshape(x, (3 * 3 * 64, 128))

print (np.max(x, axis=0) - np.min(x, axis=0))
