
import numpy as np
import matplotlib.pyplot as plt

weights = np.load('imagenet_weights.npy').item()

print (weights[1][0].flatten()[0:50])

plt.hist(weights[1][0].flatten(), bins = 128)
plt.show()
