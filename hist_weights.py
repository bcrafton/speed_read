
import numpy as np
import matplotlib.pyplot as plt

weights = np.load('imagenet_weights.npy').item()

# fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].hist(weights[3][0].flatten(), bins=128)
ax[1].hist(weights[5][0].flatten(), bins=128)
plt.show()
