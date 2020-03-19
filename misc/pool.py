

import numpy as np
import matplotlib.pyplot as plt

x = np.load('imagenet.npy', allow_pickle=True).item()
x = x['data'][0]
x = np.reshape(x, (3, 64, 64))
x = np.transpose(x, (1, 2, 0))
x = x / 255.

#plt.imshow(x)
#plt.show()

x = np.reshape(x, (32, 2, 32, 2, 3))
x = np.transpose(x, (0, 2, 1, 3, 4))
x = np.mean(x, axis=(2, 3))
plt.imshow(x)
plt.show()

