
import numpy as np
import matplotlib.pyplot as plt

results = np.load('results.npy', allow_pickle=True).item()

x = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
y_psum = np.zeros(shape=(2, len(x), 6))
y_std = np.zeros(shape=(2, len(x), 6))

for key in sorted(results.keys()):
    (cards, sigma) = key
    layer_results = results[key]

    for layer in layer_results.keys():
        [psum, mean, std] = np.mean(layer_results[layer], axis=0)
        sigma_index = np.where(x == sigma)[0][0]
        y_psum[cards][sigma_index][layer] = psum
        y_std[cards][sigma_index][layer] = std

####################

# print (y_psum[0])
# print (y_std[0])

# print (y_psum[1])
print (y_std[1])

####################

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, y_psum[0, :, 3], color='red', linestyle='--', label='skip performance')
ax1.plot(x, y_psum[1, :, 3], color='green', linestyle='--', label='cards performance')

ax2.plot(x, y_std[0, :, 3], color='red', label='skip std')
ax2.plot(x, y_std[1, :, 3], color='green', label='cards std')

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()

####################


