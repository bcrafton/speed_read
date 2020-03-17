
import numpy as np
import matplotlib.pyplot as plt

results = np.load('results.npy', allow_pickle=True).item()

x = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
y_psum = np.zeros(shape=(2, 2, len(x), 6))
y_mean = np.zeros(shape=(2, 2, len(x), 6))
y_std = np.zeros(shape=(2, 2, len(x), 6))

for key in sorted(results.keys()):
    (skip, cards, sigma) = key
    layer_results = results[key]

    for layer in layer_results.keys():
        [psum, mean, std] = np.mean(layer_results[layer], axis=0)
        sigma_index = np.where(x == sigma)[0][0]
        y_psum[skip][cards][sigma_index][layer] = psum
        y_mean[skip][cards][sigma_index][layer] = mean
        y_std[skip][cards][sigma_index][layer] = std

####################

# print (y_psum[0, 0])
# print (y_psum[1, 0])
# print (y_psum[1, 1])

print ()

print (np.around(y_mean[0, 0],  3))
print ()
print (np.around(y_mean[1, 0],  3))
print ()
print (np.around(y_mean[1, 1],  3))

print ()

# print (y_std[0, 0])
# print (y_std[1, 0])
# print (y_std[1, 1])

####################

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, y_psum[0, 0, :, 3], color='red', linestyle='--', label='baseline')
ax1.plot(x, y_psum[1, 0, :, 3], color='blue', linestyle='--', label='skip')
ax1.plot(x, y_psum[1, 1, :, 3], color='green', linestyle='--', label='cards')

ax2.plot(x, y_std[0, 0, :, 3], color='red', label='baseline')
ax2.plot(x, y_std[1, 0, :, 3], color='blue', label='skip')
ax2.plot(x, y_std[1, 1, :, 3], color='green', label='cards')

ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

ax1.legend(loc='center left')
ax2.legend(loc='center right')

ax1.set_ylabel("MAC / Cycle")
ax2.set_ylabel("Average STD from Truth")
plt.xticks(x)

ax1.set_xlabel('Cell to Cell Variance')

fig = plt.gcf()
fig.set_size_inches(6, 4)
fig.savefig('cards.png', dpi=300)
# plt.show()

####################


