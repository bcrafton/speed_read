
import numpy as np
import matplotlib.pyplot as plt

####################

def merge_dicts(list_of_dicts):
    results = {}
    for d in list_of_dicts:
        for key in d.keys():
            if key in results.keys():
                results[key].append(d[key])
            else:
                results[key] = [d[key]]

    return results

####################

num_layers = 7
results = np.load('results.npy', allow_pickle=True).item()
results_tf = np.load('results_tf.npy', allow_pickle=True).item()

x = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])

y_perf = np.zeros(shape=(2, 2, len(x), num_layers))
y_mean = np.zeros(shape=(2, 2, len(x), num_layers))
y_std = np.zeros(shape=(2, 2, len(x), num_layers))
acc = results_tf['acc_tf']

for key in sorted(results.keys()):
    (skip, cards, sigma) = key
    layer_results = results[key]

    for layer in range(num_layers):
        example_results = merge_dicts(layer_results[layer])
        sigma_index = np.where(x == sigma)[0][0]
        y_perf[skip][cards][sigma_index][layer] = np.sum(example_results['nmac']) / np.sum(example_results['cycle'])
        y_mean[skip][cards][sigma_index][layer] = np.mean(example_results['mean'])
        y_std[skip][cards][sigma_index][layer] = np.mean(example_results['std'])

####################
'''
print (np.around(y_perf[0, 0], 1))
print (np.around(y_perf[1, 0], 1))
print (np.around(y_perf[1, 1], 1))
'''
# print (np.around(y_mean[1, 1],  3))

print (np.around(y_std[0, 0],  3))
print (np.around(y_std[1, 0],  3))
print (np.around(y_std[1, 1],  3))

####################

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, y_perf[0, 0, :, 4], color='red', linestyle='--', label='baseline')
ax1.plot(x, y_perf[1, 0, :, 4], color='blue', linestyle='--', label='skip')
ax1.plot(x, y_perf[1, 1, :, 4], color='green', linestyle='--', label='cards')
ax1.set_ylim(bottom=0)

####################

y2 = 'ACC'

if y2 == 'STD':
  ax2.plot(x, y_std[0, 0, :, 4], color='red', label='baseline')
  ax2.plot(x, y_std[1, 0, :, 4], color='blue', label='skip')
  ax2.plot(x, y_std[1, 1, :, 4], color='green', label='cards')
  ax2.set_ylim(bottom=0)
  ax2.set_ylabel("Average STD from Truth")
elif y2 == 'ACC':
  ax2.plot(x, acc[0, 0, :], color='red', label='baseline')
  ax2.plot(x, acc[1, 0, :], color='blue', label='skip')
  ax2.plot(x, acc[1, 1, :], color='green', label='cards')
  ax2.set_ylabel("Classification Accuracy")
else:
  assert (False)

####################

ax1.legend(loc='center left')
ax2.legend(loc='center right')

ax1.set_ylabel("MAC / Cycle")
plt.xticks(x)

ax1.set_xlabel('Cell to Cell Variance')

fig = plt.gcf()
fig.set_size_inches(6, 4)
fig.savefig('cards.png', dpi=300)
# plt.show()

####################


