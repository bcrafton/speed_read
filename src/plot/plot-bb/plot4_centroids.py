
import numpy as np
np.set_printoptions(precision=2, suppress=True) 

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

comp_pJ = 22. * 1e-12 / 32. / 16.

num_layers = 6
num_comparator = 8
results = np.load('results.npy', allow_pickle=True).item()

x = np.array([0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])

y_mean = np.zeros(shape=(2, 2, 2, len(x), num_layers))
y_std = np.zeros(shape=(2, 2, 2, len(x), num_layers))

y_mac_per_cycle = np.zeros(shape=(2, 2, 2, len(x), num_layers))
y_mac_per_pJ = np.zeros(shape=(2, 2, 2, len(x), num_layers))

y_mac = np.zeros(shape=(2, 2, 2, len(x), num_layers))
y_cycle = np.zeros(shape=(2, 2, 2, len(x), num_layers))

y_ron = np.zeros(shape=(2, 2, 2, len(x), num_layers))
y_roff = np.zeros(shape=(2, 2, 2, len(x), num_layers))
y_adc = np.zeros(shape=(2, 2, 2, len(x), num_layers, num_comparator))

y_energy = np.zeros(shape=(2, 2, 2, len(x), num_layers))

####################

for key in sorted(results.keys()):
    (skip, cards, alloc, profile, narray, sigma, rpr) = key
    layer_results = results[key]

    if rpr == 'dynamic':
        rpr = 0
    elif rpr == 'centroids':
        rpr = 1
    else:
        assert (False)

    for layer in range(num_layers):
        example_results = merge_dicts(layer_results[layer])
        sigma_index = np.where(x == sigma)[0][0]
        
        y_mean[skip][cards][rpr][sigma_index][layer] = np.mean(example_results['mean'])
        y_std[skip][cards][rpr][sigma_index][layer] = np.mean(example_results['std'])

        y_mac_per_cycle[skip][cards][rpr][sigma_index][layer]  = np.sum(example_results['nmac']) / np.sum(example_results['cycle'])
        y_mac[skip][cards][rpr][sigma_index][layer]   = np.mean(example_results['nmac']) 
        y_cycle[skip][cards][rpr][sigma_index][layer] = np.mean(example_results['cycle'])

        y_ron[skip][cards][rpr][sigma_index][layer] = np.sum(example_results['ron'])
        y_roff[skip][cards][rpr][sigma_index][layer] = np.sum(example_results['roff'])
        y_adc[skip][cards][rpr][sigma_index][layer] = np.sum(example_results['adc'], axis=0)
        
        y_energy[skip][cards][rpr][sigma_index][layer] += y_ron[skip][cards][rpr][sigma_index][layer] * 2e-16
        y_energy[skip][cards][rpr][sigma_index][layer] += y_roff[skip][cards][rpr][sigma_index][layer] * 2e-16
        y_energy[skip][cards][rpr][sigma_index][layer] += np.sum(y_adc[skip][cards][rpr][sigma_index][layer] * np.array([1,2,3,4,5,6,7,8]) * comp_pJ)

        y_mac_per_pJ[skip][cards][rpr][sigma_index][layer] = np.sum(example_results['nmac']) / 1e12 / np.sum(y_energy[skip][cards][rpr][sigma_index][layer])

####################

plot_layer = 0

####################

TOPs_skip      = 2 * 700e6 * np.sum(y_mac_per_cycle[1, 0, 0, :, :], axis=1) / 1e12 
TOPs_cards     = 2 * 700e6 * np.sum(y_mac_per_cycle[1, 1, 0, :, :], axis=1) / 1e12
TOPs_centroids = 2 * 700e6 * np.sum(y_mac_per_cycle[1, 1, 1, :, :], axis=1) / 1e12

####################

MAC_pJ_skip      = np.sum(y_mac[1, 0, 0, :, :], axis=1) / 1e12 / np.sum(y_energy[1, 0, 0, :, :], axis=1)
MAC_pJ_cards     = np.sum(y_mac[1, 1, 0, :, :], axis=1) / 1e12 / np.sum(y_energy[1, 1, 0, :, :], axis=1)
MAC_pJ_centroids = np.sum(y_mac[1, 1, 1, :, :], axis=1) / 1e12 / np.sum(y_energy[1, 1, 1, :, :], axis=1)

####################

plt.cla()
ax = plt.gca()
# plt.plot(x, y_mac_per_cycle[0, 0, :, plot_layer], color='green', marker="D", markersize=5, label='baseline')
plt.plot(x, TOPs_skip, color='green', marker="D", markersize=5, label='skip')
plt.plot(x, TOPs_cards, color='blue', marker="s", markersize=6, label='cards')
plt.plot(x, TOPs_centroids, color='black', marker="^", markersize=6, label='k-means')
plt.ylim(bottom=0)
plt.xticks(x)
# plt.xticks([0.08, 0.12])
# plt.yticks([])
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
plt.xlabel('Variance')
plt.ylabel('TOPs')
plt.grid(True, linestyle='dotted')
fig = plt.gcf()
# fig.set_size_inches(4., 2.5)
plt.tight_layout()
plt.legend()
fig.savefig('TOPs.png', dpi=300)

plt.cla()
ax = plt.gca()
# plt.plot(x, y_mac_per_pJ[0, 0, :, plot_layer], color='green', marker="D", markersize=5, label='baseline')
plt.plot(x, MAC_pJ_skip, color='green', marker="D", markersize=5, label='skip')
plt.plot(x, MAC_pJ_cards, color='blue', marker="s", markersize=6, label='cards')
plt.plot(x, MAC_pJ_centroids, color='black', marker="^", markersize=6, label='k-means')
plt.ylim(bottom=0)
plt.xticks(x)
# plt.xticks([0.08, 0.12])
# plt.yticks([])
plt.xlabel('Variance')
plt.ylabel('MAC / pJ')
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='dotted')
fig = plt.gcf()
# fig.set_size_inches(4., 2.5)
plt.tight_layout()
plt.legend()
fig.savefig('mac_per_pJ.png', dpi=300)

plt.cla()
ax = plt.gca()
# plt.plot(x, y_std[0, 0, :, plot_layer], color='green', marker="D", markersize=5, label='baseline')

# plt.plot(x, y_std[1, 0, 0, :, plot_layer], color='green', marker="D", markersize=5, label='skip')
# plt.plot(x, y_std[1, 1, 0, :, plot_layer], color='blue', marker="s", markersize=6, label='cards')
# plt.plot(x, y_std[1, 1, 1, :, plot_layer], color='black', marker="^", markersize=6, label='k-means')

plt.plot(x, np.mean(y_std[1, 0, 0, :, :], axis=1), color='green', marker="D", markersize=5, label='skip')
plt.plot(x, np.mean(y_std[1, 1, 0, :, :], axis=1), color='blue', marker="s", markersize=6, label='cards')
plt.plot(x, np.mean(y_std[1, 1, 1, :, :], axis=1), color='black', marker="^", markersize=6, label='k-means')

plt.ylim(bottom=0, top=1)
plt.xticks(x)
# plt.xticks([0.08, 0.12])
# plt.yticks([])
plt.xlabel('Variance')
plt.ylabel('Mean Squared Error')
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='dotted')
fig = plt.gcf()
# fig.set_size_inches(4., 2.5)
plt.tight_layout()
plt.legend()
fig.savefig('mse.png', dpi=300)

'''
plt.cla()
ax = plt.gca()
plt.plot(x, acc[0, 0, :], color='green', marker="D", markersize=5, label='baseline')
plt.plot(x, acc[1, 0, :], color='blue', marker="s", markersize=5, label='skip')
plt.plot(x, acc[1, 1, :], color='black', marker="^", markersize=6, label='cards')
plt.ylim(bottom=0, top=1)
plt.xticks([0.08, 0.12])
# plt.yticks([])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='dotted')
fig = plt.gcf()
fig.set_size_inches(4., 2.5)
plt.tight_layout()
fig.savefig('acc.png', dpi=300)
'''

####################

# print (y_std[1, 0, :, :])
# print ('------')
# print (y_std[1, 1, :, :])

'''
print ('mac / pJ')
print (np.around(y_mac_per_pJ[1, 1, 1, :, plot_layer] / y_mac_per_pJ[1, 0, 0, :, plot_layer], 3))
print ('mac / cycle')
print (np.around(y_mac_per_cycle[1, 1, 1, :, plot_layer] / y_mac_per_cycle[1, 0, 0, :, plot_layer], 3))
print ('mse')
print (np.around(y_std[1, 1, 1, :, :] / y_std[1, 0, 0, :, :], 2))

print ('----------')
print ('----------')
print ('----------')

print ('mac / pJ')
print (np.around(y_mac_per_pJ[1, 1, 1, :, plot_layer] / y_mac_per_pJ[1, 1, 0, :, plot_layer], 3))
print ('mac / cycle')
print (np.around(y_mac_per_cycle[1, 1, 1, :, plot_layer] / y_mac_per_cycle[1, 1, 0, :, plot_layer], 3))
print ('mse')
print (np.around(y_std[1, 1, 1, :, :] / y_std[1, 1, 0, :, :], 2))
'''

####################

print ('mac / pJ')
print (np.around(MAC_pJ_centroids / MAC_pJ_cards, 3))
print ('mac / cycle')
print (np.around(TOPs_centroids / TOPs_cards, 3))
print ('mse')
print (np.around(y_std[1, 1, 1, :, :] / y_std[1, 1, 0, :, :], 2))

print ('----------')
print ('----------')
print ('----------')

print ('mac / pJ')
print (np.around(MAC_pJ_centroids / MAC_pJ_skip, 3))
print ('mac / cycle')
print (np.around(TOPs_centroids / TOPs_skip, 3))
print ('mse')
print (np.around(y_std[1, 1, 1, :, :] / y_std[1, 0, 0, :, :], 2))

print ('----------')
print ('----------')
print ('----------')

print ('mac / pJ')
print (np.around(MAC_pJ_cards / MAC_pJ_skip, 3))
print ('mac / cycle')
print (np.around(TOPs_cards / TOPs_skip, 3))
print ('mse')
print (np.around(y_std[1, 1, 0, :, :] / y_std[1, 0, 0, :, :], 2))


####################

















