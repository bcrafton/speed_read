
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

comp_pJ = 22. * 1e-12 / 32. / 16.

num_layers = 6
num_comparator = 8
results = np.load('results.npy', allow_pickle=True).item()

x = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])

y_mean = np.zeros(shape=(2, 2, len(x), num_layers))
y_std = np.zeros(shape=(2, 2, len(x), num_layers))

y_mac_per_cycle = np.zeros(shape=(2, 2, len(x), num_layers))
y_mac_per_pJ = np.zeros(shape=(2, 2, len(x), num_layers))

cycle = np.zeros(shape=(2, 2, len(x), num_layers))
nmac = np.zeros(shape=(2, 2, len(x), num_layers))
array = np.zeros(shape=(2, 2, len(x), num_layers))

y_ron = np.zeros(shape=(2, 2, len(x), num_layers))
y_roff = np.zeros(shape=(2, 2, len(x), num_layers))
y_adc = np.zeros(shape=(2, 2, len(x), num_layers, num_comparator))
y_energy = np.zeros(shape=(2, 2, len(x), num_layers))

array_util = np.zeros(shape=(2, 2, len(x), num_layers))

for key in sorted(results.keys()):
    (skip, cards, sigma) = key
    layer_results = results[key]

    max_cycle = 0
    for layer in range(num_layers):
    
        rdict = merge_dicts(layer_results[layer])
        sigma_index = np.where(x == sigma)[0][0]
        
        ############################
        
        y_mean[skip][cards][sigma_index][layer] = np.mean(rdict['mean'])
        y_std[skip][cards][sigma_index][layer] = np.mean(rdict['std'])
        
        ############################

        y_ron[skip][cards][sigma_index][layer] = np.sum(rdict['ron'])
        y_roff[skip][cards][sigma_index][layer] = np.sum(rdict['roff'])
        y_adc[skip][cards][sigma_index][layer] = np.sum(rdict['adc'], axis=0)
        y_energy[skip][cards][sigma_index][layer] += y_ron[skip][cards][sigma_index][layer] * 2e-16
        y_energy[skip][cards][sigma_index][layer] += y_roff[skip][cards][sigma_index][layer] * 2e-16
        y_energy[skip][cards][sigma_index][layer] += np.sum(y_adc[skip][cards][sigma_index][layer] * np.array([1,2,3,4,5,6,7,8]) * comp_pJ)

        y_mac_per_cycle[skip][cards][sigma_index][layer]  = np.sum(rdict['nmac']) / np.sum(rdict['cycle'])
        y_mac_per_pJ[skip][cards][sigma_index][layer] = np.sum(rdict['nmac']) / 1e12 / np.sum(y_energy[skip][cards][sigma_index][layer])
        
        ############################
        
        cycle[skip][cards][sigma_index][layer] = np.mean(rdict['cycle'])
        nmac[skip][cards][sigma_index][layer] = np.mean(rdict['nmac'])
        array[skip][cards][sigma_index][layer] = np.mean(rdict['array'])
        
        ############################

        max_cycle = max(max_cycle, np.mean(rdict['cycle']))
        
        ############################

    for layer in range(num_layers):

        rdict = merge_dicts(layer_results[layer])
        sigma_index = np.where(x == sigma)[0][0]
        
        ############################

        y_cycle = np.mean(rdict['cycle'])
        y_stall = np.mean(rdict['stall'])
        y_array = np.mean(rdict['array'])
        array_util[skip][cards][sigma_index][layer] = (y_array * y_cycle - y_stall) / (y_array * max_cycle)
        
        ############################

####################

print ('cycle')
# print (np.around(cycle[0, 0, 0], 1))
print (np.around(cycle[1, 0, 0], 1))
# print (np.around(cycle[1, 1, 0], 1))

print ('nmac')
# print (np.around(nmac[0, 0, 0], 1))
print (np.around(nmac[1, 0, 0], 1))
# print (np.around(nmac[1, 1, 0], 1))

print ('array')
# print (np.around(array[0, 0, 0], 1))
print (np.around(array[1, 0, 0], 1))
# print (np.around(array[1, 1, 0], 1))

####################

# print ('mean')
# print (np.around(y_mean[0, 0],  3))
# print (np.around(y_mean[1, 0],  3))
# print (np.around(y_mean[1, 1],  3))

# print ('std')
# print (np.around(y_std[0, 0],  3))
# print (np.around(y_std[1, 0],  3))
# print (np.around(y_std[1, 1],  3))

print ('mac / cycle')
# print (np.around(y_mac_per_cycle[0, 0, 0], 1))
print (np.around(y_mac_per_cycle[1, 0, 0], 1))
# print (np.around(y_mac_per_cycle[1, 1, 0], 1))

print ('mac / cycle / array')
print (np.around(y_mac_per_cycle[1, 0, 0] / array[1, 0, 0], 1))

# print ('mac / pJ')
# print (np.around(y_mac_per_pJ[0, 0],  3))
# print (np.around(y_mac_per_pJ[1, 0],  3))
# print (np.around(y_mac_per_pJ[1, 1],  3))

print ('array util')
# print (np.around(array_util[0, 0, 0],  3))
print (np.around(array_util[1, 0, 0],  3))
# print (np.around(array_util[1, 1, 0],  3))

####################

layers = np.array(range(1, 6+1))
baseline = np.around(array_util[0, 0, 0],  3)
zero_skip = np.around(array_util[1, 0, 0],  3)
cards = np.around(array_util[1, 1, 0],  3)

'''
width = 0.35
plt.bar(x=layers - width/2, height=baseline, width=width, label='baseline')
plt.bar(x=layers + width/2, height=zero_skip, width=width, label='zero skip')
plt.legend()
'''

width = 0.2
plt.bar(x=layers - width, height=baseline,  width=width, label='baseline')
plt.bar(x=layers,         height=zero_skip, width=width, label='zero skip')
plt.bar(x=layers + width, height=cards,     width=width, label='counting cards')
plt.legend()

plt.ylabel('Array Utilization')
plt.xlabel('Layer #')
plt.ylim(0, 1)

# plt.show()
plt.savefig('util.png', dpi=300)


















