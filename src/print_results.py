
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

y_mean = np.zeros(shape=(2, 2, 2, num_layers))
y_std = np.zeros(shape=(2, 2, 2, num_layers))

y_mac_per_cycle = np.zeros(shape=(2, 2, 2, num_layers))
y_mac_per_pJ = np.zeros(shape=(2, 2, 2, num_layers))

cycle = np.zeros(shape=(2, 2, 2, num_layers))
nmac = np.zeros(shape=(2, 2, 2, num_layers))
array = np.zeros(shape=(2, 2, 2, num_layers))

y_ron = np.zeros(shape=(2, 2, 2, num_layers))
y_roff = np.zeros(shape=(2, 2, 2, num_layers))
y_adc = np.zeros(shape=(2, 2, 2, num_layers, num_comparator))
y_energy = np.zeros(shape=(2, 2, 2, num_layers))

array_util = np.zeros(shape=(2, 2, 2, num_layers))

for key in sorted(results.keys()):
    (skip, cards, alloc, profile) = key
    alloc = 1 if alloc == 'block' else 0
    layer_results = results[key]

    max_cycle = 0
    for layer in range(num_layers):
    
        rdict = merge_dicts(layer_results[layer])
        
        ############################
        
        y_mean[skip][cards][alloc][layer] = np.mean(rdict['mean'])
        y_std[skip][cards][alloc][layer] = np.mean(rdict['std'])
        
        ############################

        y_ron[skip][cards][alloc][layer] = np.sum(rdict['ron'])
        y_roff[skip][cards][alloc][layer] = np.sum(rdict['roff'])
        y_adc[skip][cards][alloc][layer] = np.sum(rdict['adc'], axis=0)
        y_energy[skip][cards][alloc][layer] += y_ron[skip][cards][alloc][layer] * 2e-16
        y_energy[skip][cards][alloc][layer] += y_roff[skip][cards][alloc][layer] * 2e-16
        y_energy[skip][cards][alloc][layer] += np.sum(y_adc[skip][cards][alloc][layer] * np.array([1,2,3,4,5,6,7,8]) * comp_pJ)

        y_mac_per_cycle[skip][cards][alloc][layer]  = np.sum(rdict['nmac']) / np.sum(rdict['cycle'])
        y_mac_per_pJ[skip][cards][alloc][layer] = np.sum(rdict['nmac']) / 1e12 / np.sum(y_energy[skip][cards][alloc][layer])
        
        ############################
        
        cycle[skip][cards][alloc][layer] = np.mean(rdict['cycle'])
        nmac[skip][cards][alloc][layer] = np.mean(rdict['nmac'])
        array[skip][cards][alloc][layer] = np.mean(rdict['array'])
        
        ############################

        max_cycle = max(max_cycle, np.mean(rdict['cycle']))
        
        ############################

    for layer in range(num_layers):

        rdict = merge_dicts(layer_results[layer])
        
        ############################

        y_cycle = np.mean(rdict['cycle'])
        y_stall = np.mean(rdict['stall'])
        y_array = np.mean(rdict['array'])
        array_util[skip][cards][alloc][layer] = (y_array * y_cycle - y_stall) / (y_array * max_cycle)
        
        ############################

####################

print ('mean')
# print (np.around(y_mean[0, 0],  3))
# print (np.around(y_mean[1, 0],  3))
# print (np.around(y_mean[1, 1],  3))
print (np.around(y_mean, 3))

print ('std')
# print (np.around(y_std[0, 0],  3))
# print (np.around(y_std[1, 0],  3))
# print (np.around(y_std[1, 1],  3))
print (np.around(y_std, 3))

print ('mac / cycle')
# print (np.around(y_mac_per_cycle[0, 0], 1))
# print (np.around(y_mac_per_cycle[1, 0], 1))
# print (np.around(y_mac_per_cycle[1, 1], 1))
print (np.around(y_mac_per_cycle, 1))

print ('cycle')
# print (np.around(y_mac_per_cycle[0, 0], 1))
# print (np.around(y_mac_per_cycle[1, 0], 1))
# print (np.around(y_mac_per_cycle[1, 1], 1))
print (np.around(cycle, 1))

####################






















