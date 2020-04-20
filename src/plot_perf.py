
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

num_layers = 20
num_comparator = 8
results = np.load('results.npy', allow_pickle=True).item()

####################

mean = {}
std = {}

mac_per_cycle = {}
mac_per_pJ = {}

cycle = {}
nmac = {}
array = {}

ron = {}
roff = {}
adc = {}
energy = {}

array_util = {}

####################

for key in sorted(results.keys()):

    ###################################

    (skip, cards, alloc, profile, narray) = key
    layer_results = results[key]

    ###################################

    y_mean = np.zeros(shape=num_layers)
    y_std = np.zeros(shape=num_layers)

    y_mac_per_cycle = np.zeros(shape=num_layers)
    y_mac_per_pJ = np.zeros(shape=num_layers)

    y_cycle = np.zeros(shape=num_layers)
    y_nmac = np.zeros(shape=num_layers)
    y_array = np.zeros(shape=num_layers)

    y_ron = np.zeros(shape=num_layers)
    y_roff = np.zeros(shape=num_layers)
    y_adc = np.zeros(shape=num_layers)
    y_energy = np.zeros(shape=num_layers)

    y_array_util = np.zeros(shape=num_layers)
    
    ###################################

    max_cycle = 0
    for layer in range(num_layers):
    
        rdict = merge_dicts(layer_results[layer])
        
        ############################
        
        y_mean[layer] = np.mean(rdict['mean'])
        y_std[layer] = np.mean(rdict['std'])
        
        ############################
        
        y_cycle[layer] = np.mean(rdict['cycle'])
        y_nmac[layer] = np.mean(rdict['nmac'])
        y_array[layer] = np.mean(rdict['array'])
        y_mac_per_cycle[layer]  = np.sum(rdict['nmac']) / np.sum(rdict['cycle'])

    ###################################

    cycle[key] = y_cycle

############################

print (cycle)


















