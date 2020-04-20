
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
    nmac[key] = y_nmac

############################

lut = {
5472: 0, 
2 ** 13: 1, 
1.5 * 2 ** 13: 2, 
2 ** 14: 3, 
1.5 * 2 ** 14: 4
}

ys = {}
for key in cycle.keys():
    (skip, cards, alloc, profile, narray) = key
    ncycle = np.max(cycle[key])
    perf = np.sum(nmac[key]) / np.max(cycle[key])

    config = (skip, cards, alloc, profile)
    if config not in ys.keys():
        ys[config] = np.zeros(5)
    
    ys[config][lut[narray]] = perf # ncycle
    
############################

print (ys.keys())

for key in ys.keys():
    plt.plot(lut.keys(), ys[key], marker='.')

# plt.ylim(bottom=0, top=1e4)
plt.xticks( list(lut.keys()) )
plt.show()

############################














