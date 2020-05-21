
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

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

print (results.keys())

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
density = {}

####################

for key in sorted(results.keys()):

    ###################################

    print (key)
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
    y_density = np.zeros(shape=num_layers)
    
    ###################################

    max_cycle = 0
    for layer in range(num_layers):
    
        rdict = merge_dicts(layer_results[layer])
        
        ############################
        
        y_mean[layer] = np.mean(rdict['mean'])
        y_std[layer] = np.mean(rdict['std'])

        # print ('mean', y_mean[layer], 'std', y_std[layer])
        
        ############################
        
        y_cycle[layer] = np.mean(rdict['cycle'])
        y_nmac[layer] = np.mean(rdict['nmac'])
        y_array[layer] = np.mean(rdict['array'])
        y_mac_per_cycle[layer]  = np.sum(rdict['nmac']) / np.sum(rdict['cycle'])

        y_ron = np.sum(rdict['ron'])
        y_roff = np.sum(rdict['roff'])
        y_adc = np.sum(rdict['adc'], axis=0)
        
        # y_energy[layer] += y_ron * 2e-16
        # y_energy[layer] += y_roff * 2e-16
        y_energy[layer] += np.sum(y_adc * np.array([1,2,3,4,5,6,7,8]) * comp_pJ)
        y_mac_per_pJ[layer] = np.sum(rdict['nmac']) / 1e12 / np.sum(y_energy[layer])
        # print (layer, 'mac/pJ', y_mac_per_pJ[layer], 'std', y_std[layer], 'nmac', np.sum(rdict['nmac']))

        y_density[layer] = rdict['density'][0]

    ###################################

    cycle[key] = y_cycle
    nmac[key] = y_nmac
    array[key] = y_array
    mac_per_pJ[key] = y_mac_per_pJ
    density[key] = y_density

    # total_mac_per_pJ = np.sum(y_nmac) / np.sum(y_nmac / y_mac_per_pJ)
    total_mac_per_pJ = np.sum(y_nmac) / 1e12 / np.sum(y_energy)

    print (y_mac_per_pJ)
    print (y_density)
    print (total_mac_per_pJ)
    # print (y_energy)

############################

# print (mac_per_pJ[(1, 1, 'block', 1, 8192)] / mac_per_pJ[(1, 0, 'block', 1, 8192)])

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
    
    ys[config][lut[narray]] = perf * 100e6 / 1e12
    
############################

'''
print (ys.keys())

for key in ys.keys():
    plt.plot(lut.keys(), ys[key], marker='.')
'''

############################

# config = (skip, cards, alloc, profile)
keys = [
(0, 0, 'block', 0), 
(0, 0, 'block', 1), 
(0, 0, 'layer', 0), 
(0, 0, 'layer', 1), 
(1, 0, 'block', 0), 
(1, 0, 'block', 1), 
(1, 0, 'layer', 0), 
(1, 0, 'layer', 1)
]

'''
plt.plot(lut.keys(), ys[(0, 0, 'layer', 0)], marker='.', label='Baseline')
plt.plot(lut.keys(), ys[(1, 0, 'layer', 0)], marker='.', label='Zero Skip')

plt.plot(lut.keys(), ys[(0, 0, 'layer', 1)], marker='.', label='Baseline')
plt.plot(lut.keys(), ys[(1, 0, 'layer', 1)], marker='.', label='Zero Skip')

plt.plot(lut.keys(), ys[(0, 0, 'block', 1)], marker='.', label='Baseline')
plt.plot(lut.keys(), ys[(1, 0, 'block', 1)], marker='.', label='Zero Skip')
'''

x = sorted(lut.keys())
plt.plot(x, ys[(1, 0, 'block', 1)], marker='.', label='ZS')
plt.plot(x, ys[(1, 1, 'block', 1)], marker='.', label='CC')
# plt.plot(x, ys[(1, 0, 'layer', 1)], marker='.', label='Perf-Based Layer-wise')
# plt.plot(x, ys[(1, 0, 'layer', 0)], marker='.', label='Weight-Based')
# plt.plot(x, ys[(0, 0, 'layer', 1)], marker='.', label='Baseline')

############################

# print (ys[(1, 0, 'layer', 1)] / ys[(0, 0, 'layer', 1)])

# print (cycle[(0, 0, 'layer', 1, 2 ** 14)])
# print (np.sum(array[(0, 0, 'layer', 1, 2 ** 14)]), array[(0, 0, 'layer', 1, 2 ** 14)])

############################

# plt.ylim(bottom=0, top=1e4)
plt.xticks( list(lut.keys()) )
plt.legend()
plt.grid(True, linestyle='dotted')
#plt.show()

plt.xlabel('Arrays / Design')
plt.ylabel('Performance (TMAC/s)')

############################

fig = plt.gcf()
fig.set_size_inches(4, 3)
plt.tight_layout()
fig.savefig('perf.png', dpi=300)

############################








