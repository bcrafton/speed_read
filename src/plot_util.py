
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

num_layers = 10
num_comparator = 8
results = np.load('results.npy', allow_pickle=True).item()

y_mean = np.zeros(shape=(2, 2, 2, 2, num_layers))
y_std = np.zeros(shape=(2, 2, 2, 2, num_layers))

y_mac_per_cycle = np.zeros(shape=(2, 2, 2, 2, num_layers))
y_mac_per_pJ = np.zeros(shape=(2, 2, 2, 2, num_layers))

cycle = np.zeros(shape=(2, 2, 2, 2, num_layers))
nmac = np.zeros(shape=(2, 2, 2, 2, num_layers))
array = np.zeros(shape=(2, 2, 2, 2, num_layers))

y_ron = np.zeros(shape=(2, 2, 2, 2, num_layers))
y_roff = np.zeros(shape=(2, 2, 2, 2, num_layers))
y_adc = np.zeros(shape=(2, 2, 2, 2, num_layers, num_comparator))
y_energy = np.zeros(shape=(2, 2, 2, 2, num_layers))

array_util = np.zeros(shape=(2, 2, 2, 2, num_layers))

for key in sorted(results.keys()):
    print (key)
    (skip, cards, alloc, profile, narray) = key
    alloc = 1 if alloc == 'block' else 0
    layer_results = results[key]

    max_cycle = 0
    for layer in range(num_layers):
    
        rdict = merge_dicts(layer_results[layer])
        
        ############################
        
        y_mean[skip][cards][alloc][profile][layer] = np.mean(rdict['mean'])
        y_std[skip][cards][alloc][profile][layer] = np.mean(rdict['std'])
        
        ############################

        y_ron[skip][cards][alloc][profile][layer] = np.sum(rdict['ron'])
        y_roff[skip][cards][alloc][profile][layer] = np.sum(rdict['roff'])
        y_adc[skip][cards][alloc][profile][layer] = np.sum(rdict['adc'], axis=0)
        y_energy[skip][cards][alloc][profile][layer] += y_ron[skip][cards][alloc][profile][layer] * 2e-16
        y_energy[skip][cards][alloc][profile][layer] += y_roff[skip][cards][alloc][profile][layer] * 2e-16
        y_energy[skip][cards][alloc][profile][layer] += np.sum(y_adc[skip][cards][alloc][profile][layer] * np.array([1,2,3,4,5,6,7,8]) * comp_pJ)

        y_mac_per_cycle[skip][cards][alloc][profile][layer]  = np.sum(rdict['nmac']) / np.sum(rdict['cycle'])
        y_mac_per_pJ[skip][cards][alloc][profile][layer] = np.sum(rdict['nmac']) / 1e12 / np.sum(y_energy[skip][cards][alloc][profile][layer])
        
        ############################
        
        cycle[skip][cards][alloc][profile][layer] = np.mean(rdict['cycle'])
        nmac[skip][cards][alloc][profile][layer] = np.mean(rdict['nmac'])
        array[skip][cards][alloc][profile][layer] = np.mean(rdict['array'])
        
        ############################

        max_cycle = max(max_cycle, np.mean(rdict['cycle']))
        
        ############################

    for layer in range(num_layers):

        rdict = merge_dicts(layer_results[layer])
        
        ############################

        y_cycle = np.mean(rdict['cycle'])
        y_stall = np.mean(rdict['stall'])
        y_array = np.mean(rdict['array'])
        array_util[skip][cards][alloc][profile][layer] = (y_array * y_cycle - y_stall) / (y_array * max_cycle)
        
        ############################



####################

layers = np.array(range(1, num_layers+1))

####################

skip_none_array  = np.around(array[1, 0, 0, 0],  3)
skip_layer_array  = np.around(array[1, 0, 0, 1],  3)
skip_block_array  = np.around(array[1, 0, 1, 1],  3)

cards_none_array = np.around(array[1, 1, 0, 0],  3)
cards_layer_array = np.around(array[1, 1, 0, 1],  3)
cards_block_array = np.around(array[1, 1, 1, 1],  3)

####################

skip_none  = np.around(array_util[1, 0, 0, 0],  3)
skip_layer  = np.around(array_util[1, 0, 0, 1],  3)
skip_block  = np.around(array_util[1, 0, 1, 1],  3)

####################

print ()
print (skip_block / skip_layer)
print (skip_block / skip_none)

'''
this is wild, how do we get these #s ? 
looks like we calculated something this way instead of observing it.
well utilization is a function of the max cycle across all layers so it sorta makes sense.
if we were going to discuss utilization i dont think we wud be looking at these prints below.
'''
print ()
print ((skip_block_array * skip_block) / (skip_layer_array * skip_layer))
print ((skip_block_array * skip_block) / (skip_none_array * skip_none))

####################

plt.rcParams.update({'font.size': 10})

####################

plt.cla()
plt.clf()
plt.close()

fig = plt.gcf()
ax = plt.gca()

width = 0.2
plt.bar(x=layers - width, height=skip_none,   width=width, label='weight-based', color='silver')
plt.bar(x=layers,         height=skip_layer,  width=width, label='performance-based layer-wise', color='royalblue')
plt.bar(x=layers + width, height=skip_block,  width=width, label='performance-based block-wise', color='black')

# plt.ylabel('Array Utilization (%)')
# plt.xlabel('Layer #')
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

plt.xticks(layers)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# plt.grid(True, axis='y', linestyle='dotted', color='black')
# plt.grid(True, axis='y', linestyle=(0, (5, 10)), color='black')
plt.grid(True, axis='y', linestyle=(0, (5, 8)), color='black')
# plt.grid(True, axis='y', linestyle='--', color='black')

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

fig.set_size_inches(8., 2.5)
plt.tight_layout()
fig.savefig('skip-util.png', dpi=300)

####################







