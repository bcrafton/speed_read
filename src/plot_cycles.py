
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
    (skip, cards, alloc, profile) = key
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

layers      = np.array(range(1, num_layers+1))

skip_none  = np.around(cycle[1, 0, 0, 0],  3)
skip_layer  = np.around(cycle[1, 0, 0, 1],  3)
skip_block  = np.around(cycle[1, 0, 1, 1],  3)

cards_none = np.around(cycle[1, 1, 0, 0],  3)
cards_layer = np.around(cycle[1, 1, 0, 1],  3)
cards_block = np.around(cycle[1, 1, 1, 1],  3)

####################

plt.rcParams.update({'font.size': 10})

####################

plt.cla()
plt.clf()
plt.close()

plt.ylabel('# Cycles')
plt.xlabel('Layer #')

width = 0.2
plt.bar(x=layers - width, height=skip_none,   width=width, label='weight-based', color='silver')
plt.bar(x=layers,         height=skip_layer,  width=width, label='performance-based layer-wise', color='royalblue')
plt.bar(x=layers + width, height=skip_block,  width=width, label='performance-based block-wise', color='black')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

plt.plot([0., 20.], [np.max(skip_none),  np.max(skip_none)],  "k--", color='silver')
plt.plot([0., 20.], [np.max(skip_layer), np.max(skip_layer)], "k--", color='royalblue')
plt.plot([0., 20.], [np.max(skip_block), np.max(skip_block)], "k--", color='black')
plt.xticks(layers)

fig = plt.gcf()
# fig.set_size_inches(9, 9)
# plt.tight_layout()
fig.savefig('skip-cycles.png', dpi=300)

####################

plt.clf()
plt.cla()
plt.close()

plt.ylabel('# Cycles')
plt.xlabel('Layer #')

width = 0.2
plt.bar(x=layers - width, height=cards_none,   width=width, label='weight-based ', color='silver')
plt.bar(x=layers,         height=cards_layer,  width=width, label='performance-based layer-wise', color='royalblue')
plt.bar(x=layers + width, height=cards_block,  width=width, label='performance-based block-wise', color='black')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

plt.plot([0., 20.], [np.max(cards_none),  np.max(cards_none)],  "k--", color='silver')
plt.plot([0., 20.], [np.max(cards_layer), np.max(cards_layer)], "k--", color='royalblue')
plt.plot([0., 20.], [np.max(cards_block), np.max(cards_block)], "k--", color='black')
plt.xticks(layers)

fig = plt.gcf()
# fig.set_size_inches(9, 9)
# plt.tight_layout()
fig.savefig('cards-cycles.png', dpi=300)

####################






