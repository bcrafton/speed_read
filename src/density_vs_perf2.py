
import numpy as np
import matplotlib.pyplot as plt

####################

plt.rcParams.update({'font.size': 10})

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

num_layers = 6
results = np.load('results.npy', allow_pickle=True).item()

####################

key = list(results.keys())[0]
(skip, cards, alloc, profile) = key
alloc = 1 if alloc == 'block' else 0
layer_results = results[key]

# why is there a difference between performance here ? 
# think it has to do with stalls, since block-wise only counts actice cycles.
# and there are probably 45 duplicates.
print (layer_results.keys())
print (layer_results['layer_mac'])
# print (layer_results['block_mac'])

density = []
block_density = []
for layer in range(num_layers):
    rdict = merge_dicts(layer_results[layer])

    #print (rdict['density'])
    density.append(rdict['density'])
    
    #print (rdict['block_density'])
    block_density.append(rdict['block_density'])
    
####################

x = block_density[4][0]
y = layer_results['block_mac'][20:29]
print (x)
print (y)
plt.scatter(x, y, marker='o')
    
'''
for layer in range(num_layers):
    print (layer, density[layer], layer_results['layer_mac'][layer])
    plt.annotate(str(layer + 1), (density[layer][0] + np.max(density) * 0.01, layer_results['layer_mac'][layer]))
'''
    
####################
    
# plt.ylim(bottom=0)
plt.ylabel('MAC/Cycle/Array')
plt.xlabel('Percent (%) 1s')
# plt.show()

fig = plt.gcf()
fig.set_size_inches(3.5, 3.)
plt.tight_layout()
fig.savefig('density_vs_perf2.png', dpi=300)

####################

















