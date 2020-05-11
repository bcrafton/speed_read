
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

num_layers = 20
results = np.load('results.npy', allow_pickle=True).item()

####################

# key = list(results.keys())[0]
key = (1, 0, 'block', 1, 24576.0)
(skip, cards, alloc, profile, narray) = key
alloc = 1 if alloc == 'block' else 0
layer_results = results[key]

# why is there a difference between performance here ? 
# think it has to do with stalls, since block-wise only counts actice cycles.
# and there are probably 45 duplicates.
print (layer_results.keys())
# print (layer_results['layer_mac'])
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

s = 0
for layer in range(num_layers):
    e = s + len(block_density[layer][0])
    # print (layer, s, e)
    print (layer, (s, e), layer_results['block_mac'][s:e])
    s = e

####################

# x = block_density[16][0]
# y = layer_results['block_mac'][137:173]

x1 = block_density[15][0] * 100
y1 = 128 * 16 / layer_results['block_mac'][119:137]

x2 = block_density[10][0] * 100
y2 = 128 * 16 / layer_results['block_mac'][55:64]

# print (x)
# print (y)

plt.scatter(x1, y1, marker='+', color='blue')
plt.scatter(x2, y2, marker='x', color='black')
    
####################

fig = plt.gcf()
ax = plt.gca()

# plt.ylim(bottom=0)
# plt.ylabel('Cycle / Array')
# plt.xlabel('Percent (%) 1s')
# plt.show()

plt.grid(True, linestyle='dotted')
plt.xticks([3, 4, 5, 6, 7, 8])
plt.yticks([90, 100, 110, 120, 130])

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

fig.set_size_inches(3.5, 2.25)
plt.tight_layout()
fig.savefig('density_vs_perf2.png', dpi=300)

####################

















