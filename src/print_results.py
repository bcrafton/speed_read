
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

num_layers = 20
results = np.load('results.npy', allow_pickle=True).item()

y_mean  = np.zeros(shape=(2, 2, 2, num_layers))
y_std   = np.zeros(shape=(2, 2, 2, num_layers))
y_cycle = np.zeros(shape=(2, 2, 2, num_layers))
y_stall = np.zeros(shape=(2, 2, 2, num_layers))
y_array = np.zeros(shape=(2, 2, 2, num_layers))

for key in sorted(results.keys()):
    print (key)

    (skip, cards, alloc, profile) = key
    alloc = 1 if alloc == 'block' else 0
    layer_results = results[key]

    for layer in range(num_layers):
    
        rdict = merge_dicts(layer_results[layer])
        
        ############################
        
        y_mean[skip][cards][alloc][layer]  = np.mean(rdict['mean'])
        y_std[skip][cards][alloc][layer]   = np.mean(rdict['std'])
        y_cycle[skip][cards][alloc][layer] = np.mean(rdict['cycle'])
        y_stall[skip][cards][alloc][layer] = np.mean(rdict['stall'])
        y_array[skip][cards][alloc][layer] = np.mean(rdict['array'])

        ############################

print (y_cycle)
print (y_std)
print (y_mean)
















