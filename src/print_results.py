
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

num_layers = 1
results = np.load('results.npy', allow_pickle=True).item()

for key in sorted(results.keys()):
    print (key)

    (skip, cards, alloc, profile) = key
    alloc = 1 if alloc == 'block' else 0
    layer_results = results[key]

    for layer in range(num_layers):
    
        rdict = merge_dicts(layer_results[layer])
        
        ############################
        
        y_mean  = np.mean(rdict['mean'])
        y_std   = np.mean(rdict['std'])
        y_cycle = np.mean(rdict['cycle'])
        y_stall = np.mean(rdict['stall'])
        y_array = np.mean(rdict['array'])
        
        ############################

        print ('mean', y_mean)
        print ('std', y_std)
        print ('cycles', y_cycle)

        ############################



















