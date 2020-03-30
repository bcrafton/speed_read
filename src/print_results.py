
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

x = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])

y_mean = np.zeros(shape=(2, 2, len(x), num_layers))
y_std = np.zeros(shape=(2, 2, len(x), num_layers))

for key in sorted(results.keys()):
    (skip, cards, sigma) = key
    layer_results = results[key]

    for layer in range(num_layers):
        example_results = merge_dicts(layer_results[layer])
        sigma_index = np.where(x == sigma)[0][0]
        
        y_mean[skip][cards][sigma_index][layer] = np.mean(example_results['mean'])
        y_std[skip][cards][sigma_index][layer] = np.mean(example_results['std'])

####################

print ('mean')
# print (np.around(y_mean[0, 0],  3))
print (np.around(y_mean[1, 0],  3))
# print (np.around(y_mean[1, 1],  3))

print ('std')
# print (np.around(y_std[0, 0],  3))
print (np.around(y_std[1, 0],  3))
# print (np.around(y_std[1, 1],  3))

####################






















