
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

num_layers = 7
num_comparator = 8
results = np.load('results.npy', allow_pickle=True).item()

x = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])

y_mean = np.zeros(shape=(2, 2, len(x), num_layers))
y_std = np.zeros(shape=(2, 2, len(x), num_layers))

y_perf = np.zeros(shape=(2, 2, len(x), num_layers))
y_ron = np.zeros(shape=(2, 2, len(x), num_layers))
y_roff = np.zeros(shape=(2, 2, len(x), num_layers))
y_adc = np.zeros(shape=(2, 2, len(x), num_layers, num_comparator))

y_power = np.zeros(shape=(2, 2, len(x), num_layers))
acc = np.zeros(shape=(2, 2, len(x)))

####################

for key in sorted(results.keys()):
    (skip, cards, sigma) = key
    layer_results = results[key]

    for layer in range(num_layers):
        example_results = merge_dicts(layer_results[layer])
        sigma_index = np.where(x == sigma)[0][0]
        y_perf[skip][cards][sigma_index][layer] = np.sum(example_results['nmac']) / np.sum(example_results['cycle'])
        y_mean[skip][cards][sigma_index][layer] = np.mean(example_results['mean'])
        y_std[skip][cards][sigma_index][layer] = np.mean(example_results['std'])
        acc[skip][cards][sigma_index] = layer_results['acc']
        
        y_ron[skip][cards][sigma_index][layer] = np.mean(example_results['ron'])
        y_roff[skip][cards][sigma_index][layer] = np.mean(example_results['roff'])
        y_adc[skip][cards][sigma_index][layer] = np.mean(example_results['adc'], axis=0)
        
        y_power[skip][cards][sigma_index][layer] += y_ron[skip][cards][sigma_index][layer] * 1e-12
        y_power[skip][cards][sigma_index][layer] += y_roff[skip][cards][sigma_index][layer] * 1e-13
        y_power[skip][cards][sigma_index][layer] += y_roff[skip][cards][sigma_index][layer] * 1e-13

####################

print (y_ron[0, 0, 0])

####################


