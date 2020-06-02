
import numpy as np

'''
num_layers = 6
weights = np.load('cifar10_weights.npy', allow_pickle=True).item()
print (weights.keys())

new_weights = {}
for layer in range(num_layers):
    new_weights[layer] = {}
    new_weights[layer]['f'] = weights[layer]['f']
    new_weights[layer]['b'] = weights[layer]['b']
    new_weights[layer]['y'] = weights[layer]['q']
    
np.save('cifar10_weights', new_weights)
'''

num_layers = 6
weights = np.load('cifar10_weights_old.npy', allow_pickle=True).item()
print (weights.keys())

new_weights = {}
for layer in range(num_layers):
    new_weights[layer] = {}
    f, b, y = weights[layer]
    new_weights[layer]['f'] = f
    new_weights[layer]['b'] = b
    new_weights[layer]['y'] = y
    
np.save('cifar10_weights_old', new_weights)
