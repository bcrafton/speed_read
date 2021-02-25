
import numpy as np
import sys

def quantize(x):
    scale = np.max(np.absolute(x)) / 127
    x = x / scale
    x = np.around(x)
    x = np.clip(x, -128, 127)
    return x, scale

weights = np.load('bert.npy', allow_pickle=True).item()

# if isinstance(dic[key], dict):
# elif isinstance(dic[key], np.ndarray):

'''
for key in weights['embed'].keys():
    print (key)
for key in weights['encoder'].keys():
    print (key)
for key in weights['pool'].keys():
    print (key)
for key in weights['class'].keys():
    print (key)
'''

for key1 in weights['encoder'].keys():
    for key2 in weights['encoder'][key1].keys():
        qw, scale = quantize(weights['encoder'][key1][key2]['w'])
        weights['encoder'][key1][key2]['w'] = qw.astype(np.int8)
        weights['encoder'][key1][key2]['s'] = scale
                
qw, scale = quantize(weights['pool']['w'])
weights['pool']['w'] = qw
weights['pool']['s'] = scale

qw, scale = quantize(weights['class']['w'])
weights['class']['w'] = qw
weights['class']['s'] = scale

np.save('bert8b', weights)
