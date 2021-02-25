
import numpy as np
import sys

def quantize(x):
    scale = np.max(np.absolute(x)) / 127
    x = x / scale
    x = np.around(x)
    x = np.clip(x, -128, 127)
    return x, scale

weights = np.load('bert.npy', allow_pickle=True).item()

def memory(dic):
    total = 0.
    for key in dic.keys():
        if isinstance(dic[key], dict):
            total += memory(dic[key])
        elif isinstance(dic[key], np.ndarray):
            matrix = dic[key].flatten()
            shape = np.shape(matrix)
            size = sys.getsizeof(matrix)
            total += size
    return total

print (memory(weights['embed']) / 1e9)
    
