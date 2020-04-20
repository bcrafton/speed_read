
import numpy as np
import copy

def array_allocation(narray, nmac, factor, mac_per_array, params):
    if (np.sum(factor) > narray): return np.inf
    
    alloc = copy.copy(factor)    
    cycles = nmac / mac_per_array / alloc
    argmax = np.argmax(cycles)
    while (np.sum(alloc) + factor[argmax]) <= narray:
        alloc[argmax] += factor[argmax]
        cycles = nmac / mac_per_array / alloc
        argmax = np.argmax(cycles)
        
    # print (narray - np.sum(alloc), factor[argmax], np.max(cycles))
    
    return alloc
