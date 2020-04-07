
import numpy as np
import copy

############################

factor = np.array([288, 144, 72, 40, 20, 4])
narray = 4096
nlayer = 6

input_shape = [
(8*8),
(8*8),
(16*16),
(16*16),
(32*32),
(32*32),
]

nmac = np.array([37748736, 18874368, 37748736, 18874368, 37748736, 1769472])

############################

# array_density = np.array([0.06, 0.12, 0.13, 0.145, 0.19, 0.41])
array_density = np.array([0.0763, 0.1605, 0.1877, 0.1966, 0.2489, 0.52])

params = {
'bpa': 8,
'bpw': 8,
'adc': 8,
'adc_mux': 8,
'wl': 128,
'bl': 128,
}

############################

class BB:

    def __init__(self, narray, params):
        self.narray = narray
        self.params = params
        self.row_per_array = params['wl'] * array_density
        self.mac_per_array = (params['wl'] / self.row_per_array) * (params['bl'] / params['adc_mux']) / 8

    def bound(self):
        layer = len(self.narray)
        remainder = narray - np.sum(self.narray)

        if layer == 0:
            min_cycle = np.sum(nmac / self.mac_per_array) / remainder
        elif layer == nlayer:
            min_cycle = np.max(nmac / self.mac_per_array / self.narray)
        else:
            ndup = np.array(self.narray) / factor[0:layer]
            scale = input_shape[0:layer] / (np.ceil(input_shape[0:layer] / ndup) * ndup)
            scaled_mac_per_array = self.mac_per_array[0:layer] # * scale
        
            actual = np.max(nmac[0:layer] / scaled_mac_per_array[0:layer] / self.narray)
            upper_bound = np.sum(nmac[layer:] / self.mac_per_array[layer:]) / remainder
            min_cycle = max(actual, upper_bound)
            
        return min_cycle
        
    def value(self):
        upper_bound = self.bound()
    
        new_narray = copy.copy(self.narray)
        remainder = narray - np.sum(self.narray)
        layer = len(self.narray)
        for n in range(layer, nlayer):
            need = np.ceil(nmac[n] / self.mac_per_array[n] / upper_bound)
            need = need - (need % (factor[n]))
            assert (remainder > factor[n])
            new_narray.append(min(need, remainder))
            remainder -= new_narray[n]

        ndup = np.array(new_narray) / factor
        scale = input_shape / (np.ceil(input_shape / ndup) * ndup)
        scaled_mac_per_array = self.mac_per_array # * scale
        # print (self.mac_per_array)
        # print (scaled_mac_per_array)
        max_cycle = np.max(nmac / scaled_mac_per_array / new_narray)
        return max_cycle

    def branch(self, lower_bound):
        layer = len(self.narray)        
        lower_bound = min(lower_bound, self.value())
        branches = []
        remainder = narray - np.sum(self.narray)
        for n in range(factor[layer], int(remainder), factor[layer]):
            new_narray = copy.copy(self.narray)
            new_narray.append(n)
            new_BB = BB(new_narray, params)
            new_bound = new_BB.bound()
            if new_bound <= lower_bound:
                branches.append(new_BB)
        
        return branches

############################

def branch_and_bound_help(branches, lower_bound):
    new_branches = []
    for branch in branches:
        new_branches.extend(branch.branch(lower_bound))
    return new_branches

def branch_and_bound():
    root = BB([], params)
    branches = [root]
    lower_bound = root.value()
    for layer in range(nlayer):
        branches = branch_and_bound_help(branches, lower_bound)
        for branch in branches:
            lower_bound = min(lower_bound, branch.value())
    return branches
        
############################

branches = branch_and_bound()

branch_table = {}
for branch in branches:
    if branch.value() in branch_table.keys():
        branch_table[branch.value()].append(branch)
    else:
        branch_table[branch.value()] = [branch]

best = np.min(list(branch_table.keys()))
print (best)
for branch in branch_table[best]:
    print (np.sum(branch.narray), branch.narray, branch.value())

############################



















