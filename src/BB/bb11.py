
import numpy as np
import copy

############################

factor = np.array([288, 144, 72, 40, 20, 4])
narray = 4096
nlayer = 6

filter_shape = [
(3*3*  3* 64),
(3*3* 64* 64),
(3*3* 64*128),
(3*3*128*128),
(3*3*128*256),
(3*3*256*256)
]

input_shape = [
(32*32),
(32*32),
(16*16),
(16*16),
(8*8),
(8*8)
]

nmac = np.array([37748736, 18874368, 37748736, 18874368, 37748736, 1769472])
ndup = narray // factor

############################

# array_density = np.array([0.06, 0.12, 0.13, 0.145, 0.19, 0.41])
array_density = np.array([0.024, 0.085, 0.115, 0.114, 0.165, 0.41])

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
        self.row_per_array = np.ceil(params['wl'] * array_density / params['adc']) * params['adc']
        self.mac_per_array = ((params['wl'] / self.row_per_array * (params['bl'] / params['adc_mux'])) / params['adc'])

    def bound(self):
        layer = len(self.narray)
        remainder = narray - np.sum(self.narray)

        if layer == 0:
            min_cycle = np.sum(nmac / self.mac_per_array) / remainder
        elif layer == nlayer:
            min_cycle = np.max(nmac / self.mac_per_array / self.narray)
        else:
            actual = np.max(nmac[0:layer] / self.mac_per_array[0:layer] / self.narray)
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

        scale = input_shape / (np.ceil(input_shape / factor) * factor)
        scaled_mac_per_array = self.mac_per_array * scale
        assert (len(scale) == nlayer)
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

min_branch = branches[0]
for branch in branches:
    if branch.value() < min_branch.value():
        min_branch = branch

    # print (np.sum(branch.narray), branch.narray, branch.value())

print ()
print (min_branch.value())
print (min_branch.narray)

############################



















