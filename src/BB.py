
import numpy as np
import copy

############################

class BB:

    def __init__(self, narray, nlayer, alloc, mac_per_array, nmac, factor, params):
        self.narray = narray
        self.nlayer = nlayer
        self.alloc = alloc
        self.mac_per_array = mac_per_array
        self.nmac = nmac
        self.factor = factor
        self.params = params

    def bound(self):
        layer = len(self.alloc)
        remainder = self.narray - np.sum(self.alloc)

        if layer == 0:
            min_cycle = np.sum(self.nmac / self.mac_per_array) / remainder
        elif layer == self.nlayer:
            min_cycle = np.max(self.nmac / self.mac_per_array / self.alloc)
        else:
            actual = np.max(self.nmac[0:layer] / self.mac_per_array[0:layer] / self.alloc)
            upper_bound = np.sum(self.nmac[layer:] / self.mac_per_array[layer:]) / remainder
            min_cycle = max(actual, upper_bound)
            
        return min_cycle
        
    def value(self):
        upper_bound = self.bound()
    
        new_alloc = copy.copy(self.alloc)
        remainder = self.narray - np.sum(self.alloc)
        layer = len(self.alloc)
        for n in range(layer, self.nlayer):
            need = np.ceil(self.nmac[n] / self.mac_per_array[n] / upper_bound)
            need = need - (need % (self.factor[n]))
            assert (remainder > self.factor[n])
            new_alloc.append(min(need, remainder))
            remainder -= new_alloc[n]

        max_cycle = np.max(self.nmac / self.mac_per_array / new_alloc)
        return max_cycle

    def branch(self, lower_bound):
        layer = len(self.alloc)        
        lower_bound = min(lower_bound, self.value())
        branches = []
        remainder = self.narray - np.sum(self.alloc)
        for n in range(self.factor[layer], int(remainder), self.factor[layer]):
            new_alloc = copy.copy(self.alloc)
            new_alloc.append(n)
            new_BB = BB(self.narray, self.nlayer, new_alloc, self.mac_per_array, self.nmac, self.factor, self.params)
            new_bound = new_BB.bound()
            if new_bound <= lower_bound:
                branches.append(new_BB)
        
        return branches

############################

def branch_and_bound(narray, layers, mac_per_array, params):
    nlayer = len(layers)
    nmac = np.zeros(shape=nlayer, dtype=np.int32)
    factor = np.zeros(shape=nlayer, dtype=np.int32)
    for layer in range(nlayer):
        nmac[layer] = layers[layer].nmac
        factor[layer] = layers[layer].factor

    ################################
    
    mac_per_array = mac_per_array[::-1]
    nmac = nmac[::-1]
    factor = factor[::-1]
    
    ################################

    def branch_and_bound_help(branches, lower_bound):
        new_branches = []
        for branch in branches:
            new_branches.extend(branch.branch(lower_bound))
        return new_branches

    ################################

    root = BB(narray, nlayer, [], mac_per_array, nmac, factor, params)
    branches = [root]
    lower_bound = root.value()
    for layer in range(nlayer):
        branches = branch_and_bound_help(branches, lower_bound)
        for branch in branches:
            lower_bound = min(lower_bound, branch.value())
            
    ################################
           
    best_branch = None
    for branch in branches:
        if best_branch is None:
            best_branch = branch
        elif branch.value() < best_branch.value():
            best_branch = branch
        elif (branch.value() == best_branch.value()) and (np.sum(branch.alloc) > np.sum(best_branch.alloc)):
            best_branch = branch
            
    best_alloc = best_branch.alloc[::-1]
    return best_alloc

############################



















