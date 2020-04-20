
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
            if remainder < 1: return np.inf
            actual = np.max(self.nmac[0:layer] / self.mac_per_array[0:layer] / self.alloc)
            upper_bound = np.sum(self.nmac[layer:] / self.mac_per_array[layer:]) / remainder
            min_cycle = max(actual, upper_bound)
            
        return min_cycle
        
    def value(self, p=False):
        upper_bound = self.bound()
        layer = len(self.alloc)
        if (layer < self.nlayer):
            remainder = self.narray - np.sum(self.alloc)

            target = self.nmac[layer:] / self.mac_per_array[layer:] / upper_bound
            alloc_floor = np.clip(target // self.factor[layer:], 1, np.inf) * self.factor[layer:]
            if np.any(remainder < np.sum(alloc_floor)): return np.inf

            alloc = alloc_floor
            cycles = self.nmac[layer:] / self.mac_per_array[layer:] / alloc
            argmin = np.argmin(cycles)
            while (np.sum(alloc) + self.factor[argmin]) < remainder:
                alloc[argmin] += self.factor[argmin]
                cycles = self.nmac[layer:] / self.mac_per_array[layer:] / alloc
                argmin = np.argmin(cycles)

            new_alloc = np.concatenate((self.alloc, alloc))
        else:
            new_alloc = self.alloc

        if p: print (new_alloc)

        assert (np.all(np.absolute(self.mac_per_array) > 0))
        assert (np.all(np.absolute(new_alloc) > 0))
        assert (np.sum(new_alloc) <= self.narray)
        max_cycle = np.max(self.nmac / self.mac_per_array / new_alloc)
        return max_cycle

    def branch(self, lower_bound):
        layer = len(self.alloc)        
        lower_bound = min(lower_bound, self.value())
        branches = []
        remainder = self.narray - np.sum(self.alloc)
        for n in range(self.factor[layer], int(remainder) + 1, self.factor[layer]): # do NOT forget (+1)
            new_alloc = copy.copy(self.alloc)
            new_alloc.append(n)
            new_BB = BB(self.narray, self.nlayer, new_alloc, self.mac_per_array, self.nmac, self.factor, self.params)
            new_bound = new_BB.bound()
            if new_bound <= lower_bound:
                branches.append(new_BB)
        
        return branches

############################

def branch_and_bound(narray, nmac, factor, mac_per_array, params):

    nlayer = len(nmac)

    ################################
    
    mac_per_array = mac_per_array[::-1]
    nmac = nmac[::-1]
    factor = factor[::-1]
    
    ################################

    def branch_and_bound_help(branches, lower_bound):
        new_branches = []
        for branch in branches:
            next_branches = branch.branch(lower_bound)
            new_branches.extend(next_branches)
            
        # thing I worry about here is losing all path diversity.
        if len(new_branches) > 5:
            new_bounds = []
            for new_branch in new_branches:
                new_bounds.append(new_branch.bound())
            new_bounds = np.array(new_bounds)
            order1 = np.argsort(new_bounds)[0:5]
            new_branches1 = [new_branches[i] for i in order1]

            '''
            new_values = []
            for new_branch in new_branches:
                new_values.append(new_branch.value())
            new_values = np.array(new_values)
            order2 = np.argsort(new_values)[0:25]
            new_branches2 = [new_branches[i] for i in order2]
            '''

            new_branches = []
            new_branches.extend(new_branches1)
            # new_branches.extend(new_branches2)

        return new_branches

    ################################

    root = BB(narray, nlayer, [], mac_per_array, nmac, factor, params)
    branches = [root]
    # lower_bound = 3300
    lower_bound = root.value(p=True)
    for layer in range(nlayer):
        print (layer, lower_bound, len(branches))
        branches = branch_and_bound_help(branches, lower_bound)
        
        print (lower_bound, branches[0].value(), branches[0].bound(), branches[0].alloc)
        # branch.value(p=True)
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



















