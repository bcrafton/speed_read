
import numpy as np
import copy

############################

factor = np.array([4, 20, 40, 72, 144, 288])
narray = 4096
nlayer = 6

nmac = np.array([1769472, 37748736, 18874368, 37748736, 18874368, 37748736])
ndup = narray // factor

############################

array_density = np.array([0.41, 0.19, 0.145, 0.13, 0.12, 0.06])
row_per_array = np.ceil(128 * array_density / 8) * 8
mac_per_array = ((128 / row_per_array * 16) / 8)
cycles = np.sum(nmac / mac_per_array) / narray

############################

remainder = narray
dup = [1] * nlayer

for n in range(nlayer-1, -1, -1):
    need = np.ceil(nmac[n] / mac_per_array[n] / cycles)
    need = need - (need % (factor[n]))    
    assert (remainder > factor[n])
    dup[n] = min(need, remainder)
    remainder -= dup[n]
    assert (remainder >= 0)

# print (dup)

############################

# so this will give us:
# upper bound
# lower bound

array_density = np.array([0.41, 0.19, 0.145, 0.13, 0.12, 0.06])
row_per_array = np.ceil(128 * array_density / 8) * 8
mac_per_array = ((128 / row_per_array * 16) / 8)
cycles = np.sum(nmac / mac_per_array) / narray

############################

class BB:
    nmac = np.array([1769472, 37748736, 18874368, 37748736, 18874368, 37748736])
    array_density = np.array([0.41, 0.19, 0.145, 0.13, 0.12, 0.06])

    def __init__(self, narray):
        self.narray = narray
        
    def bound(self):
        # theoretical upper bound
        layer = len(self.narray)
        remainder = narray - np.sum(self.narray)
        row_per_array = np.ceil(128 * BB.array_density / 8) * 8
        mac_per_array = ((128 / row_per_array * 16) / 8)
        
        if layer == 0:
            min_cycle = np.sum(nmac / mac_per_array) / remainder
        elif layer == nlayer:
            min_cycle = np.max(nmac / mac_per_array / self.narray)
        else:
            actual = np.max(nmac[0:layer] / mac_per_array[0:layer] / self.narray)
            upper_bound = np.sum(nmac[layer:] / mac_per_array[layer:]) / remainder
            min_cycle = max(actual, upper_bound)
            
        return min_cycle
        
    def value(self):
        upper_bound = self.bound()
    
        new_narray = copy.copy(self.narray)
        remainder = narray - np.sum(self.narray)
        layer = len(self.narray)
        for n in range(layer, nlayer):
            need = np.ceil(nmac[n] / mac_per_array[n] / upper_bound)
            need = need - (need % (factor[n]))
            # assert (remainder > factor[n])
            new_narray.append(min(need, remainder))
            remainder -= new_narray[n]
            
        return np.max(nmac / mac_per_array / new_narray)

    def branch(self):
        layer = len(self.narray)        
        lower_bound = self.value()
        branches = []
        remainder = narray - np.sum(self.narray)
        for n in range(factor[layer], int(remainder), factor[layer]):
            new_narray = copy.copy(self.narray)
            new_narray.append(n)
            new_BB = BB(new_narray)
            new_bound = new_BB.bound()
            if new_bound < lower_bound:
                branches.append(new_BB)
            
        return branches

############################

def branch_and_bound(branches, lower_bound):
    new_branches = []
    for branch in branches:
        new_branches.extend(branch.branch())
        
    return new_branches
        
############################

bb = BB([])
bound = bb.bound()
value = bb.value()
# print (bound, value)

############################

print ("SECOND BRANCH")

branches = branch_and_bound([bb], value)
# print (len(branches))

for branch in branches:
    print (branch.narray, branch.value(), branch.bound())

############################

print ("THIRD BRANCH")

branches = branch_and_bound(branches, value)
# print (len(branches))

for branch in branches:
    print (branch.narray, branch.value(), branch.bound())

############################
'''
print ("FOURTH BRANCH")

branches = branch_and_bound(branches, value)
# print (len(branches))

for branch in branches:
    print (branch.narray, branch.value(), branch.bound())
'''
############################












