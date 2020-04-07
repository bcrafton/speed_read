
import numpy as np
import copy

############################

factor = np.array([288, 144, 72, 40, 20, 4])
narray = 4096
nlayer = 6

nmac = np.array([37748736, 18874368, 37748736, 18874368, 37748736, 1769472])
ndup = narray // factor

############################

array_density = np.array([0.06, 0.12, 0.13, 0.145, 0.19, 0.41])
row_per_array = np.ceil(128 * array_density / 8) * 8
mac_per_array = ((128 / row_per_array * 16) / 8)
cycles = np.sum(nmac / mac_per_array) / narray

############################

class BB:
    def __init__(self, narray):
        self.narray = narray
        
    def bound(self):
        # theoretical upper bound
        layer = len(self.narray)
        remainder = narray - np.sum(self.narray)
        row_per_array = np.ceil(128 * array_density / 8) * 8
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

    def branch(self, lower_bound):
        layer = len(self.narray)        
        lower_bound = min(lower_bound, self.value())
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
        new_branches.extend(branch.branch(lower_bound))
        
    return new_branches
        
############################

bb = BB([])
bound = bb.bound()
value = bb.value()
# print (bound, value)

############################

print ("1 BRANCH")

branches = branch_and_bound([bb], value)
print (len(branches))
'''
for branch in branches:
    print (branch.narray, branch.value(), branch.bound())
'''
############################

print ("2 BRANCH")

branches = branch_and_bound(branches, value)
print (len(branches))
'''
for branch in branches:
    print (branch.narray, branch.value(), branch.bound())
'''
############################

print ("3 BRANCH")

branches = branch_and_bound(branches, value)
print (len(branches))
'''
for branch in branches:
    print (branch.narray, branch.value(), branch.bound())
'''
############################

print ("4 BRANCH")

branches = branch_and_bound(branches, value)
print (len(branches))
'''
for branch in branches:
    print (branch.narray, branch.value(), branch.bound())
'''

############################

print ("5 BRANCH")

branches = branch_and_bound(branches, value)
print (len(branches))
'''
for branch in branches:
    print (branch.narray, branch.value(), branch.bound())
'''

############################

print ("6 BRANCH")

branches = branch_and_bound(branches, value)
print (len(branches))

for branch in branches:
    print (np.sum(branch.narray), branch.narray, branch.value(), branch.bound())

############################





