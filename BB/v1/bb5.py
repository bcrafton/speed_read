
import numpy as np
import copy

############################

array = np.array([4, 20, 40, 72, 144, 288])
narray = 4096
nlayer = 6

nmac = np.array([1769472, 37748736, 18874368, 37748736, 18874368, 37748736])
mac = array / np.array([0.41, 0.19, 0.145, 0.13, 0.12, 0.06])

ndup = narray // array

############################

# you are supposed to use a greedy solution as lower bound.
# so should we write a greedy solution first ?
# yes.

############################

# loop -> get all close to max as possible.
# what is max performance ? 
# if everyone was divided perfectly.

array_density = np.array([0.41, 0.19, 0.145, 0.13, 0.12, 0.06])
row_per_array = np.ceil(128 * array_density / 8) * 8
mac_per_array = ((128 / row_per_array * 16) / 8)
cycles = np.sum(nmac / mac_per_array) / narray

# print (row_per_array)
# print (mac_per_array)
# print (cycles)

############################

remainder = narray
dup = [1] * nlayer

for n in range(nlayer-1, -1, -1):
    need = np.ceil(nmac[n] / mac_per_array[n] / cycles)
    need = need - (need % (array[n]))    
    assert (remainder > array[n])
    dup[n] = min(need, remainder)
    remainder -= dup[n]
    assert (remainder >= 0)

print (dup)

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
            upper_bound = np.sum(nmac / mac_per_array) / remainder
            min_cycle = max(actual, upper_bound)

        return min_cycle
        
    def value(self):
        # greedy lower bound 
        return 0.

    def branch(self):
        layer = len(self.narray)
        
        branches = []
        for n in range(1, ndup[layer]):
            new_narray = copy.copy(self.narray)    
            new_narray.append(n)
            new_BB = BB(new_narray)
            branches.append(new_BB)
            
        return branches

############################

bb = BB([])
bound = bb.bound()
# print (bound)

branches = bb.branch()
for branch in branches:
    print (branch.bound())
























