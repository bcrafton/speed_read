
import numpy as np
import copy

############################

array = np.array([4, 20, 40, 72, 144, 288])
narray = 2048
nlayer = 6

nmac = np.array([1769472, 37748736, 18874368, 37748736, 18874368, 37748736])
mac = array / np.array([0.085, 0.19, 0.145, 0.13, 0.12, 0.06])

ndup = narray // array

############################

class BB:
    # use the class specific stuff for that up there ^^ 

    def __init__(self, narray):
        self.narray = narray
        
    def bound(self):
        return 0.
        
    def value(self):
        return 0.

    def branch(self):
        layer = len(self.narray)
        
        branches = []
        for n in range(ndup[layer]):
            new_narray = copy.copy(self.narray)    
            new_narray.append(n)
            new_BB = BB(new_narray)
            branches.append(new_BB)
            
        return branches

############################

bb = BB([])
bb = bb.branch()

print (len(bb))

# okay so now we have to:
# make it recursive/idk
# implement the bound function

############################








