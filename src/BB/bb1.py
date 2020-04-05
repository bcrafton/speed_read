
import numpy as np

############################

array = np.array([4, 20, 40, 72, 144, 288])
narray = 2048
nlayer = 6

nmac = np.array([1769472, 37748736, 18874368, 37748736, 18874368, 37748736])
mac = array / np.array([0.085, 0.19, 0.145, 0.13, 0.12, 0.06])

dup = [1] * nlayer
ndup = narray // array

############################

class BB:
    # use the class specific stuff for that up there ^^ 

    def __init__(self, arrays):
        self.arrays = arrays
        
    def bound(self):
        return 0.
        
    def value(self):
        return 0.

    # you can have branch() return:
    # 1: all the possible (+1) branches
    # 2: just the branch for a single item.
           
    def branch(self, array):
        new_arrays = copy.copy(self.arrays)    
        new_arrays.append(array)
        return BB(new_arrays)

############################















