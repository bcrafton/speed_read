
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







