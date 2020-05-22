
import numpy as np

# 2 bit, 128x128
rram_array = 0.0003

# 5 bit, 8 col / adc
adc = 0.0036

narray = 24960

area = (adc + rram_array) * narray

print (area)

################

# 512x512, 1 bit
rram_array = rram_array * 16 / 2

# 3 bit 512 vs 128 columns
adc = adc / 4 * 4 

narray = area / (adc + rram_array)

print (narray)
# ~16384
