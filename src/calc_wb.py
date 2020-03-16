
import numpy as np

C = 256
BL = 128
bl = np.array(range(8 * C // BL))

wb = ((bl + 1) * (BL // C)) - 1
print (wb)

wb = (bl // (C // BL)) 
print (wb)

################################

C = 1024
BL = 128
bl = np.array(range(8 * C // BL))

wb = ((bl + 1) * (BL // C)) - 1
print (wb)

wb = (bl // (C // BL)) 
print (wb)
