
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def lut_var(var, rpr):
    lut = np.zeros(shape=(rpr, 1000), dtype=np.int32)
    for s in range(1, rpr):
        
        std = var * np.sqrt(s)
        
        p5 = norm.cdf( 5.5, 0, std) - norm.cdf( 4.5, 0, std)
        p4 = norm.cdf( 4.5, 0, std) - norm.cdf( 3.5, 0, std)
        p3 = norm.cdf( 3.5, 0, std) - norm.cdf( 2.5, 0, std)
        p2 = norm.cdf( 2.5, 0, std) - norm.cdf( 1.5, 0, std)
        p1 = norm.cdf( 1.5, 0, std) - norm.cdf( 0.5, 0, std)

        p5 = int(round(p5, 3) * 1000)
        p4 = int(round(p4, 3) * 1000)
        p3 = int(round(p3, 3) * 1000)
        p2 = int(round(p2, 3) * 1000)
        p1 = int(round(p1, 3) * 1000)
        p0 = 1000 - 2 * (p5 + p4 + p3 + p2 + p1)
        
        pos = [5]*p5 + [4]*p4 + [3]*p3 + [2]*p2 + [1]*p1
        neg = [-5]*p5 + [-4]*p4 + [-3]*p3 + [-2]*p2 + [-1]*p1
        e = pos + neg + [0]*p0
        lut[s, :] = e
        
    return lut
    
##############################
    
lut = lut_var(0.1, 10)

##############################

var = []
for i in range(100000):
   r = np.random.randint(0, 1000)
   var.append(lut[8][r]) 

# plt.hist(var)
# plt.show()

print (np.sum(np.array(var) == 2))
print (np.sum(np.array(var) == 1))
print (np.sum(np.array(var) == 0))
print (np.sum(np.array(var) == -1))
print (np.sum(np.array(var) == -2))

##############################
