
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

samples = 1000000
sigma = 0.11
states = 32
on_states = 1
lut = lut_var(sigma, states)

##############################

var1 = []
for i in range(samples):
   r = np.random.randint(0, 1000)
   var1.append(lut[on_states][r]) 

print (np.mean(np.array(var1) == 2))
print (np.mean(np.array(var1) == 1))
print (np.mean(np.array(var1) == 0))
print (np.mean(np.array(var1) == -1))
print (np.mean(np.array(var1) == -2))
print ()

##############################

var2 = np.random.normal(loc=0., scale=sigma * np.sqrt(on_states), size=samples)
var2 = np.around(var2)

print (np.mean(np.array(var2) == 2))
print (np.mean(np.array(var2) == 1))
print (np.mean(np.array(var2) == 0))
print (np.mean(np.array(var2) == -1))
print (np.mean(np.array(var2) == -2))
print ()

##############################

print (np.mean(var1), np.mean(var2))
print (np.std(var1), np.std(var2))
print ()

##############################













