
import numpy as np
from scipy.stats import norm

'''
def lut_var(var, states):
    lut = np.zeros(shape=(states + 1, 1000), dtype=np.int32)
    for s in range(1, states + 1):
        
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
'''

# https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p

def lut_var(var, states):
    lut = np.zeros(shape=(states + 1, 1000), dtype=np.int32)
    for s in range(1, states + 1):
        std = var * np.sqrt(s)
        # cannot do 0 or 1, these are np.inf.
        # norm.cdf([-3, -2, -1, 0, 1, 2, 3], loc=0, scale=1)
        minval = norm.cdf(-3, loc=0, scale=1)
        maxval = norm.cdf(3, loc=0, scale=1)
        lut[s, :] = norm.ppf(q=np.linspace(minval, maxval, 1000), loc=0., scale=std)
        
    return lut
    
    
    
    
    
    
    
