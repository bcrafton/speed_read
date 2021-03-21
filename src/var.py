
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def lut_var_dyn(var, states):
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

# https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p

def lut_var(ratio, lrs, hrs, max_rpr):
    lut = np.zeros(shape=(max_rpr + 1, max_rpr + 1, 1001), dtype=np.float32)
    var1 = lrs
    var2 = hrs / ratio
    for wl in range(1, max_rpr + 1):
        for on in range(0, wl + 1):
            off = wl - on
            var = on*var1**2 + off*var2**2
            assert(var > 0)
            std = np.sqrt(var)
            
            '''
            q = norm.cdf(np.linspace(-3, 3, 1001), loc=0, scale=1)
            A = norm.ppf(q=q, loc=0., scale=std)
            '''

            minval = norm.cdf(-3, loc=0, scale=1)
            maxval = norm.cdf( 3, loc=0, scale=1)
            B = norm.ppf(q=np.linspace(minval, maxval, 1001), loc=0., scale=std)
            
            '''
            # B yields higher expected value than what we want.
            print (np.mean(np.absolute(B)))
            normal = np.absolute(np.random.normal(loc=0., scale=std, size=1000000))
            normal = normal[np.where(normal <= 3 * std)]
            print (np.mean(normal))
            assert (False)
            '''

            '''
            # plot to see why B is correct.
            plt.scatter(np.arange(1001), np.abs(A), label='A')
            plt.scatter(np.arange(1001), np.abs(B), label='B')
            plt.legend()
            plt.show()

            print (A)
            print (B)
            print (A - B)
            assert (np.allclose(A, B))
            assert (False)
            '''

            lut[wl, on, :] = B

    return lut
    
    
    
    
    
    
    
