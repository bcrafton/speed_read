
import math
import numpy as np
from scipy.stats import norm, binom
import matplotlib.pyplot as plt
    
#####################

def prob_err(e, p, var, adc, rpr):
    psum = 0
    for s in range(1, rpr + 1):
        bin = binom.pmf(s, rpr, p)
        psum += ((s + e) < adc) * bin * (norm.cdf(e + 0.5, 0, var * np.sqrt(s)) - norm.cdf(e - 0.5, 0, var * np.sqrt(s)))
        psum += ((s + e) == adc) * bin * (1 - norm.cdf(adc - s - 0.5, 0, var * np.sqrt(s)))

    # The PDF for normal becomes dirac delta when s = 0, so prob. over some range is 1 as long as range includes 0:
    psum += ((e - 0.5 < 0) * (0 < e + 0.5)) * binom.pmf(0, rpr, p)
    return psum
    
#####################

rpr = 10
adc = 8
col_density = 0.8
sigma = 0.2

p = np.array([0.4, 0.5, 0.6, 0.7, 0.8]).reshape(-1, 1, 1)
s = np.array(range(-4, 4)).reshape(-1, 1)
e = prob_err(s, p, sigma, adc, rpr)

print (np.shape(p), np.shape(s), np.shape(e))

plt.plot(s, e[4])
plt.show()

#####################







