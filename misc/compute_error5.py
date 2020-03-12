
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

    # zero case:
    psum += ((e - 0.5 < 0) * (0 < e + 0.5)) * binom.pmf(0, rpr, p)
    return psum
    
#####################

rpr = 10
adc = 8
sigma = 0.2
reads = 16
p = 0.8

s = np.array(range(-16, 16))
pe = prob_err(s, p, sigma, adc, rpr)
mu = np.sum(pe * s)
var = np.sum(pe * (s - mu) ** 2)

# extend to 16 reads.
sigma = np.sqrt(var * reads)
mu = mu * reads
print (mu, sigma)

#####################

s = np.array(range(-256, 256))
# plt.plot(s, norm.pdf(s, mu, sigma))
# plt.show()

#####################







