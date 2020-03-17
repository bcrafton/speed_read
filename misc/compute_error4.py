
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

rpr = 16
adc = 8
sigma = 0.2
reads = 8

ps = np.array([0.4, 0.5, 0.6, 0.7, 0.8]).reshape(-1, 1)
s = np.array(range(-16, 16))
pe = prob_err(s, ps, sigma, adc, rpr)
mu = np.sum(pe * s, axis=1)

sigma = np.sqrt(np.sum((pe * s - mu.reshape(-1, 1)) ** 2, axis=1))

print (mu[4], sigma[4])

plt.plot(s, norm.pdf(s, mu[4], sigma[4]))
plt.show()

#####################






